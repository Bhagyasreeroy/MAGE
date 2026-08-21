"""
data_pipeline/ingestion.py
───────────────────────────
DataIngestionEngine — multi-source data loading and normalisation.

Sources supported:
    • Files / uploads  — CSV, TSV, JSON, Parquet, Excel (XLSX/XLS), PDF tables
    • Database URIs    — any SQLAlchemy-supported dialect (via load_from_database)
    • REST / HTTP URLs — JSON or delimited/tabular endpoints (via load_from_url)

``classify_source`` decides which path a caller's source string takes; the
IngestionAgent uses it to route without the caller having to specify a type.
"""

from __future__ import annotations

import csv
import re
import io
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from fastapi import UploadFile

logger = logging.getLogger(__name__)

# Delimiters the sniffer is allowed to consider, and the same list the
# post-failure fallback searches for. One definition, so the two cannot drift.
_CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]

# pandas reports ragged rows as e.g.
#   "Error tokenizing data. C error: Expected 3 fields in line 5, saw 4"
_RAGGED_ROW_RE = re.compile(
    r"Expected (\d+) fields in line (\d+), saw (\d+)", re.IGNORECASE
)


def _describe_parser_error(exc: Exception) -> str:
    """Turn a pandas tokenizing error into something a reader can act on.

    The common case by far is ragged rows, and pandas already knows exactly
    which line and how many fields it found — that detail just never reached
    the user, who saw a delimiter complaint about a file whose delimiter was
    fine. Anything unrecognised is passed through rather than guessed at.
    """
    match = _RAGGED_ROW_RE.search(str(exc))
    if match:
        expected, line, saw = match.group(1), match.group(2), match.group(3)
        return (
            f"Inconsistent row length: line {line} has {saw} fields, but the "
            f"header declares {expected}. Every row must have the same number "
            f"of fields as the header."
        )
    return f"Failed to parse delimited file: {exc}"



class IngestionError(Exception):
    """Custom exception raised for all non-recoverable data ingestion errors."""


class DataIngestionEngine:
    """
    Multi-source data loader that normalises CSV, TSV, JSON, Parquet, Excel
    (XLSX/XLS), and PDF-table files — plus SQL databases and REST endpoints —
    to Pandas DataFrames.
    """

    #: File formats this engine can load.
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (
        ".csv", ".tsv", ".json", ".parquet", ".xlsx", ".xls", ".pdf",
    )

    #: URL schemes routed to REST/HTTP ingestion.
    URL_SCHEMES: frozenset[str] = frozenset({"http", "https"})

    #: SQLAlchemy dialect schemes routed to database ingestion. Matched on the
    #: part before the first ``+`` or ``:`` (e.g. "postgresql+psycopg2" → "postgresql").
    DB_SCHEMES: frozenset[str] = frozenset(
        {"postgresql", "postgres", "mysql", "mariadb", "sqlite", "mssql", "oracle"}
    )

    @classmethod
    def classify_source(cls, source: Any) -> str:
        """
        Classify a source into "file", "url", or "database".

        Upload objects and filesystem paths are files; ``http(s)://`` strings are
        URLs; strings whose scheme is a known SQL dialect are databases. Anything
        else falls back to "file" (the loader then validates the extension).
        """
        if hasattr(source, "filename") and hasattr(source, "file"):
            return "file"
        if isinstance(source, Path):
            return "file"
        if isinstance(source, str):
            scheme = urlparse(source).scheme.lower()
            if scheme in cls.URL_SCHEMES:
                return "url"
            # "postgresql+psycopg2" → "postgresql"
            base_scheme = scheme.split("+", 1)[0]
            if base_scheme in cls.DB_SCHEMES:
                return "database"
        return "file"

    def load(self, source: str | Path | UploadFile, **kwargs: Any) -> pd.DataFrame:
        """
        Load tabular data from a supported source and return a Pandas DataFrame.

        Parameters
        ----------
        source : str | Path | UploadFile
            The file source to load. Can be a file path string, a Path object,
            or a FastAPI UploadFile.
        **kwargs
            Additional arguments passed to the underlying Pandas loader.

        Returns
        -------
        pd.DataFrame
            The parsed and loaded dataset as a Pandas DataFrame.

        Raises
        ------
        IngestionError
            If the file is not found, empty, has an unsupported extension,
            fails encoding checks, or fails delimiter/parse detection.
        """
        filename = ""
        content = b""

        if self._is_upload_file(source):
            filename = source.filename or ""
            try:
                source.file.seek(0)
                content = source.file.read()
                source.file.seek(0)
            except Exception as exc:
                raise IngestionError(f"Failed to read uploaded file content: {exc}") from exc
        else:
            file_path = Path(source)
            filename = file_path.name
            if not file_path.exists():
                raise IngestionError(f"File not found: {file_path}")
            if not file_path.is_file():
                raise IngestionError(f"Path is not a file: {file_path}")
            try:
                content = file_path.read_bytes()
            except Exception as exc:
                raise IngestionError(f"Failed to read file from path: {exc}") from exc

        if not content or len(content.strip()) == 0:
            raise IngestionError("Empty file: the provided dataset has no content.")

        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(self.SUPPORTED_EXTENSIONS)
            raise IngestionError(
                f"Unsupported extension '{ext}': supported types are {supported}."
            )

        if ext == ".csv":
            return self._load_csv(content, **kwargs)
        if ext == ".tsv":
            return self._load_csv(content, sep="\t", **kwargs)
        if ext == ".json":
            return self._load_json(content, **kwargs)
        if ext == ".parquet":
            return self._load_parquet(content, **kwargs)
        if ext in (".xlsx", ".xls"):
            return self._load_xlsx(content, **kwargs)
        if ext == ".pdf":
            return self._load_pdf(content, **kwargs)
        raise IngestionError(f"Unsupported file type: {ext}")

    def _load_csv(self, content: bytes, sep: str | None = None, **kwargs: Any) -> pd.DataFrame:
        """Decode, detect delimiter (unless forced), and parse CSV/TSV contents."""
        try:
            sample_text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestionError(f"Encoding error: failed to decode file as UTF-8: {exc}") from exc

        if not sample_text.strip():
            raise IngestionError("Empty file.")

        if sep is None:
            try:
                sample_lines = "\n".join(sample_text.splitlines()[:20])
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample_lines, delimiters=_CANDIDATE_DELIMITERS)
                sep = dialect.delimiter
            except csv.Error:
                # The sniffer failing does not mean the delimiter is the
                # problem. Two very different files land here:
                #
                #   * a legitimate single-column upload, which has no delimiter
                #     to find — not an error at all; and
                #   * a file with ragged rows, which *does* have a delimiter the
                #     sniffer cannot settle on because the field counts disagree.
                #
                # Reporting both as "Delimiter detection failure" pointed the
                # reader at the wrong thing in both directions.
                sep = self._delimiter_after_sniff_failure(sample_text)

        try:
            return pd.read_csv(io.StringIO(sample_text), sep=sep, **kwargs)
        except pd.errors.EmptyDataError as exc:
            raise IngestionError("Empty file.") from exc
        except pd.errors.ParserError as exc:
            raise IngestionError(_describe_parser_error(exc)) from exc
        except Exception as exc:
            raise IngestionError(f"Failed to parse delimited file: {exc}") from exc

    @staticmethod
    def _delimiter_after_sniff_failure(sample_text: str) -> str:
        """Choose a delimiter when `csv.Sniffer` could not.

        If **no** candidate delimiter occurs anywhere, the file genuinely has
        one column: any separator parses it correctly, so the first candidate
        is returned and pandas yields a single-column frame.

        Otherwise a delimiter is present and the sniffer failed for some other
        reason — ragged rows, overwhelmingly. The most frequent candidate is
        returned so pandas can produce its real, specific complaint, which
        `_describe_parser_error` then translates. Falling back to
        single-column here instead would be worse than the original error: it
        would turn a broken file into a silently wrong one.
        """
        counts = {d: sample_text.count(d) for d in _CANDIDATE_DELIMITERS}
        if not any(counts.values()):
            logger.info("No delimiter present; reading as a single-column file.")
            return _CANDIDATE_DELIMITERS[0]
        return max(counts, key=counts.get)

    def _load_json(self, content: bytes, **kwargs: Any) -> pd.DataFrame:
        """Parse JSON contents into a DataFrame."""
        try:
            return pd.read_json(io.BytesIO(content), **kwargs)
        except ValueError as exc:
            raise IngestionError(f"Failed to parse JSON: {exc}") from exc
        except Exception as exc:
            raise IngestionError(f"Failed to parse JSON: {exc}") from exc

    def _load_parquet(self, content: bytes, **kwargs: Any) -> pd.DataFrame:
        """Parse Parquet contents into a DataFrame."""
        try:
            return pd.read_parquet(io.BytesIO(content), **kwargs)
        except Exception as exc:
            raise IngestionError(f"Failed to parse Parquet: {exc}") from exc

    def _load_xlsx(self, content: bytes, **kwargs: Any) -> pd.DataFrame:
        """Parse Excel (XLSX/XLS) contents."""
        try:
            return pd.read_excel(io.BytesIO(content), engine="openpyxl", **kwargs)
        except ImportError as exc:
            raise IngestionError(
                f"Cannot read Excel file: a required library is missing ({exc}). "
                "Install the 'openpyxl' extra."
            ) from exc
        except Exception as exc:
            raise IngestionError(f"Failed to parse Excel file: {exc}") from exc

    def _load_pdf(self, content: bytes, *, page: int | None = None, table_index: int = 0, **kwargs: Any) -> pd.DataFrame:
        """
        Extract a table from a PDF using pdfplumber.

        Scans pages in order and collects every table found. By default returns
        the first table (``table_index=0``); pass ``page`` to restrict to a single
        1-indexed page. The first row of the chosen table is treated as the header.
        """
        try:
            import pdfplumber
        except ImportError as exc:
            raise IngestionError(
                "Cannot read PDF file: 'pdfplumber' is not installed. "
                "Install it (pip install pdfplumber) to enable PDF ingestion."
            ) from exc

        tables: list[list[list[str | None]]] = []
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = pdf.pages
                if page is not None:
                    if page < 1 or page > len(pages):
                        raise IngestionError(
                            f"PDF page {page} out of range (document has {len(pages)} page(s))."
                        )
                    pages = [pages[page - 1]]
                for pdf_page in pages:
                    for extracted in pdf_page.extract_tables() or []:
                        if extracted and len(extracted) >= 2:
                            tables.append(extracted)
        except IngestionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Failed to parse PDF: {exc}") from exc

        if not tables:
            raise IngestionError(
                "No tables found in PDF. Scanned/image PDFs require OCR, which is not yet supported."
            )
        if table_index < 0 or table_index >= len(tables):
            raise IngestionError(
                f"PDF table index {table_index} out of range ({len(tables)} table(s) found)."
            )

        table = tables[table_index]
        header = [str(h).strip() if h is not None else "" for h in table[0]]
        rows = table[1:]
        try:
            df = pd.DataFrame(rows, columns=header)
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Failed to build DataFrame from PDF table: {exc}") from exc

        # PDF cells arrive as strings; recover numeric dtypes where unambiguous.
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() == df[col].replace("", pd.NA).notna().sum() and converted.notna().any():
                df[col] = converted
        return df

    def load_from_url(self, url: str, *, format_hint: str | None = None, timeout: float = 30.0, **kwargs: Any) -> pd.DataFrame:
        """
        Fetch a REST/HTTP endpoint and parse its body into a DataFrame.

        The parser is chosen from ``format_hint`` if given, else the response
        Content-Type, else the URL path's file extension, defaulting to JSON.

        Parameters
        ----------
        url : str
            An ``http(s)://`` endpoint returning JSON or delimited/tabular data.
        format_hint : str, optional
            One of "json", "csv", "tsv", "parquet", "xlsx" to force the parser.
        timeout : float
            Request timeout in seconds.
        """
        try:
            import httpx
        except ImportError as exc:
            raise IngestionError(
                "Cannot fetch URL: 'httpx' is not installed."
            ) from exc

        try:
            response = httpx.get(url, timeout=timeout, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise IngestionError(
                f"REST source returned HTTP {exc.response.status_code} for {url}."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Failed to fetch REST source {url}: {exc}") from exc

        content = response.content
        if not content or not content.strip():
            raise IngestionError(f"REST source {url} returned an empty body.")

        fmt = (format_hint or self._infer_url_format(url, response.headers.get("content-type", ""))).lower()
        if fmt == "json":
            return self._load_json(content, **kwargs)
        if fmt == "csv":
            return self._load_csv(content, **kwargs)
        if fmt == "tsv":
            return self._load_csv(content, sep="\t", **kwargs)
        if fmt == "parquet":
            return self._load_parquet(content, **kwargs)
        if fmt in ("xlsx", "xls"):
            return self._load_xlsx(content, **kwargs)
        raise IngestionError(f"Unsupported REST response format '{fmt}' for {url}.")

    @staticmethod
    def _infer_url_format(url: str, content_type: str) -> str:
        """Infer a parser key from Content-Type, then URL suffix, defaulting to json."""
        ct = content_type.lower()
        if "json" in ct:
            return "json"
        if "csv" in ct:
            return "csv"
        if "tab-separated" in ct or "tsv" in ct:
            return "tsv"
        if "parquet" in ct:
            return "parquet"
        if "spreadsheet" in ct or "excel" in ct:
            return "xlsx"
        suffix = Path(urlparse(url).path).suffix.lower()
        return {
            ".json": "json", ".csv": "csv", ".tsv": "tsv",
            ".parquet": "parquet", ".xlsx": "xlsx", ".xls": "xlsx",
        }.get(suffix, "json")

    def load_from_database(
        self,
        uri: str,
        *,
        table: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load a table (or a custom query) from a SQL database via SQLAlchemy.

        Exactly one of ``table`` or ``query`` may be given. If neither is
        provided, the database is introspected: a single-table database is
        loaded automatically, otherwise an error lists the available tables so
        the caller can pick one. A ``#fragment`` on the URI is honoured as the
        table name (e.g. ``postgresql://…/db#customers``).

        Parameters
        ----------
        uri : str
            A SQLAlchemy connection URI (e.g. ``sqlite:///path.db``,
            ``postgresql://user:pass@host/db``).
        table : str, optional
            Table to load (``SELECT * FROM <table>``).
        query : str, optional
            A full SQL query to run instead of a table read.
        """
        if table and query:
            raise IngestionError("Provide either 'table' or 'query' for database ingestion, not both.")

        parsed = urlparse(uri)
        if not table and not query and parsed.fragment:
            table = parsed.fragment
            uri = uri.split("#", 1)[0]

        try:
            from sqlalchemy import create_engine, inspect, text
        except ImportError as exc:
            raise IngestionError(
                "Cannot read from database: 'sqlalchemy' is not installed."
            ) from exc

        try:
            engine = create_engine(uri)
        except Exception as exc:  # noqa: BLE001 - driver import / URI parse errors
            raise IngestionError(
                f"Failed to create database engine for URI (check the driver is installed): {exc}"
            ) from exc

        try:
            with engine.connect() as conn:
                if query:
                    df = pd.read_sql(text(query), conn, **kwargs)
                else:
                    if not table:
                        available = inspect(engine).get_table_names()
                        if len(available) == 1:
                            table = available[0]
                        elif not available:
                            raise IngestionError("Database has no tables to ingest.")
                        else:
                            raise IngestionError(
                                "Multiple tables found; specify one via 'table' or a URI "
                                f"'#table' fragment. Available: {sorted(available)}"
                            )
                    df = pd.read_sql_table(table, conn, **kwargs)
        except IngestionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Failed to read from database: {exc}") from exc
        finally:
            engine.dispose()

        return df

    def _is_upload_file(self, source: str | Path | UploadFile) -> bool:
        """Return True for FastAPI/Starlette upload objects without relying on class identity."""
        return hasattr(source, "filename") and hasattr(source, "file")

    # TODO: Add OCR-based ingestion for scanned/image PDFs (pytesseract / OCR.space).
    # TODO: Add Spark/Dask hooks for large datasets and out-of-core processing.
