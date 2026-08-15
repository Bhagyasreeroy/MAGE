"""
tests/backend/test_transform_service.py
─────────────────────────────────────────
Integration tests for backend/services/transform_service.py — real
Postgres, real users, real Dataset rows. Each test creates its own user
directly via the User model (lower-level than the HTTP-flow tests) since
this exercises the service layer, not the router.
"""

from __future__ import annotations

import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.core.database import async_session
from backend.models.user import User
from backend.services import dataset_service, transform_service

SAMPLE_CSV = (
    b"id,region,revenue,units\n"
    b"1,East,100.0,3\n"
    b"2,West,,5\n"
    b"3,East,300.0,2\n"
    b"4,East,400.0,7\n"
    b"5,West,400.0,7\n"
)


async def _make_user() -> str:
    async with async_session() as db:
        user = User(
            email=f"transform-{uuid.uuid4().hex[:12]}@example.com",
            full_name="Transform Test",
            hashed_password="x",
            is_active=True,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user.id


async def _make_dataset(user_id: str, filename: str = "sales.csv", content: bytes = SAMPLE_CSV):
    async with async_session() as db:
        return await dataset_service.save_dataset(db, user_id, filename, content)


class _FakeLLMClient:
    """Duck-types GeminiClient for tests — no network, no API key."""

    def __init__(self, configured: bool = True, response: str = "SELECT * FROM df", error: Exception | None = None) -> None:
        self._configured = configured
        self._response = response
        self._error = error
        self.last_prompt: str | None = None

    @property
    def is_configured(self) -> bool:
        return self._configured

    def generate(self, prompt: str, timeout: float = 20.0) -> str:
        self.last_prompt = prompt
        if self._error is not None:
            raise self._error
        return self._response


class TestApplyTransform:
    @pytest.mark.asyncio
    async def test_drop_columns_produces_new_version(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)

        async with async_session() as db:
            new_ds, report = await transform_service.apply_transform(
                db, user_id, dataset.id, [{"type": "drop_columns", "columns": ["units"]}]
            )

        assert new_ds.id != dataset.id
        assert new_ds.parent_id == dataset.id
        assert new_ds.root_id == dataset.root_id == dataset.id
        assert new_ds.version == 2
        assert new_ds.transform_type == "clean"
        assert new_ds.filename == "sales.parquet"
        assert "Dropped 1 column(s)" in report[0]

        async with async_session() as db:
            df = await transform_service.load_dataframe(db, user_id, new_ds.id)
        assert "units" not in df.columns
        assert len(df) == 5

    @pytest.mark.asyncio
    async def test_chained_transforms_increment_version(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)

        async with async_session() as db:
            v2, _ = await transform_service.apply_transform(
                db, user_id, dataset.id, [{"type": "fill_missing", "column": "revenue", "strategy": "mean"}]
            )
        async with async_session() as db:
            v3, _ = await transform_service.apply_transform(
                db, user_id, v2.id, [{"type": "dedupe", "subset": ["units"]}]
            )

        assert v3.version == 3
        assert v3.root_id == dataset.id
        assert v3.parent_id == v2.id

    @pytest.mark.asyncio
    async def test_nonexistent_dataset_raises(self) -> None:
        user_id = await _make_user()
        async with async_session() as db:
            with pytest.raises(transform_service.TransformNotFoundError):
                await transform_service.apply_transform(db, user_id, "does-not-exist", [{"type": "dedupe"}])

    @pytest.mark.asyncio
    async def test_other_users_dataset_raises(self) -> None:
        owner_id = await _make_user()
        other_id = await _make_user()
        dataset = await _make_dataset(owner_id)
        async with async_session() as db:
            with pytest.raises(transform_service.TransformNotFoundError):
                await transform_service.apply_transform(
                    db, other_id, dataset.id, [{"type": "drop_columns", "columns": ["units"]}]
                )


class TestQuery:
    @pytest.mark.asyncio
    async def test_basic_select(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        async with async_session() as db:
            result = await transform_service.run_query(
                db, user_id, dataset.id, "SELECT region, revenue FROM df WHERE region = 'East'"
            )
        assert list(result.columns) == ["region", "revenue"]
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_rejects_non_select(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        async with async_session() as db:
            with pytest.raises(transform_service.QueryValidationError):
                await transform_service.run_query(db, user_id, dataset.id, "DROP TABLE df")

    @pytest.mark.asyncio
    async def test_rejects_stacked_statements(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        async with async_session() as db:
            with pytest.raises(transform_service.QueryValidationError):
                await transform_service.run_query(
                    db, user_id, dataset.id, "SELECT 1; DROP TABLE df;"
                )

    @pytest.mark.asyncio
    async def test_blocks_filesystem_access(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        async with async_session() as db:
            with pytest.raises(transform_service.QueryValidationError):
                await transform_service.run_query(
                    db, user_id, dataset.id, "SELECT * FROM read_csv('/etc/passwd')"
                )

    @pytest.mark.asyncio
    async def test_query_does_not_persist(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        async with async_session() as db:
            await transform_service.run_query(db, user_id, dataset.id, "SELECT * FROM df")
        async with async_session() as db:
            versions = await dataset_service.list_versions(db, user_id, dataset.root_id)
        assert len(versions) == 1


class TestSaveQueryResult:
    @pytest.mark.asyncio
    async def test_persists_new_version(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        async with async_session() as db:
            new_ds, row_count = await transform_service.save_query_result(
                db, user_id, dataset.id, "SELECT * FROM df WHERE region = 'East'"
            )
        assert row_count == 3
        assert new_ds.transform_type == "query_save"
        assert new_ds.transform_params["sql"] == "SELECT * FROM df WHERE region = 'East'"
        assert new_ds.parent_id == dataset.id
        assert new_ds.version == 2

        async with async_session() as db:
            versions = await dataset_service.list_versions(db, user_id, dataset.root_id)
        assert [v.version for v in versions] == [1, 2]


class TestListVersions:
    @pytest.mark.asyncio
    async def test_fresh_upload_is_its_own_root(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        assert dataset.root_id == dataset.id
        assert dataset.version == 1
        assert dataset.parent_id is None

        async with async_session() as db:
            versions = await dataset_service.list_versions(db, user_id, dataset.root_id)
        assert len(versions) == 1
        assert versions[0].id == dataset.id


class TestGenerateSqlFromQuestion:
    """The LLM only ever produces SQL text — that text is subject to the
    exact same validation/sandboxed execution as hand-typed SQL. These
    tests prove that claim rather than just asserting it."""

    @pytest.mark.asyncio
    async def test_successful_translation_and_execution(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        fake = _FakeLLMClient(response="SELECT region, revenue FROM df WHERE region = 'East'")

        async with async_session() as db:
            sql, result = await transform_service.generate_sql_from_question(
                db, user_id, dataset.id, "show me east region revenue", llm_client=fake
            )

        assert sql == "SELECT region, revenue FROM df WHERE region = 'East'"
        assert list(result.columns) == ["region", "revenue"]
        assert len(result) == 3
        assert "region" in fake.last_prompt and "revenue" in fake.last_prompt

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        fake = _FakeLLMClient(response="```sql\nSELECT * FROM df\n```")

        async with async_session() as db:
            sql, result = await transform_service.generate_sql_from_question(
                db, user_id, dataset.id, "show me everything", llm_client=fake
            )

        assert sql == "SELECT * FROM df"
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_non_sql_response_raises_validation_error(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        fake = _FakeLLMClient(response="I'm not sure how to answer that question.")

        async with async_session() as db:
            with pytest.raises(transform_service.QueryValidationError):
                await transform_service.generate_sql_from_question(
                    db, user_id, dataset.id, "what is the meaning of life", llm_client=fake
                )

    @pytest.mark.asyncio
    async def test_unconfigured_client_raises_helpful_error(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        fake = _FakeLLMClient(configured=False)

        async with async_session() as db:
            with pytest.raises(transform_service.QueryValidationError, match="API key"):
                await transform_service.generate_sql_from_question(
                    db, user_id, dataset.id, "show me everything", llm_client=fake
                )

    @pytest.mark.asyncio
    async def test_malicious_generated_sql_is_still_blocked(self) -> None:
        # Proves there's no new trust boundary: even if the LLM produced a
        # filesystem-reading query, the same sandboxed DuckDB execution
        # (enable_external_access=False) that blocks hand-typed attempts
        # blocks this too.
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        fake = _FakeLLMClient(response="SELECT * FROM read_csv('/etc/passwd')")

        async with async_session() as db:
            with pytest.raises(transform_service.QueryValidationError):
                await transform_service.generate_sql_from_question(
                    db, user_id, dataset.id, "read the passwd file", llm_client=fake
                )

    @pytest.mark.asyncio
    async def test_stacked_statements_still_blocked(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        fake = _FakeLLMClient(response="SELECT 1; DROP TABLE df;")

        async with async_session() as db:
            with pytest.raises(transform_service.QueryValidationError):
                await transform_service.generate_sql_from_question(
                    db, user_id, dataset.id, "do something bad", llm_client=fake
                )

    @pytest.mark.asyncio
    async def test_does_not_persist(self) -> None:
        user_id = await _make_user()
        dataset = await _make_dataset(user_id)
        fake = _FakeLLMClient(response="SELECT * FROM df")

        async with async_session() as db:
            await transform_service.generate_sql_from_question(
                db, user_id, dataset.id, "show me everything", llm_client=fake
            )
        async with async_session() as db:
            versions = await dataset_service.list_versions(db, user_id, dataset.root_id)
        assert len(versions) == 1
