"""
tests/agents/test_row_lookup.py
────────────────────────────────
Tests for QAAgent's row lookup — questions about the *contents* of the table.

Reported from the running app: with a Spotify dataset loaded, "tell me about
drake artist" was answered with a card about SHAP and LIME. Nothing upstream
could have answered it. Every statistic MiningAgent computes describes the
*shape* of the data, and the knowledge base holds EDA methodology, so no amount
of retrieval was ever going to find Drake. The answer has to come from the
table itself.

Two properties carry most of the weight here.

**It must not fire on a group.** "Which country has the most artists" names a
value in the data too, but a country is a group of rows and the question is an
aggregate over it — a different question, and one the Query tab answers. A
lookup that returned some arbitrary Colombian artist would be worse than
returning nothing.

**It must not outrank the handlers that already work.** It runs last, so a
question about the shape of the data is still answered as one even when some
cell value happens to appear in it.

Written test-first.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.qa_agent import MAX_ROW_FIELDS, QAAgent


@pytest.fixture
def qa() -> QAAgent:
    return QAAgent()


@pytest.fixture
def df() -> pd.DataFrame:
    """Shaped like the dataset that surfaced the bug: a unique name column, a
    handful of grouping columns, a year, and large floats."""
    return pd.DataFrame(
        {
            "Artist Name": ["Drake", "Taylor Swift", "Bad Bunny", "Ty Dolla $ign", "Ozuna"],
            "Country of Origin": ["Canada", "United States", "United States", "United States", "United States"],
            "Primary Genre": ["Hip-Hop", "Pop", "Reggaeton", "R&B", "Reggaeton"],
            "Debut Year": [2006, 2006, 2013, 2011, 2012],
            "Total Streams (in millions)": [137492.1, 127861.0, 125899.8, 26465.2, 47649.2],
            "Verified": [True, True, True, False, True],
        }
    )


@pytest.fixture
def computed() -> tuple[dict, dict]:
    """Stand-ins for MiningAgent / IngestionAgent output, so the established
    handlers are live and can compete for the question."""
    mining = {
        "statistics": {"Debut Year": {"type": "numeric", "mean": 2009.6}},
        "data_quality": {"Debut Year": {"missing_count": 0, "completeness_pct": 100.0}},
    }
    return mining, {"row_count": 5, "column_count": 6}


class TestItFindsTheRow:
    @pytest.mark.parametrize(
        "question",
        [
            "tell me about drake artist",
            "tell me about Drake",
            "who is Drake?",
            "Drake",
            "what genre is drake",
        ],
    )
    def test_any_phrasing_naming_the_record_finds_it(self, qa, df, question) -> None:
        """Matching is on the value, not on the phrasing — which is what keeps
        this from becoming another list of regexes to extend forever."""
        answer = qa.try_answer(question, {}, {}, dataframe=df)
        assert answer is not None
        assert "Drake" in answer.text

    def test_the_row_carries_its_other_fields(self, qa, df) -> None:
        answer = qa.try_answer("tell me about Drake", {}, {}, dataframe=df)
        assert "Canada" in answer.text
        assert "Hip-Hop" in answer.text

    def test_the_longest_name_wins(self, qa, df) -> None:
        """"Bad Bunny" and a hypothetical "Bunny" would both match; the phrase
        the user actually typed is the longer one."""
        answer = qa.try_answer("tell me about bad bunny", {}, {}, dataframe=df)
        assert "Bad Bunny" in answer.text
        assert "Reggaeton" in answer.text

    def test_punctuation_in_the_name_is_not_an_obstacle(self, qa, df) -> None:
        answer = qa.try_answer("tell me about ty dolla $ign", {}, {}, dataframe=df)
        assert answer is not None
        assert "Ty Dolla $ign" in answer.text

    def test_it_cites_nothing(self, qa, df) -> None:
        """A row is computed from the user's own data. There is no
        knowledge-base document behind it and it must not claim one."""
        assert qa.try_answer("tell me about Drake", {}, {}, dataframe=df).source is None


class TestItDeclinesWhatItCannotAnswer:
    @pytest.mark.parametrize(
        "question",
        [
            "which country has the most artists",
            "how many artists are from the United States",
            "which genre is most common",
        ],
    )
    def test_a_group_is_not_a_record(self, qa, df, question) -> None:
        """A country or a genre names many rows. Returning one of them would be
        worse than returning nothing."""
        assert qa.try_answer(question, {}, {}, dataframe=df) is None

    def test_a_name_that_is_not_in_the_data(self, qa, df) -> None:
        assert qa.try_answer("tell me about Michael Jackson", {}, {}, dataframe=df) is None

    def test_a_general_question(self, qa, df) -> None:
        assert qa.try_answer("what should I investigate about this dataset", {}, {}, dataframe=df) is None

    def test_without_the_dataframe_it_does_nothing(self, qa) -> None:
        """The handler is off, not broken, when the data isn't supplied."""
        assert qa.try_answer("tell me about Drake", {}, {}) is None

    def test_an_empty_dataframe_is_harmless(self, qa) -> None:
        assert qa.try_answer("tell me about Drake", {}, {}, dataframe=pd.DataFrame()) is None

    def test_a_table_with_no_identifier_column_does_nothing(self, qa) -> None:
        """Every column here is a group. Nothing in it names a single row."""
        grouped = pd.DataFrame({"country": ["Canada"] * 6, "genre": ["Pop", "Rock"] * 3})
        assert qa.try_answer("tell me about Canada", {}, {}, dataframe=grouped) is None


class TestEstablishedHandlersKeepPriority:
    """It runs last. A question about the shape of the data is answered as one
    even when a cell value happens to appear in it."""

    def test_a_row_count_question(self, qa, df, computed) -> None:
        mining, ingestion = computed
        answer = qa.try_answer("how many rows are there", mining, ingestion, dataframe=df)
        assert "5 row(s)" in answer.text

    def test_a_missing_values_question(self, qa, df, computed) -> None:
        mining, ingestion = computed
        answer = qa.try_answer("which column has the most missing values", mining, ingestion, dataframe=df)
        assert "missing" in answer.text.lower()

    def test_a_summary_stat_question(self, qa, df, computed) -> None:
        mining, ingestion = computed
        answer = qa.try_answer("what is the mean of Debut Year", mining, ingestion, dataframe=df)
        # The established handler answered, not the row lookup — that is the
        # property under test. (It renders the mean as "2.01e+03"; its `%.3g`
        # formatting predates this handler and is left alone here.)
        assert "mean of 'Debut Year'" in answer.text


class TestTheAnswerReadsWell:
    def test_a_year_is_not_thousands_separated(self, qa, df) -> None:
        answer = qa.try_answer("tell me about Drake", {}, {}, dataframe=df)
        assert "2006" in answer.text
        assert "2,006" not in answer.text

    def test_a_large_number_is(self, qa, df) -> None:
        answer = qa.try_answer("tell me about Drake", {}, {}, dataframe=df)
        assert "137,492.10" in answer.text

    def test_a_boolean_reads_as_a_word(self, qa, df) -> None:
        assert "yes" in qa.try_answer("tell me about Drake", {}, {}, dataframe=df).text

    def test_a_missing_value_is_shown_as_absent_not_as_nan(self, qa) -> None:
        sparse = pd.DataFrame({"name": ["Alpha", "Beta"], "score": [1.5, None]})
        answer = qa.try_answer("tell me about Beta", {}, {}, dataframe=sparse)
        assert "—" in answer.text
        assert "nan" not in answer.text.lower()

    def test_column_names_are_trimmed(self, qa) -> None:
        """CSV headers routinely carry stray whitespace; rendering it back
        looks like our mistake rather than the file's."""
        padded = pd.DataFrame({"name": ["Alpha", "Beta"], " Artist Type": ["Solo", "Group"]})
        answer = qa.try_answer("tell me about Alpha", {}, {}, dataframe=padded)
        assert "- Artist Type: Solo" in answer.text

    def test_a_very_wide_table_is_truncated(self, qa) -> None:
        wide = pd.DataFrame({"name": ["Alpha", "Beta"]})
        for i in range(MAX_ROW_FIELDS + 10):
            wide[f"col_{i}"] = [i, i]
        answer = qa.try_answer("tell me about Alpha", {}, {}, dataframe=wide)
        assert "more column(s)" in answer.text
        assert answer.text.count("\n- ") <= MAX_ROW_FIELDS + 1


class TestSummaryStatsUseTheSameFormatter:
    """
    `_column_summary_stat` used "%.3g", which rendered a median of 53,308.9 as
    "5.33e+04" — a correct answer that looks broken, and it was sitting on the
    clearest win the grounded path had over the LLM in a side-by-side probe.
    One formatter now serves every number this agent returns.
    """

    @pytest.fixture
    def mining(self) -> dict:
        return {
            "statistics": {"streams": {"type": "numeric", "median": 53308.9, "mean": 58458.04, "min": 0.00012}},
            "data_quality": {"streams": {"missing_count": 0, "completeness_pct": 100.0}},
        }

    def test_a_large_value_is_readable(self, qa, mining) -> None:
        answer = qa.try_answer("what is the median of streams", mining, {"row_count": 35})
        assert "53,308.90" in answer.text
        assert "e+" not in answer.text

    def test_a_small_value_keeps_its_precision(self, qa, mining) -> None:
        """The reason "%.3g" was there in the first place — a fixed 2dp would
        round this away to 0.00."""
        answer = qa.try_answer("what is the min of streams", mining, {"row_count": 35})
        assert "0.00012" in answer.text
