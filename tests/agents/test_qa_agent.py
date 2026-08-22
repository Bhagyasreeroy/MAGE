"""Tests for agents/qa_agent.py."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.mining_agent import MiningAgent
from agents.qa_agent import QAAgent


@pytest.fixture
def qa() -> QAAgent:
    return QAAgent()


@pytest.fixture
def mining_and_ingestion() -> tuple[dict, dict]:
    df = pd.DataFrame(
        {
            "order_id": range(1, 31),
            "region": ["East", "West", "North", "South"] * 7 + ["East", "West"],
            "units": [3, 5, 2, 7, 4, 100, 3, 5, 2, 7, 4, 3, 5, 2, 7, 4, 3, 5, 2, 7, 4, 3, 5, 2, 7, 4, 3, 5, 2, None],
            "revenue": [45, 120, 60, 210, 90, 45, 120, 60, 210, 90] * 3,
        }
    )
    mining_output = MiningAgent().run(context={"dataframe": df, "goal": "profile"})
    ingestion_output = {"row_count": len(df), "column_count": len(df.columns)}
    return mining_output, ingestion_output


class TestQAAgentNoMatch:
    def test_broad_goal_returns_none(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        assert qa.try_answer("summarize this dataset", mining, ingestion) is None

    def test_empty_context_returns_none(self, qa: QAAgent) -> None:
        assert qa.try_answer("which column has the most missing values", {}, {}) is None


class TestMissingValueQuestions:
    def test_most_missing_column(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("which column has the most missing values", mining, ingestion)
        assert answer is not None
        assert "units" in answer.text
        assert "1 missing" in answer.text

    def test_alternate_phrasing(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("give column with the most missing values", mining, ingestion)
        assert answer is not None
        assert "units" in answer.text

    def test_no_missing_values_case(self, qa: QAAgent) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        mining = MiningAgent().run(context={"dataframe": df, "goal": "profile"})
        answer = qa.try_answer("which column has the most missing values", mining, {"row_count": 3})
        assert answer is not None
        assert "No column" in answer.text


class TestImputationAdvice:
    def test_general_imputation_question(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("what form of imputation should I do", mining, ingestion)
        assert answer is not None
        assert "units" in answer.text
        assert answer.source == "knowledge_base/missing_values.md"

    def test_recommends_median_for_skewed_column(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("how should I handle missing values", mining, ingestion)
        assert answer is not None
        assert "median" in answer.text.lower()

    def test_impute_is_recognized_not_just_imputation(self, qa: QAAgent, mining_and_ingestion) -> None:
        """A real user wrote 'best way to impute units' — 'impute' doesn't
        contain 'imputat' as a substring, so this fell through the QA
        short-circuit entirely and landed in the generic RAG path, which had
        nothing goal-specific to say about it."""
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("best way to impute units", mining, ingestion)
        assert answer is not None
        assert "units" in answer.text

    def test_do_about_missing_phrasing(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("what should I do about missing units values", mining, ingestion)
        assert answer is not None
        assert "units" in answer.text


class TestRowColumnCounts:
    def test_row_count(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("how many rows are there", mining, ingestion)
        assert answer is not None
        assert "30" in answer.text

    def test_column_count(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("how many columns does this have", mining, ingestion)
        assert answer is not None
        assert "4" in answer.text


class TestCorrelationQuestions:
    def test_correlation_between_named_columns(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("what is the correlation between units and revenue", mining, ingestion)
        assert answer is not None
        assert "units" in answer.text and "revenue" in answer.text
        assert "r=" in answer.text

    def test_no_match_without_two_columns(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        assert qa.try_answer("what is the correlation here", mining, ingestion) is None


class TestOutlierQuestions:
    def test_outliers_for_named_column(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("how many outliers in units", mining, ingestion)
        assert answer is not None
        assert "units" in answer.text

    def test_which_columns_have_outliers(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("which columns have outliers", mining, ingestion)
        assert answer is not None


class TestColumnTypeQuestions:
    def test_numeric_columns(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("which columns are numeric", mining, ingestion)
        assert answer is not None
        assert "revenue" in answer.text

    def test_categorical_columns(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("which columns are categorical", mining, ingestion)
        assert answer is not None
        assert "region" in answer.text


class TestSummaryStatQuestions:
    def test_mean_of_column(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("what is the mean of revenue", mining, ingestion)
        assert answer is not None
        assert "revenue" in answer.text

    def test_non_numeric_column_returns_explanation(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("what is the mean of region", mining, ingestion)
        assert answer is not None
        assert "isn't numeric" in answer.text

    def test_highest_is_recognized_as_max(self, qa: QAAgent, mining_and_ingestion) -> None:
        """'highest' is the everyday phrasing a real user reaches for — not
        recognizing it as 'max' silently fell through to the RAG path, which
        (having nothing goal-specific to say) produced near-duplicate
        recommendations across unrelated follow-up questions."""
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("what is the highest revenue", mining, ingestion)
        assert answer is not None
        assert "revenue" in answer.text
        assert f"{mining['statistics']['revenue']['max']:g}" in answer.text.replace(",", "")

    def test_lowest_is_recognized_as_min(self, qa: QAAgent, mining_and_ingestion) -> None:
        mining, ingestion = mining_and_ingestion
        answer = qa.try_answer("what is the lowest revenue", mining, ingestion)
        assert answer is not None
        assert f"{mining['statistics']['revenue']['min']:g}" in answer.text.replace(",", "")

    def test_large_value_is_not_scientific_notation(self, qa: QAAgent) -> None:
        mining = {"statistics": {"total_gross": {"type": "numeric", "max": 61200.0}}}
        answer = qa.try_answer("what is the highest total_gross", mining, {})
        assert answer is not None
        assert "e+" not in answer.text
        assert "61,200" in answer.text
