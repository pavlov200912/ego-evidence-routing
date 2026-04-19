"""Evaluation metrics for the EER ablation study.

Expected DataFrame columns:
    question_id, category, prototype, tool,
    predicted, correct, log_prob, is_correct
"""

from __future__ import annotations

import pandas as pd


def compute_accuracy(results: pd.DataFrame) -> dict[str, float]:
    """Overall accuracy per tool.

    Args:
        results: Results DataFrame (see module docstring for columns).

    Returns:
        Dict mapping tool name → accuracy in [0, 1].
    """
    return (
        results.groupby("tool")["is_correct"]
        .mean()
        .to_dict()
    )


def compute_per_category_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    """Accuracy grouped by (tool, category).

    Args:
        results: Results DataFrame.

    Returns:
        DataFrame with columns [tool, category, accuracy, n_questions].
    """
    grouped = (
        results.groupby(["tool", "category"])["is_correct"]
        .agg(accuracy="mean", n_questions="count")
        .reset_index()
    )
    return grouped


def compute_per_prototype_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    """Accuracy grouped by (tool, prototype).

    Args:
        results: Results DataFrame.

    Returns:
        DataFrame with columns [tool, prototype, accuracy, n_questions].
    """
    grouped = (
        results.groupby(["tool", "prototype"])["is_correct"]
        .agg(accuracy="mean", n_questions="count")
        .reset_index()
    )
    return grouped


def compute_oracle_routing(results: pd.DataFrame) -> dict[str, float]:
    """Upper-bound accuracy when we always pick the best tool per question.

    For each question, oracle is correct if *any* tool answered correctly.

    Args:
        results: Results DataFrame.

    Returns:
        Dict with key ``"oracle"`` → accuracy float, plus
        ``"baseline"`` → accuracy of the worst single tool.
    """
    per_question = results.groupby("question_id")["is_correct"].max()
    oracle_acc = float(per_question.mean())

    tool_accs = compute_accuracy(results)
    baseline_acc = min(tool_accs.values()) if tool_accs else float("nan")

    return {"oracle": oracle_acc, "baseline": baseline_acc}


def compute_agreement(results: pd.DataFrame) -> pd.DataFrame:
    """Accuracy stratified by how many tools agree on the predicted answer.

    For each question, count the number of tools predicting the same
    letter as the plurality answer.  Compute accuracy per agreement level.

    Args:
        results: Results DataFrame.

    Returns:
        DataFrame with columns [n_tools_agree, accuracy, n_questions].
    """
    # Majority vote per question
    plurality = (
        results.groupby(["question_id", "predicted"])
        .size()
        .reset_index(name="count")
    )
    majority = (
        plurality.sort_values("count", ascending=False)
        .drop_duplicates("question_id")
        .rename(columns={"predicted": "plurality_answer", "count": "n_agree"})
    )

    merged = results.merge(majority[["question_id", "n_agree"]], on="question_id")

    agg = (
        merged.groupby("n_agree")["is_correct"]
        .agg(accuracy="mean", n_questions="count")
        .reset_index()
        .rename(columns={"n_agree": "n_tools_agree"})
    )
    return agg
