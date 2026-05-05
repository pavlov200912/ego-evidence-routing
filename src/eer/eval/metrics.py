"""Evaluation metrics for the EER ablation study.

Expected DataFrame columns:
    question_id, category, mode, tool, predicted, correct, is_correct

`mode` is either "replace" (auxiliary frames only) or "augment" (native video
+ auxiliary frames).
"""

from __future__ import annotations

import pandas as pd


def compute_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    """Overall accuracy per (tool, mode).

    Returns:
        DataFrame with columns [tool, mode, accuracy, n_questions].
    """
    return (
        results.groupby(["tool", "mode"])["is_correct"]
        .agg(accuracy="mean", n_questions="count")
        .reset_index()
    )


def compute_per_category_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    """Accuracy grouped by (tool, mode, category).

    Returns:
        DataFrame with columns [tool, mode, category, accuracy, n_questions].
    """
    return (
        results.groupby(["tool", "mode", "category"])["is_correct"]
        .agg(accuracy="mean", n_questions="count")
        .reset_index()
    )


def compute_oracle_routing(results: pd.DataFrame) -> pd.DataFrame:
    """Upper-bound accuracy when we always pick the best tool per question.

    Computed separately per mode. For each (mode, question_id), oracle is
    correct if any tool answered correctly.

    Returns:
        DataFrame with columns [mode, oracle, best_single_tool, best_single_accuracy].
    """
    rows = []
    for mode, grp in results.groupby("mode"):
        per_question = grp.groupby("question_id")["is_correct"].max()
        oracle_acc = float(per_question.mean())

        tool_accs = grp.groupby("tool")["is_correct"].mean()
        best_tool = str(tool_accs.idxmax())
        best_acc = float(tool_accs.max())

        rows.append({
            "mode": mode,
            "oracle": oracle_acc,
            "best_single_tool": best_tool,
            "best_single_accuracy": best_acc,
        })
    return pd.DataFrame(rows)


def compute_agreement(results: pd.DataFrame) -> pd.DataFrame:
    """Accuracy stratified by how many tools agree on the predicted answer.

    Computed per mode.

    Returns:
        DataFrame with columns [mode, n_tools_agree, accuracy, n_questions].
    """
    plurality = (
        results.groupby(["mode", "question_id", "predicted"])
        .size()
        .reset_index(name="count")
    )
    majority = (
        plurality.sort_values("count", ascending=False)
        .drop_duplicates(["mode", "question_id"])
        .rename(columns={"predicted": "plurality_answer", "count": "n_agree"})
    )

    merged = results.merge(majority[["mode", "question_id", "n_agree"]], on=["mode", "question_id"])

    return (
        merged.groupby(["mode", "n_agree"])["is_correct"]
        .agg(accuracy="mean", n_questions="count")
        .reset_index()
        .rename(columns={"n_agree": "n_tools_agree"})
    )
