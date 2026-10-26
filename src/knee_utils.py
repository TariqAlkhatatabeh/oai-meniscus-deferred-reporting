"""Helpers for knee-level tables."""
from __future__ import annotations

import pandas as pd


def dedupe_knee_rows(df: pd.DataFrame,
                     key_cols: tuple[str, str] = ("ID", "SIDE"),
                     priority_col: str = "label") -> pd.DataFrame:
    """Collapse duplicate knee rows to one record per ID/SIDE.

    Rows are ordered by the priority column descending before de-duplication,
    so a positive label wins over a negative one when conflicts exist.
    """
    if df.empty:
        return df.copy()

    work = df.copy()
    sort_cols = list(key_cols)
    asc = [True] * len(key_cols)
    if priority_col in work.columns:
        sort_cols.append(priority_col)
        asc.append(False)
    work = work.sort_values(sort_cols, ascending=asc, kind="mergesort")
    return work.drop_duplicates(subset=list(key_cols), keep="first") \
               .reset_index(drop=True)
