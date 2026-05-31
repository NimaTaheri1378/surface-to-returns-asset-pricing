from __future__ import annotations

import pandas as pd


def validate_nonoverlapping_links(
    links: pd.DataFrame,
    entity_col: str,
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    """Return overlapping link rows; empty means the link history is clean for each entity."""
    if links.empty:
        return links.copy()
    frame = links[[entity_col, start_col, end_col]].copy()
    frame[start_col] = pd.to_datetime(frame[start_col])
    frame[end_col] = pd.to_datetime(frame[end_col]).fillna(pd.Timestamp("2099-12-31"))
    frame = frame.sort_values([entity_col, start_col, end_col])
    frame["prev_end"] = frame.groupby(entity_col)[end_col].shift()
    return frame[frame["prev_end"].notna() & (frame[start_col] <= frame["prev_end"])]
