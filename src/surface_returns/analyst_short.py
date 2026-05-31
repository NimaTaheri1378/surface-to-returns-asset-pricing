from __future__ import annotations

import numpy as np
import pandas as pd

from surface_returns.trading_costs import standardize_ticker


def aggregate_ibes_estimates(statsum: pd.DataFrame) -> pd.DataFrame:
    if statsum.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "ibes_analyst_coverage",
                "ibes_forecast_dispersion",
                "ibes_revision_breadth",
                "ibes_mean_estimate",
            ]
        )
    frame = statsum.copy()
    frame["ticker"] = frame["ticker"].map(standardize_ticker)
    frame["statpers"] = pd.to_datetime(frame["statpers"])
    frame = frame[frame["ticker"].notna()].copy()
    for col in ["numest", "numup", "numdown", "meanest", "stdev"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "measure" in frame:
        frame = frame[frame["measure"].astype(str).str.upper().eq("EPS")]
    if "fpi" in frame:
        frame = frame[frame["fpi"].astype(str).isin(["1", "2", "6", "7"])]
    frame["date"] = frame["statpers"].dt.to_period("M").dt.to_timestamp()
    frame["ibes_forecast_dispersion"] = frame["stdev"] / frame["meanest"].abs().replace(0, np.nan)
    frame["ibes_revision_breadth"] = (frame["numup"].fillna(0.0) - frame["numdown"].fillna(0.0)) / frame[
        "numest"
    ].replace(0, np.nan)
    out = (
        frame.sort_values(["ticker", "date", "statpers"])
        .groupby(["ticker", "date"], as_index=False)
        .agg(
            ibes_analyst_coverage=("numest", "last"),
            ibes_forecast_dispersion=("ibes_forecast_dispersion", "last"),
            ibes_revision_breadth=("ibes_revision_breadth", "last"),
            ibes_mean_estimate=("meanest", "last"),
        )
    )
    return out


def expand_ibes_link_to_months(links: pd.DataFrame, panel_keys: pd.DataFrame) -> pd.DataFrame:
    keys = panel_keys[["permno", "date"]].drop_duplicates().copy()
    keys["date"] = pd.to_datetime(keys["date"]).dt.to_period("M").dt.to_timestamp()
    link = links.copy()
    link["ticker"] = link["ticker"].map(standardize_ticker)
    link["sdate"] = pd.to_datetime(link["sdate"], errors="coerce")
    link["edate"] = pd.to_datetime(link["edate"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    link = link[link["ticker"].notna() & link["permno"].notna()].copy()
    merged = keys.merge(link, on="permno", how="left")
    merged = merged[(merged["sdate"] <= merged["date"]) & (merged["date"] <= merged["edate"])]
    if "score" in merged:
        merged = merged.sort_values(["permno", "date", "score", "sdate"])
    return merged.drop_duplicates(["permno", "date"], keep="first")[["permno", "date", "ticker"]]


def merge_ibes_to_panel(panel: pd.DataFrame, ibes_monthly: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    mapping = expand_ibes_link_to_months(links, panel[["permno", "date"]])
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp()
    ibes = ibes_monthly.copy()
    ibes["date"] = pd.to_datetime(ibes["date"]).dt.to_period("M").dt.to_timestamp()
    return out.merge(mapping, on=["permno", "date"], how="left").merge(ibes, on=["ticker", "date"], how="left")


def aggregate_short_volume(short_volume: pd.DataFrame) -> pd.DataFrame:
    if short_volume.empty:
        return pd.DataFrame(columns=["ticker", "date", "regsho_short_share", "regsho_short_exempt_share"])
    frame = short_volume.copy()
    symbol_col = "symbol" if "symbol" in frame else "ticker"
    frame["ticker"] = frame[symbol_col].map(standardize_ticker)
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    short_cols = [col for col in frame.columns if col.startswith("short_") and not col.startswith("shortexempt_")]
    total_cols = [col for col in frame.columns if col.startswith("total_")]
    exempt_cols = [col for col in frame.columns if col.startswith("shortexempt_")]
    for col in short_cols + total_cols + exempt_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["short_volume"] = frame[short_cols].sum(axis=1, min_count=1)
    frame["total_volume"] = frame[total_cols].sum(axis=1, min_count=1)
    frame["short_exempt_volume"] = frame[exempt_cols].sum(axis=1, min_count=1) if exempt_cols else np.nan
    out = (
        frame.groupby(["ticker", "date"], as_index=False)
        .agg(
            short_volume=("short_volume", "sum"),
            total_volume=("total_volume", "sum"),
            short_exempt_volume=("short_exempt_volume", "sum"),
        )
    )
    out["regsho_short_share"] = out["short_volume"] / out["total_volume"].replace(0, np.nan)
    out["regsho_short_exempt_share"] = out["short_exempt_volume"] / out["total_volume"].replace(0, np.nan)
    return out[["ticker", "date", "regsho_short_share", "regsho_short_exempt_share"]]


def merge_short_volume_to_panel(panel: pd.DataFrame, short_monthly: pd.DataFrame, ticker_map: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp()
    out = out.drop(columns=["ticker"], errors="ignore")
    mapping = ticker_map.copy()
    mapping["date"] = pd.to_datetime(mapping["date"]).dt.to_period("M").dt.to_timestamp()
    mapping["ticker"] = mapping["ticker"].map(standardize_ticker)
    short = short_monthly.copy()
    short["date"] = pd.to_datetime(short["date"]).dt.to_period("M").dt.to_timestamp()
    return out.merge(mapping[["permno", "date", "ticker"]], on=["permno", "date"], how="left").merge(
        short, on=["ticker", "date"], how="left"
    )
