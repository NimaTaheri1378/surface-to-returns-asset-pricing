from __future__ import annotations

import numpy as np
import pandas as pd


def month_key(date: pd.Series) -> pd.Series:
    return pd.to_datetime(date).dt.to_period("M").dt.to_timestamp()


def prepare_ff_monthly(factors: pd.DataFrame, five_factors: pd.DataFrame | None = None) -> pd.DataFrame:
    ff = factors.copy()
    ff["date"] = pd.to_datetime(ff["date"])
    ff["month"] = month_key(ff["date"])
    for col in ["mktrf", "smb", "hml", "rf", "umd"]:
        if col in ff:
            ff[col] = pd.to_numeric(ff[col], errors="coerce")
    keep = ["month"] + [col for col in ["mktrf", "smb", "hml", "rf", "umd"] if col in ff]
    out = ff[keep].drop_duplicates("month", keep="last")
    if five_factors is not None and not five_factors.empty:
        five = five_factors.copy()
        five["date"] = pd.to_datetime(five["date"])
        five["month"] = month_key(five["date"])
        for col in ["rmw", "cma"]:
            if col in five:
                five[col] = pd.to_numeric(five[col], errors="coerce")
        five_keep = ["month"] + [col for col in ["rmw", "cma"] if col in five]
        out = out.merge(five[five_keep].drop_duplicates("month", keep="last"), on="month", how="left")
    return out


def prepare_frb_monthly(rates: pd.DataFrame) -> pd.DataFrame:
    frame = rates.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = month_key(frame["date"])
    for col in frame.columns:
        if col not in {"date", "month"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    keep = ["month"]
    for col in ["fedfunds", "aaa", "baa", "mswp1", "mswp10", "d_tcmnom_y20", "ltiit"]:
        if col in frame:
            keep.append(col)
    out = frame[keep].drop_duplicates("month", keep="last")
    if {"baa", "aaa"}.issubset(out.columns):
        out["baa_aaa_spread"] = out["baa"] - out["aaa"]
    if {"mswp10", "fedfunds"}.issubset(out.columns):
        out["term_spread_10y_ff"] = out["mswp10"] - out["fedfunds"]
    if {"mswp10", "mswp1"}.issubset(out.columns):
        out["term_spread_10y_1y"] = out["mswp10"] - out["mswp1"]
    return out


def prepare_cboe_monthly(cboe: pd.DataFrame) -> pd.DataFrame:
    frame = cboe.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = month_key(frame["date"])
    for col in ["vix", "vxo", "vxn", "vxd"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    agg = {}
    for col in ["vix", "vxo", "vxn", "vxd"]:
        if col in frame:
            agg[f"{col}_eom"] = (col, "last")
            agg[f"{col}_mean"] = (col, "mean")
            agg[f"{col}_realized_state_vol"] = (col, "std")
    out = frame.sort_values("date").groupby("month").agg(**agg).reset_index()
    if "vix_eom" in out:
        out["vix_log"] = np.log(out["vix_eom"].clip(lower=1e-6))
        out["vix_change"] = out["vix_eom"].diff()
    return out


def merge_state_controls(panel: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["month"] = month_key(out["date"])
    merged = out.merge(state, on="month", how="left")
    return merged.drop(columns=["month"])


def build_state_control_table(
    ff: pd.DataFrame,
    ff5: pd.DataFrame,
    frb: pd.DataFrame,
    cboe: pd.DataFrame,
) -> pd.DataFrame:
    state = prepare_ff_monthly(ff, ff5)
    state = state.merge(prepare_frb_monthly(frb), on="month", how="outer")
    state = state.merge(prepare_cboe_monthly(cboe), on="month", how="outer")
    return state.sort_values("month").reset_index(drop=True)
