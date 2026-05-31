from __future__ import annotations

import numpy as np
import pandas as pd


def clean_option_quotes(options: pd.DataFrame) -> pd.DataFrame:
    frame = options.copy()
    for col in ["best_bid", "best_offer", "impl_volatility", "delta", "strike_price"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"])
    if "exdate" in frame:
        frame["exdate"] = pd.to_datetime(frame["exdate"])
        frame["dte"] = (frame["exdate"] - frame["date"]).dt.days
    if {"best_bid", "best_offer"}.issubset(frame.columns):
        frame = frame[(frame["best_bid"] > 0) & (frame["best_offer"] > frame["best_bid"])]
        frame["mid_quote"] = (frame["best_bid"] + frame["best_offer"]) / 2.0
        frame["quoted_spread_pct"] = (frame["best_offer"] - frame["best_bid"]) / frame["mid_quote"]
    if "impl_volatility" in frame:
        frame = frame[frame["impl_volatility"].between(0.01, 5.0)]
    if "dte" in frame:
        frame = frame[frame["dte"].between(10, 365)]
    return frame


def surface_features(options: pd.DataFrame) -> pd.DataFrame:
    if options.empty:
        return pd.DataFrame()
    frame = clean_option_quotes(options)
    if frame.empty:
        return pd.DataFrame()
    keys = [col for col in ["date", "secid"] if col in frame.columns]
    if len(keys) != 2:
        raise ValueError("Option smoke features require date and secid columns.")
    if "cp_flag" in frame:
        frame["cp_flag_norm"] = frame["cp_flag"].astype(str).str.upper().str[0]
    else:
        frame["cp_flag_norm"] = ""
    frame["abs_delta_gap_atm"] = np.nan
    if "delta" in frame:
        frame["abs_delta_gap_atm"] = (frame["delta"].abs() - 0.5).abs()

    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        group = group.sort_values("abs_delta_gap_atm")
        puts = group[group["cp_flag_norm"].eq("P")]
        calls = group[group["cp_flag_norm"].eq("C")]
        row = {
            "date": key[0],
            "secid": key[1],
            "n_contracts": int(len(group)),
            "mean_iv": float(group["impl_volatility"].mean()),
            "median_spread_pct": float(group.get("quoted_spread_pct", pd.Series(dtype=float)).median()),
            "atm_iv": float(group["impl_volatility"].iloc[0]),
            "put_iv_mean": float(puts["impl_volatility"].mean()) if not puts.empty else np.nan,
            "call_iv_mean": float(calls["impl_volatility"].mean()) if not calls.empty else np.nan,
        }
        row["put_call_iv_spread"] = row["put_iv_mean"] - row["call_iv_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_surface_grid(options: pd.DataFrame, maturities: list[int], deltas: list[float]) -> pd.DataFrame:
    """Nearest-neighbor smoke grid; full SVI fitting is added after schema smoke passes."""
    frame = clean_option_quotes(options)
    if frame.empty or "delta" not in frame or "dte" not in frame:
        return pd.DataFrame()
    rows = []
    for (date, secid), group in frame.groupby(["date", "secid"], dropna=False):
        for maturity in maturities:
            for delta in deltas:
                metric = (group["dte"] - maturity).abs() + (group["delta"].abs() - delta).abs() * 365
                idx = metric.idxmin()
                rows.append(
                    {
                        "date": date,
                        "secid": secid,
                        "target_dte": maturity,
                        "target_abs_delta": delta,
                        "impl_volatility": float(group.loc[idx, "impl_volatility"]),
                    }
                )
    return pd.DataFrame(rows)
