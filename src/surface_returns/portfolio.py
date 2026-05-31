from __future__ import annotations

import numpy as np
import pandas as pd

from surface_returns.backtest import portfolio_turnover
from surface_returns.inference import to_month_start


def sic_to_ff12(sic: float | int | str | None) -> str:
    try:
        code = int(float(sic))
    except Exception:
        return "Other"
    if 100 <= code <= 999 or 2000 <= code <= 2399 or 2700 <= code <= 2749 or 2770 <= code <= 2799 or 3100 <= code <= 3199 or 3940 <= code <= 3989:
        return "Consumer"
    if 2500 <= code <= 2519 or 2590 <= code <= 2599 or 3630 <= code <= 3659 or 3710 <= code <= 3711 or 3714 <= code <= 3714 or 3716 <= code <= 3716 or 3750 <= code <= 3751 or 3792 <= code <= 3792 or 3900 <= code <= 3939 or 3990 <= code <= 3999:
        return "Manufacturing"
    if 2520 <= code <= 2589 or 2600 <= code <= 2699 or 2750 <= code <= 2769 or 3000 <= code <= 3099 or 3200 <= code <= 3569 or 3580 <= code <= 3629 or 3700 <= code <= 3709 or 3712 <= code <= 3713 or 3715 <= code <= 3715 or 3717 <= code <= 3749 or 3752 <= code <= 3791 or 3793 <= code <= 3799 or 3830 <= code <= 3839 or 3860 <= code <= 3899:
        return "Durables"
    if 1200 <= code <= 1399 or 2900 <= code <= 2999:
        return "Energy"
    if 2800 <= code <= 2829 or 2840 <= code <= 2899:
        return "Chemicals"
    if 3570 <= code <= 3579 or 3660 <= code <= 3692 or 3694 <= code <= 3699 or 3810 <= code <= 3829 or 7370 <= code <= 7379:
        return "BusinessEq"
    if 4800 <= code <= 4899:
        return "Telecom"
    if 4900 <= code <= 4949:
        return "Utilities"
    if 5000 <= code <= 5999 or 7200 <= code <= 7299 or 7600 <= code <= 7699:
        return "Shops"
    if 2830 <= code <= 2839 or 3693 <= code <= 3693 or 3840 <= code <= 3859 or 8000 <= code <= 8099:
        return "Health"
    if 6000 <= code <= 6999:
        return "Finance"
    return "Other"


def rolling_monthly_beta(frame: pd.DataFrame, return_col: str = "ret", market_col: str = "mktrf", window: int = 60) -> pd.DataFrame:
    data = frame[["permno", "date", return_col, market_col]].copy()
    data["date"] = to_month_start(data["date"])
    data[return_col] = pd.to_numeric(data[return_col], errors="coerce")
    data[market_col] = pd.to_numeric(data[market_col], errors="coerce")
    if data[market_col].abs().median(skipna=True) > 0.5:
        data[market_col] = data[market_col] / 100.0
    rows = []
    for permno, group in data.sort_values(["permno", "date"]).groupby("permno"):
        group = group.dropna(subset=[return_col, market_col])
        if len(group) < 24:
            continue
        cov = group[return_col].rolling(window, min_periods=24).cov(group[market_col])
        var = group[market_col].rolling(window, min_periods=24).var()
        beta = cov / var.replace(0, np.nan)
        rows.append(pd.DataFrame({"permno": permno, "date": group["date"], "beta_60m": beta}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["permno", "date", "beta_60m"])


def buffered_memberships(
    frame: pd.DataFrame,
    score_col: str = "pred",
    entry_pct: float = 0.10,
    exit_pct: float = 0.20,
    min_assets: int = 50,
) -> pd.DataFrame:
    rows = []
    prev_long: set[int] = set()
    prev_short: set[int] = set()
    data = frame.copy()
    data["date"] = to_month_start(data["date"])
    for date, group in data.dropna(subset=[score_col, "permno"]).groupby("date", sort=True):
        if len(group) < min_assets:
            continue
        ranks = group[score_col].rank(method="first", pct=True)
        permnos = group["permno"].astype(int)
        entry_long = set(permnos[ranks >= 1.0 - entry_pct])
        keep_long = set(permnos[(ranks >= 1.0 - exit_pct) & permnos.isin(prev_long)])
        entry_short = set(permnos[ranks <= entry_pct])
        keep_short = set(permnos[(ranks <= exit_pct) & permnos.isin(prev_short)])
        long_set = entry_long | keep_long
        short_set = entry_short | keep_short
        selected = group[permnos.isin(long_set | short_set)].copy()
        selected["side"] = np.where(selected["permno"].astype(int).isin(long_set), "long", "short")
        selected["rank_pct"] = ranks.loc[selected.index].to_numpy()
        rows.append(selected)
        prev_long = long_set
        prev_short = short_set
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def sector_balanced_weights(members: pd.DataFrame, sector_col: str = "sector") -> pd.DataFrame:
    if members.empty:
        return members.assign(weight=pd.Series(dtype=float))
    data = members.copy()
    data[sector_col] = data.get(sector_col, "Other")
    rows = []
    for (date, side), group in data.groupby(["date", "side"], sort=True):
        sectors = sorted(group[sector_col].fillna("Other").unique())
        side_gross = 0.5
        sign = 1.0 if side == "long" else -1.0
        for sector in sectors:
            subgroup = group[group[sector_col].fillna("Other").eq(sector)].copy()
            subgroup["weight"] = sign * side_gross / len(sectors) / len(subgroup)
            rows.append(subgroup)
    return pd.concat(rows, ignore_index=True) if rows else data.assign(weight=0.0)


def portfolio_returns_with_hedge(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    one_way_cost_bps: float = 10.0,
    beta_col: str = "beta_60m",
) -> pd.DataFrame:
    w = weights.copy()
    r = returns.copy()
    f = factors.copy()
    w["date"] = to_month_start(w["date"])
    r["date"] = to_month_start(r["date"])
    f["date"] = to_month_start(f["date"])
    merged = w.merge(r, on=["date", "permno"], how="left", suffixes=("", "_ret"))
    merged["weighted_return"] = merged["weight"] * pd.to_numeric(merged["next_ret"], errors="coerce")
    monthly = (
        merged.groupby("date")
        .agg(
            gross_return=("weighted_return", "sum"),
            n_positions=("permno", "nunique"),
            n_long=("side", lambda x: int((x == "long").sum())),
            n_short=("side", lambda x: int((x == "short").sum())),
            beta_exposure=(beta_col, lambda x: np.nan),
        )
        .reset_index()
    )
    if beta_col in merged:
        merged["_beta_contrib"] = merged["weight"] * pd.to_numeric(merged[beta_col], errors="coerce")
        beta = merged.groupby("date")["_beta_contrib"].sum().rename("beta_exposure").reset_index()
        monthly = monthly.drop(columns=["beta_exposure"]).merge(beta, on="date", how="left")
    else:
        monthly["beta_exposure"] = np.nan
    turnover = portfolio_turnover(w)
    monthly = monthly.merge(turnover.rename("turnover").reset_index(), on="date", how="left")
    monthly["fixed_cost"] = monthly["turnover"].fillna(0.0) * one_way_cost_bps / 10000.0
    f = f[["date", "mktrf"]].copy()
    f["mktrf"] = pd.to_numeric(f["mktrf"], errors="coerce")
    if f["mktrf"].abs().median(skipna=True) > 0.5:
        f["mktrf"] = f["mktrf"] / 100.0
    monthly = monthly.merge(f, on="date", how="left")
    monthly["market_hedge_weight"] = -monthly["beta_exposure"].fillna(0.0)
    monthly["market_hedge_return"] = monthly["market_hedge_weight"] * monthly["mktrf"].fillna(0.0)
    monthly["beta_neutral_gross_return"] = monthly["gross_return"] + monthly["market_hedge_return"]
    monthly["net_return"] = monthly["beta_neutral_gross_return"] - monthly["fixed_cost"]
    return monthly.sort_values("date").reset_index(drop=True)
