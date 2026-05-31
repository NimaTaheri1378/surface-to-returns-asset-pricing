from __future__ import annotations

import numpy as np
import pandas as pd


def one_way_transaction_cost(
    half_spread: pd.Series,
    volatility: pd.Series,
    trade_fraction_adv: pd.Series,
    eta: float = 0.10,
) -> pd.Series:
    return half_spread + eta * volatility * trade_fraction_adv.clip(lower=0).pow(0.5)


def portfolio_turnover(weights: pd.DataFrame, date_col: str = "date", asset_col: str = "permno") -> pd.Series:
    pivot = weights.pivot(index=date_col, columns=asset_col, values="weight").fillna(0.0).sort_index()
    turnover = pivot.diff().abs().sum(axis=1, min_count=1)
    if not turnover.empty:
        turnover.iloc[0] = pivot.iloc[0].abs().sum()
    return turnover


def long_short_decile_weights(
    frame: pd.DataFrame,
    score_col: str = "pred",
    date_col: str = "date",
    asset_col: str = "permno",
    quantiles: int = 10,
    min_assets: int = 20,
) -> pd.DataFrame:
    """Build dollar-neutral weights with 50% gross exposure on each side."""
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    required = {date_col, asset_col, score_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns for decile weights: {sorted(missing)}")
    rows = []
    clean = frame.dropna(subset=[date_col, asset_col, score_col]).copy()
    clean[date_col] = pd.to_datetime(clean[date_col])
    for date, group in clean.groupby(date_col, sort=True):
        if len(group) < min_assets:
            continue
        ranks = group[score_col].rank(method="first", pct=True)
        long_mask = ranks > 1.0 - (1.0 / quantiles)
        short_mask = ranks <= 1.0 / quantiles
        n_long = int(long_mask.sum())
        n_short = int(short_mask.sum())
        if n_long == 0 or n_short == 0:
            continue
        selected = group.loc[long_mask | short_mask, [date_col, asset_col, score_col]].copy()
        selected["rank_pct"] = ranks.loc[selected.index].to_numpy()
        selected["side"] = np.where(selected["rank_pct"] > 1.0 - (1.0 / quantiles), "long", "short")
        selected["weight"] = np.where(selected["side"].eq("long"), 0.5 / n_long, -0.5 / n_short)
        selected[date_col] = date
        rows.append(selected)
    if not rows:
        return pd.DataFrame(columns=[date_col, asset_col, score_col, "rank_pct", "side", "weight"])
    return pd.concat(rows, ignore_index=True)


def weight_trades(weights: pd.DataFrame, date_col: str = "date", asset_col: str = "permno") -> pd.DataFrame:
    pivot = weights.pivot(index=date_col, columns=asset_col, values="weight").fillna(0.0).sort_index()
    trades = pivot.diff()
    if not trades.empty:
        trades.iloc[0] = pivot.iloc[0]
    long = (
        trades.abs()
        .stack()
        .rename("abs_trade_weight")
        .reset_index()
        .rename(columns={date_col: "date", asset_col: "permno"})
    )
    return long[long["abs_trade_weight"].gt(0)].reset_index(drop=True)


def long_short_portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    return_col: str = "next_ret",
    half_spread_col: str | None = None,
    one_way_cost_bps: float = 10.0,
    date_col: str = "date",
    asset_col: str = "permno",
) -> pd.DataFrame:
    """Compute monthly gross and net returns for the decile portfolio."""
    if weights.empty:
        return pd.DataFrame()
    needed = {date_col, asset_col, "weight"}
    missing = needed.difference(weights.columns)
    if missing:
        raise KeyError(f"Missing weight columns: {sorted(missing)}")
    ret_needed = {date_col, asset_col, return_col}
    ret_missing = ret_needed.difference(returns.columns)
    if ret_missing:
        raise KeyError(f"Missing return columns: {sorted(ret_missing)}")

    w = weights.copy()
    r = returns.copy()
    w[date_col] = pd.to_datetime(w[date_col])
    r[date_col] = pd.to_datetime(r[date_col])
    merged = w.merge(r, on=[date_col, asset_col], how="left")
    merged[return_col] = pd.to_numeric(merged[return_col], errors="coerce")
    merged["weighted_return"] = merged["weight"] * merged[return_col]
    monthly = (
        merged.groupby(date_col)
        .agg(
            gross_return=("weighted_return", "sum"),
            n_positions=(asset_col, "nunique"),
            n_long=("side", lambda s: int((s == "long").sum()) if s.notna().any() else 0),
            n_short=("side", lambda s: int((s == "short").sum()) if s.notna().any() else 0),
            return_coverage=(return_col, lambda s: float(s.notna().mean())),
        )
        .reset_index()
        .rename(columns={date_col: "date"})
    )

    turnover = portfolio_turnover(w, date_col=date_col, asset_col=asset_col)
    turnover.name = "turnover"
    monthly = monthly.merge(turnover.reset_index().rename(columns={date_col: "date"}), on="date", how="left")
    fixed_rate = one_way_cost_bps / 10000.0
    monthly["fixed_cost"] = monthly["turnover"].fillna(0.0) * fixed_rate

    monthly["spread_cost"] = 0.0
    if half_spread_col and half_spread_col in r.columns:
        trades = weight_trades(w, date_col=date_col, asset_col=asset_col)
        spreads = r[[date_col, asset_col, half_spread_col]].copy()
        spreads[half_spread_col] = pd.to_numeric(spreads[half_spread_col], errors="coerce").clip(lower=0)
        spread_trades = trades.merge(
            spreads.rename(columns={date_col: "date", asset_col: "permno"}),
            on=["date", "permno"],
            how="left",
        )
        spread_trades["spread_cost"] = (
            spread_trades["abs_trade_weight"] * spread_trades[half_spread_col].fillna(0.0)
        )
        spread_cost = spread_trades.groupby("date")["spread_cost"].sum().reset_index()
        monthly = monthly.drop(columns=["spread_cost"]).merge(spread_cost, on="date", how="left")
        monthly["spread_cost"] = monthly["spread_cost"].fillna(0.0)

    monthly["total_cost"] = monthly["fixed_cost"] + monthly["spread_cost"]
    monthly["net_return"] = monthly["gross_return"] - monthly["total_cost"]
    return monthly.sort_values("date").reset_index(drop=True)


def max_drawdown(returns: pd.Series) -> float:
    series = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    wealth = (1.0 + series).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min()) if len(drawdown) else float("nan")


def performance_summary(monthly: pd.DataFrame, return_col: str) -> dict[str, float | int | None]:
    if monthly.empty or return_col not in monthly:
        return {
            "months": 0,
            "mean_monthly_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "annualized_sharpe": None,
            "t_stat_naive": None,
            "hit_rate": None,
            "cumulative_return": None,
            "max_drawdown": None,
        }
    ret = pd.to_numeric(monthly[return_col], errors="coerce").dropna()
    if ret.empty:
        return {"months": 0}
    mean = float(ret.mean())
    std = float(ret.std(ddof=1)) if len(ret) > 1 else float("nan")
    vol = std * np.sqrt(12.0) if np.isfinite(std) else None
    sharpe = mean / std * np.sqrt(12.0) if std and np.isfinite(std) and std > 0 else None
    t_stat = mean / (std / np.sqrt(len(ret))) if std and np.isfinite(std) and std > 0 else None
    return {
        "months": int(len(ret)),
        "mean_monthly_return": mean,
        "annualized_return": float((1.0 + mean) ** 12 - 1.0),
        "annualized_volatility": float(vol) if vol is not None else None,
        "annualized_sharpe": float(sharpe) if sharpe is not None else None,
        "t_stat_naive": float(t_stat) if t_stat is not None else None,
        "hit_rate": float((ret > 0).mean()),
        "cumulative_return": float((1.0 + ret).prod() - 1.0),
        "max_drawdown": max_drawdown(ret),
    }
