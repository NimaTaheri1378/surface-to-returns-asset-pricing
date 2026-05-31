from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd


TICKER_RE = re.compile(r"[^A-Z0-9.]")


def standardize_ticker(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = TICKER_RE.sub("", str(value).upper().strip())
    return text or None


def first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    lower = {col.lower(): col for col in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def normalize_taq_quotes(quotes: pd.DataFrame, trade_date: pd.Timestamp | str | None = None) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame(columns=["ticker", "date", "midquote", "half_spread", "quoted_spread", "depth"])
    frame = quotes.copy()
    symbol_col = first_existing(frame.columns, ["sym_root", "symbol", "ticker", "sym", "root"])
    bid_col = first_existing(frame.columns, ["bid", "bid_price", "best_bid", "bbid"])
    ask_col = first_existing(frame.columns, ["ofr", "ask", "offer", "ask_price", "best_ask", "bask"])
    bid_size_col = first_existing(frame.columns, ["bidsiz", "bid_size", "bidsize", "bid_shares"])
    ask_size_col = first_existing(frame.columns, ["ofrsiz", "ask_size", "asksize", "ask_shares"])
    date_col = first_existing(frame.columns, ["date", "trade_date", "datetime"])
    missing = [name for name, col in {"symbol": symbol_col, "bid": bid_col, "ask": ask_col}.items() if col is None]
    if missing:
        raise KeyError(f"Missing TAQ quote columns: {missing}")
    out = pd.DataFrame(
        {
            "ticker": frame[symbol_col].map(standardize_ticker),
            "bid": pd.to_numeric(frame[bid_col], errors="coerce"),
            "ask": pd.to_numeric(frame[ask_col], errors="coerce"),
        }
    )
    if date_col:
        out["date"] = pd.to_datetime(frame[date_col]).dt.normalize()
    elif trade_date is not None:
        out["date"] = pd.to_datetime(trade_date).normalize()
    else:
        raise KeyError("TAQ quote data need a date column or explicit trade_date.")
    if bid_size_col and ask_size_col:
        out["depth"] = (
            pd.to_numeric(frame[bid_size_col], errors="coerce").clip(lower=0)
            + pd.to_numeric(frame[ask_size_col], errors="coerce").clip(lower=0)
        ) / 2.0
    else:
        out["depth"] = np.nan
    out = out.dropna(subset=["ticker", "bid", "ask"])
    out = out[(out["bid"] > 0) & (out["ask"] > out["bid"])]
    out["midquote"] = (out["bid"] + out["ask"]) / 2.0
    out["half_spread"] = (out["ask"] - out["bid"]) / (2.0 * out["midquote"])
    out["quoted_spread"] = (out["ask"] - out["bid"]) / out["midquote"]
    return out[["ticker", "date", "midquote", "half_spread", "quoted_spread", "depth"]]


def normalize_taq_trades(trades: pd.DataFrame, trade_date: pd.Timestamp | str | None = None) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["ticker", "date", "price", "size", "dollar_volume"])
    frame = trades.copy()
    symbol_col = first_existing(frame.columns, ["sym_root", "symbol", "ticker", "sym", "root"])
    price_col = first_existing(frame.columns, ["price", "tr_scond_price", "trade_price", "prc"])
    size_col = first_existing(frame.columns, ["size", "tr_siz", "trade_size", "volume", "shares"])
    date_col = first_existing(frame.columns, ["date", "trade_date", "datetime"])
    missing = [name for name, col in {"symbol": symbol_col, "price": price_col, "size": size_col}.items() if col is None]
    if missing:
        raise KeyError(f"Missing TAQ trade columns: {missing}")
    out = pd.DataFrame(
        {
            "ticker": frame[symbol_col].map(standardize_ticker),
            "price": pd.to_numeric(frame[price_col], errors="coerce"),
            "size": pd.to_numeric(frame[size_col], errors="coerce"),
        }
    )
    if date_col:
        out["date"] = pd.to_datetime(frame[date_col]).dt.normalize()
    elif trade_date is not None:
        out["date"] = pd.to_datetime(trade_date).normalize()
    else:
        raise KeyError("TAQ trade data need a date column or explicit trade_date.")
    out = out.dropna(subset=["ticker", "price", "size"])
    out = out[(out["price"] > 0) & (out["size"] > 0)]
    out["dollar_volume"] = out["price"] * out["size"]
    return out[["ticker", "date", "price", "size", "dollar_volume"]]


def aggregate_taq_daily_costs(
    quotes: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    trade_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    quote_clean = normalize_taq_quotes(quotes, trade_date=trade_date)
    quote_summary = (
        quote_clean.groupby(["ticker", "date"])
        .agg(
            taq_midquote=("midquote", "median"),
            taq_half_spread=("half_spread", "median"),
            taq_quoted_spread=("quoted_spread", "median"),
            taq_depth=("depth", "median"),
            taq_quote_obs=("half_spread", "size"),
        )
        .reset_index()
    )
    if trades is None or trades.empty:
        quote_summary["taq_dollar_volume"] = np.nan
        quote_summary["taq_share_volume"] = np.nan
        quote_summary["taq_trade_obs"] = 0
        quote_summary["taq_intraday_vol"] = np.nan
        quote_summary["taq_one_way_cost"] = quote_summary["taq_half_spread"]
        return quote_summary

    trade_clean = normalize_taq_trades(trades, trade_date=trade_date)
    trade_clean = trade_clean.sort_values(["ticker", "date"])
    trade_clean["trade_return"] = trade_clean.groupby(["ticker", "date"])["price"].pct_change()
    trade_summary = (
        trade_clean.groupby(["ticker", "date"])
        .agg(
            taq_dollar_volume=("dollar_volume", "sum"),
            taq_share_volume=("size", "sum"),
            taq_trade_obs=("price", "size"),
            taq_intraday_vol=("trade_return", "std"),
        )
        .reset_index()
    )
    out = quote_summary.merge(trade_summary, on=["ticker", "date"], how="left")
    out["taq_intraday_vol"] = out["taq_intraday_vol"].fillna(0.0)
    out["taq_one_way_cost"] = out["taq_half_spread"]
    return out


def merge_taq_costs_to_panel(panel: pd.DataFrame, cost_table: pd.DataFrame, identifier_map: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    costs = cost_table.copy()
    costs["date"] = pd.to_datetime(costs["date"])
    mapping = identifier_map.copy()
    mapping["date"] = pd.to_datetime(mapping["date"])
    mapping["ticker"] = mapping["ticker"].map(standardize_ticker)
    mapped = data.merge(mapping[["permno", "date", "ticker"]], on=["permno", "date"], how="left")
    return mapped.merge(costs, on=["ticker", "date"], how="left")


def calibrated_trade_cost(
    half_spread: pd.Series,
    intraday_vol: pd.Series,
    abs_trade_weight: pd.Series,
    dollar_volume: pd.Series,
    portfolio_value: float = 1.0,
    impact_eta: float = 0.10,
) -> pd.Series:
    adv = pd.to_numeric(dollar_volume, errors="coerce").replace(0, np.nan)
    trade_dollars = pd.to_numeric(abs_trade_weight, errors="coerce").clip(lower=0) * float(portfolio_value)
    participation = (trade_dollars / adv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    impact = impact_eta * pd.to_numeric(intraday_vol, errors="coerce").fillna(0.0) * participation.pow(0.5)
    return pd.to_numeric(half_spread, errors="coerce").fillna(0.0) + impact


def _liquidity_bin(frame: pd.DataFrame, bins: int) -> pd.Series:
    if "market_equity" in frame:
        value = pd.to_numeric(frame["market_equity"], errors="coerce")
    elif "log_market_equity" in frame:
        value = pd.to_numeric(frame["log_market_equity"], errors="coerce")
    else:
        return pd.Series(0, index=frame.index, dtype="int64")
    valid = value.notna()
    if valid.sum() < 2:
        return pd.Series(0, index=frame.index, dtype="int64")
    ranked = value[valid].rank(method="first")
    q = int(min(max(2, bins), valid.sum()))
    labels = pd.qcut(ranked, q=q, labels=False, duplicates="drop")
    out = pd.Series(-1, index=frame.index, dtype="int64")
    out.loc[valid] = labels.astype("int64")
    return out


def _group_fill(
    out: pd.DataFrame,
    value_col: str,
    observed: pd.Series,
    default_value: float,
) -> pd.Series:
    values = pd.to_numeric(out[value_col], errors="coerce") if value_col in out else pd.Series(np.nan, index=out.index)
    filled = values.copy()
    if observed.any():
        by_year_bin = out.loc[observed].groupby(["_taq_year", "_taq_liquidity_bin"])[value_col].median()
        by_bin = out.loc[observed].groupby("_taq_liquidity_bin")[value_col].median()
        by_year = out.loc[observed].groupby("_taq_year")[value_col].median()
        for idx in filled[filled.isna()].index:
            key = (out.at[idx, "_taq_year"], out.at[idx, "_taq_liquidity_bin"])
            value = by_year_bin.get(key, np.nan)
            if pd.isna(value):
                value = by_bin.get(out.at[idx, "_taq_liquidity_bin"], np.nan)
            if pd.isna(value):
                value = by_year.get(out.at[idx, "_taq_year"], np.nan)
            if pd.isna(value):
                value = values[observed].median()
            filled.at[idx] = value
    return filled.fillna(default_value)


def fill_taq_costs_with_calibration(
    panel: pd.DataFrame,
    bins: int = 10,
    default_half_spread_bps: float = 10.0,
) -> pd.DataFrame:
    """Fill sparse exact TAQ costs with liquidity/year calibrated values.

    Exact TAQ rows are retained. Missing rows are filled from same-year and same-liquidity-bin
    medians when possible, then broader medians, then conservative defaults. This keeps
    cost-aware portfolio tests from treating missing TAQ coverage as free trading.
    """
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in ["taq_half_spread", "taq_intraday_vol", "taq_dollar_volume"]:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    observed = out["taq_half_spread"].notna()
    out["_taq_year"] = out["date"].dt.year
    out["_taq_liquidity_bin"] = _liquidity_bin(out, bins=bins)
    default_half_spread = float(default_half_spread_bps) / 10000.0
    out["taq_half_spread"] = _group_fill(out, "taq_half_spread", observed, default_half_spread).clip(0.0, 0.10)

    if "idio_vol_252d" in out:
        default_daily_vol = pd.to_numeric(out["idio_vol_252d"], errors="coerce") / np.sqrt(252.0)
    elif "realized_vol_12m" in out:
        default_daily_vol = pd.to_numeric(out["realized_vol_12m"], errors="coerce") / np.sqrt(252.0)
    else:
        default_daily_vol = pd.Series(np.nan, index=out.index)
    default_intraday = float(out.loc[observed, "taq_intraday_vol"].median()) if observed.any() else np.nan
    if not np.isfinite(default_intraday):
        default_intraday = float(default_daily_vol.median()) if default_daily_vol.notna().any() else 0.015
    out["taq_intraday_vol"] = _group_fill(out, "taq_intraday_vol", observed, default_intraday)
    out["taq_intraday_vol"] = out["taq_intraday_vol"].where(out["taq_intraday_vol"].notna(), default_daily_vol).clip(0.0, 1.0)

    fallback_dollar_volume = pd.Series(np.nan, index=out.index)
    if "dollar_volume" in out:
        fallback_dollar_volume = pd.to_numeric(out["dollar_volume"], errors="coerce") / 21.0
    elif {"prc", "vol"}.issubset(out.columns):
        fallback_dollar_volume = pd.to_numeric(out["prc"], errors="coerce").abs() * pd.to_numeric(out["vol"], errors="coerce") * 100.0 / 21.0
    default_dollar_volume = float(out.loc[observed, "taq_dollar_volume"].median()) if observed.any() else np.nan
    if not np.isfinite(default_dollar_volume):
        default_dollar_volume = float(fallback_dollar_volume.median()) if fallback_dollar_volume.notna().any() else 1_000_000.0
    out["taq_dollar_volume"] = out["taq_dollar_volume"].where(out["taq_dollar_volume"].notna(), fallback_dollar_volume)
    out["taq_dollar_volume"] = _group_fill(out, "taq_dollar_volume", observed, default_dollar_volume).clip(1.0, None)

    out["taq_cost_source"] = np.where(observed, "exact_taq", "calibrated_taq")
    return out.drop(columns=["_taq_year", "_taq_liquidity_bin"])


def portfolio_taq_costs(
    trades: pd.DataFrame,
    cost_panel: pd.DataFrame,
    portfolio_value: float = 1.0,
    impact_eta: float = 0.10,
) -> pd.DataFrame:
    needed = {"date", "permno", "abs_trade_weight"}
    missing = needed.difference(trades.columns)
    if missing:
        raise KeyError(f"Missing trade columns for TAQ costs: {sorted(missing)}")
    data = trades.copy()
    data["date"] = pd.to_datetime(data["date"])
    costs = cost_panel.copy()
    costs["date"] = pd.to_datetime(costs["date"])
    merged = data.merge(costs, on=["date", "permno"], how="left")
    merged["taq_calibrated_one_way_cost"] = calibrated_trade_cost(
        merged.get("taq_half_spread", pd.Series(0.0, index=merged.index)),
        merged.get("taq_intraday_vol", pd.Series(0.0, index=merged.index)),
        merged["abs_trade_weight"],
        merged.get("taq_dollar_volume", pd.Series(math.nan, index=merged.index)),
        portfolio_value=portfolio_value,
        impact_eta=impact_eta,
    )
    merged["taq_trade_cost"] = merged["abs_trade_weight"] * merged["taq_calibrated_one_way_cost"]
    return (
        merged.groupby("date")
        .agg(
            taq_total_cost=("taq_trade_cost", "sum"),
            taq_cost_coverage=("taq_half_spread", lambda s: float(s.notna().mean())),
            taq_traded_names=("permno", "nunique"),
        )
        .reset_index()
    )
