from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denom


def normalize_return_units(values: pd.Series) -> pd.Series:
    """Convert percent-style return columns to decimal units when needed."""
    numeric = pd.to_numeric(values, errors="coerce")
    typical_abs = numeric.abs().dropna().median()
    if pd.notna(typical_abs) and typical_abs > 0.05:
        return numeric / 100.0
    return numeric


def _preferred_stock(frame: pd.DataFrame) -> pd.Series:
    for col in ["pstkrv", "pstkl", "pstk"]:
        if col in frame:
            value = pd.to_numeric(frame[col], errors="coerce")
            if value.notna().any():
                return value.fillna(0.0)
    return pd.Series(0.0, index=frame.index)


def compute_compustat_annual_characteristics(funda: pd.DataFrame) -> pd.DataFrame:
    frame = funda.copy()
    frame["datadate"] = pd.to_datetime(frame["datadate"])
    frame = frame.sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "datadate"], keep="last")
    numeric_cols = [
        "at",
        "ceq",
        "seq",
        "txditc",
        "sale",
        "revt",
        "cogs",
        "xsga",
        "xint",
        "ni",
        "ib",
        "oancf",
        "capx",
        "act",
        "che",
        "lct",
        "dlc",
        "dltt",
        "txp",
        "dp",
        "csho",
        "prcc_f",
    ]
    for col in numeric_cols:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = np.nan

    preferred = _preferred_stock(frame)
    deferred_taxes = frame["txditc"].fillna(0.0)
    seq_book = frame["seq"].where(frame["seq"].notna(), frame["ceq"])
    frame["book_equity"] = seq_book + deferred_taxes - preferred
    revenue = frame["revt"].where(frame["revt"].notna(), frame["sale"])
    frame["_revenue"] = revenue
    frame["gross_profitability"] = safe_divide(frame["_revenue"] - frame["cogs"], frame["at"])
    frame["operating_profitability"] = safe_divide(
        frame["_revenue"] - frame["cogs"] - frame["xsga"].fillna(0.0) - frame["xint"].fillna(0.0),
        frame["book_equity"],
    )
    frame["roe"] = safe_divide(frame["ni"], frame["book_equity"])
    frame["leverage"] = safe_divide(frame["dltt"].fillna(0.0) + frame["dlc"].fillna(0.0), frame["at"])
    frame["capex_at"] = safe_divide(frame["capx"], frame["at"])
    working_capital_change = (
        (frame["act"] - frame["che"]) - (frame["lct"] - frame["dlc"].fillna(0.0) - frame["txp"].fillna(0.0))
    )
    balance_sheet_accruals = working_capital_change - frame["dp"].fillna(0.0)
    frame["accruals_at"] = safe_divide(balance_sheet_accruals, frame["at"])
    if frame["oancf"].notna().any():
        frame["cashflow_accruals_at"] = safe_divide(frame["ib"] - frame["oancf"], frame["at"])
    else:
        frame["cashflow_accruals_at"] = np.nan
    frame["lag_at"] = frame.groupby("gvkey")["at"].shift(1)
    frame["lag_revenue"] = frame.groupby("gvkey")["_revenue"].shift(1)
    frame["lag_debt"] = frame.groupby("gvkey")["dltt"].shift(1).fillna(0.0) + frame.groupby("gvkey")["dlc"].shift(1).fillna(0.0)
    frame["lag_csho"] = frame.groupby("gvkey")["csho"].shift(1)
    frame["investment"] = safe_divide(frame["at"], frame["lag_at"]) - 1.0
    frame["firm_size_comp"] = np.log(frame["at"].clip(lower=1))
    frame["comp_market_equity"] = frame["csho"] * frame["prcc_f"]
    frame["book_to_market_comp"] = safe_divide(frame["book_equity"], frame["comp_market_equity"])
    debt = frame["dltt"].fillna(0.0) + frame["dlc"].fillna(0.0)
    frame["asset_growth"] = frame["investment"]
    frame["sales_growth"] = safe_divide(frame["_revenue"], frame["lag_revenue"]) - 1.0
    frame["cash_at"] = safe_divide(frame["che"], frame["at"])
    frame["cashflow_at"] = safe_divide(frame["oancf"], frame["at"])
    frame["net_working_capital_at"] = safe_divide(working_capital_change, frame["at"])
    frame["profit_margin"] = safe_divide(frame["ni"], frame["_revenue"])
    frame["sales_at"] = safe_divide(frame["_revenue"], frame["at"])
    frame["debt_growth"] = safe_divide(debt, frame["lag_debt"]) - 1.0
    frame["equity_issuance"] = safe_divide(frame["csho"], frame["lag_csho"]) - 1.0
    frame["earnings_to_price"] = safe_divide(frame["ni"], frame["comp_market_equity"])
    frame["available_date"] = frame["datadate"] + pd.DateOffset(months=6)
    keep = [
        "gvkey",
        "datadate",
        "fyear",
        "available_date",
        "book_equity",
        "gross_profitability",
        "operating_profitability",
        "roe",
        "leverage",
        "capex_at",
        "accruals_at",
        "cashflow_accruals_at",
        "investment",
        "asset_growth",
        "sales_growth",
        "cash_at",
        "cashflow_at",
        "net_working_capital_at",
        "profit_margin",
        "sales_at",
        "debt_growth",
        "equity_issuance",
        "earnings_to_price",
        "firm_size_comp",
        "book_to_market_comp",
    ]
    return frame[[col for col in keep if col in frame]]


def compute_crsp_monthly_characteristics(monthly: pd.DataFrame) -> pd.DataFrame:
    frame = monthly.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for col in ["ret", "prc", "shrout", "vol"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = np.nan
    frame = frame.sort_values(["permno", "date"]).drop_duplicates(["permno", "date"], keep="last")
    frame["market_equity"] = frame["prc"].abs() * frame["shrout"] * 1000.0
    frame["log_market_equity"] = np.log(frame["market_equity"].clip(lower=1.0))
    frame["turnover"] = safe_divide(frame["vol"], frame["shrout"])
    frame["dollar_volume"] = frame["prc"].abs() * frame["vol"] * 100.0
    frame["log_dollar_volume"] = np.log(frame["dollar_volume"].clip(lower=1.0))
    frame["ret_1_0"] = frame.groupby("permno")["ret"].shift(0)
    frame["short_reversal"] = frame["ret_1_0"]
    lagged = frame.groupby("permno")["ret"].shift(2)
    frame["momentum_12_2"] = (
        (1.0 + lagged)
        .groupby(frame["permno"])
        .rolling(11, min_periods=8)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
        - 1.0
    )
    lagged_6_1 = frame.groupby("permno")["ret"].shift(1)
    frame["momentum_6_1"] = (
        (1.0 + lagged_6_1)
        .groupby(frame["permno"])
        .rolling(6, min_periods=4)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
        - 1.0
    )
    lagged_36_13 = frame.groupby("permno")["ret"].shift(13)
    frame["momentum_36_13"] = (
        (1.0 + lagged_36_13)
        .groupby(frame["permno"])
        .rolling(24, min_periods=18)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
        - 1.0
    )
    frame["realized_vol_12m"] = (
        frame.groupby("permno")["ret"]
        .rolling(12, min_periods=8)
        .std()
        .reset_index(level=0, drop=True)
        * np.sqrt(12.0)
    )
    amihud = safe_divide(frame["ret"].abs(), frame["dollar_volume"])
    frame["amihud_illiq_12m"] = (
        amihud.groupby(frame["permno"]).rolling(12, min_periods=8).mean().reset_index(level=0, drop=True)
    )
    frame["max_ret_12m"] = (
        frame.groupby("permno")["ret"].rolling(12, min_periods=8).max().reset_index(level=0, drop=True)
    )
    frame["ret_skew_12m"] = (
        frame.groupby("permno")["ret"].rolling(12, min_periods=8).skew().reset_index(level=0, drop=True)
    )
    frame["zero_volume_12m"] = (
        frame["vol"].le(0)
        .astype(float)
        .groupby(frame["permno"])
        .rolling(12, min_periods=8)
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame["price"] = frame["prc"].abs()
    keep = [
        "permno",
        "date",
        "market_equity",
        "log_market_equity",
        "turnover",
        "dollar_volume",
        "log_dollar_volume",
        "short_reversal",
        "momentum_12_2",
        "momentum_6_1",
        "momentum_36_13",
        "realized_vol_12m",
        "amihud_illiq_12m",
        "max_ret_12m",
        "ret_skew_12m",
        "zero_volume_12m",
        "price",
    ]
    return frame[keep]


def merge_compustat_characteristics(panel: pd.DataFrame, comp_chars: pd.DataFrame, ccm: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    links = ccm.copy()
    links["linkdt"] = pd.to_datetime(links["linkdt"], errors="coerce")
    links["linkenddt"] = pd.to_datetime(links["linkenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    if "lpermno" in links and "permno" not in links:
        links = links.rename(columns={"lpermno": "permno"})
    links = links[links["permno"].notna()].copy()
    links["permno"] = links["permno"].astype(int)
    link_cols = [col for col in ["gvkey", "permno", "linkdt", "linkenddt", "linktype", "linkprim"] if col in links]
    out = out.merge(links[link_cols], on="permno", how="left")
    out = out[(out["linkdt"].isna()) | ((out["linkdt"] <= out["date"]) & (out["date"] <= out["linkenddt"]))]

    comp = comp_chars.copy()
    comp["available_date"] = pd.to_datetime(comp["available_date"])
    comp_value_cols = [col for col in comp.columns if col != "gvkey"]
    merged = out.merge(comp, on="gvkey", how="left")
    merged["_available_ok"] = merged["available_date"].isna() | (merged["available_date"] <= merged["date"])
    sort_cols = ["date", "permno", "available_date", "datadate"]
    merged["_available_rank"] = merged["_available_ok"].astype(int)
    merged = merged.sort_values(["date", "permno", "_available_rank", "available_date", "datadate"])
    merged = merged.drop_duplicates(["date", "permno", "secid"], keep="last")
    unavailable = ~merged["_available_ok"].fillna(False)
    for col in comp_value_cols:
        if col in merged:
            merged.loc[unavailable, col] = np.nan
    merged = merged.drop(columns=["_available_ok", "_available_rank"])
    return merged.drop(columns=[col for col in ["linkdt", "linkenddt"] if col in merged])


def compute_daily_risk_characteristics(daily: pd.DataFrame, factors: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ret"] = normalize_return_units(frame["ret"])
    if factors is not None and not factors.empty:
        ff = factors.copy()
        ff["date"] = pd.to_datetime(ff["date"])
        frame = frame.merge(ff, on="date", how="left")
        market_col = "mktrf" if "mktrf" in frame else "vwretd" if "vwretd" in frame else None
    else:
        market_col = "vwretd" if "vwretd" in frame else None
    if market_col is None:
        frame["beta_252d"] = np.nan
        frame["idio_vol_252d"] = np.nan
    else:
        frame[market_col] = normalize_return_units(frame[market_col])
        rf = normalize_return_units(frame["rf"]) if "rf" in frame else 0.0
        frame["excess_ret"] = frame["ret"] - rf
        rows = []
        for permno, group in frame.sort_values(["permno", "date"]).groupby("permno"):
            group = group.dropna(subset=["excess_ret", market_col]).copy()
            if len(group) < 60:
                continue
            y = group["excess_ret"]
            x = group[market_col]
            cov = y.rolling(252, min_periods=60).cov(x)
            var = group[market_col].rolling(252, min_periods=60).var()
            beta = cov / var.replace(0, np.nan)
            alpha = y.rolling(252, min_periods=60).mean() - beta * x.rolling(252, min_periods=60).mean()
            resid = y - alpha - beta * x
            idio = resid.rolling(252, min_periods=60).std() * np.sqrt(252.0)
            total_vol = y.rolling(252, min_periods=60).std() * np.sqrt(252.0)
            corr = y.rolling(252, min_periods=60).corr(x)
            obs = y.rolling(252, min_periods=60).count()
            rows.append(
                pd.DataFrame(
                    {
                        "permno": permno,
                        "date": group["date"],
                        "beta_252d": beta,
                        "idio_vol_252d": idio,
                        "total_vol_252d": total_vol,
                        "mkt_corr_252d": corr,
                        "daily_obs_252d": obs,
                    }
                )
            )
        columns = [
            "permno",
            "date",
            "beta_252d",
            "idio_vol_252d",
            "total_vol_252d",
            "mkt_corr_252d",
            "daily_obs_252d",
        ]
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    return frame[
        [
            "permno",
            "date",
            "beta_252d",
            "idio_vol_252d",
        ]
    ]


def collapse_daily_risk_to_panel_months(daily_risk: pd.DataFrame, panel_keys: pd.DataFrame) -> pd.DataFrame:
    """Map daily rolling risk estimates to the CRSP monthly dates used by the panel."""
    if daily_risk.empty or panel_keys.empty:
        cols = ["permno", "date", "daily_risk_date"] + [
            col for col in daily_risk.columns if col not in {"permno", "date"}
        ]
        return pd.DataFrame(columns=cols)
    risk = daily_risk.copy()
    risk["date"] = pd.to_datetime(risk["date"])
    risk["month"] = risk["date"].dt.to_period("M")
    risk = risk.sort_values(["permno", "month", "date"]).drop_duplicates(["permno", "month"], keep="last")
    risk = risk.rename(columns={"date": "daily_risk_date"})
    keys = panel_keys[["permno", "date"]].drop_duplicates().copy()
    keys["date"] = pd.to_datetime(keys["date"])
    keys["month"] = keys["date"].dt.to_period("M")
    out = keys.merge(risk, on=["permno", "month"], how="left").drop(columns=["month"])
    ordered = ["permno", "date", "daily_risk_date"] + [
        col for col in out.columns if col not in {"permno", "date", "daily_risk_date"}
    ]
    return out[ordered]
