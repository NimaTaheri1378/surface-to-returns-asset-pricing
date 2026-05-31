from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_FEATURE_CANDIDATES = [
    "mean_iv",
    "atm_iv",
    "put_call_iv_spread",
    "median_spread_pct",
    "n_contracts",
    "market_equity",
    "log_market_equity",
    "turnover",
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
    "book_to_market_comp",
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
    "beta_252d",
    "idio_vol_252d",
    "total_vol_252d",
    "mkt_corr_252d",
    "vix_eom",
    "vix_mean",
    "vix_realized_state_vol",
    "vix_change",
    "vxo_eom",
    "vxn_eom",
    "vxd_eom",
    "baa_aaa_spread",
    "term_spread_10y_ff",
    "term_spread_10y_1y",
    "mktrf",
    "smb",
    "hml",
    "rmw",
    "cma",
    "umd",
    "taq_half_spread",
    "taq_quoted_spread",
    "taq_intraday_vol",
    "taq_dollar_volume",
    "taq_one_way_cost",
    "ibes_analyst_coverage",
    "ibes_forecast_dispersion",
    "ibes_revision_breadth",
    "ibes_mean_estimate",
    "regsho_short_share",
    "regsho_short_exempt_share",
    "fred_unrate",
    "fred_cpi_yoy",
    "fred_indpro_yoy",
    "fred_dgs10",
    "fred_dgs2",
    "fred_10y2y_spread",
    "fred_t10y3m",
    "fred_nfci",
    "bls_unrate",
    "bls_avg_hourly_earnings",
    "bls_cpi_u_yoy",
    "bea_gdp_yoy",
    "eia_wti_spot_return",
]

SURFACE_FEATURE_NAMES = {
    "mean_iv",
    "atm_iv",
    "put_call_iv_spread",
    "median_spread_pct",
    "n_contracts",
}

STATE_FEATURE_NAMES = {
    "vix_eom",
    "vix_mean",
    "vix_realized_state_vol",
    "vix_change",
    "vxo_eom",
    "vxn_eom",
    "vxd_eom",
    "baa_aaa_spread",
    "term_spread_10y_ff",
    "term_spread_10y_1y",
    "mktrf",
    "smb",
    "hml",
    "rmw",
    "cma",
    "umd",
}

STATE_FEATURE_PREFIXES = ("fred_", "bls_", "bea_", "eia_")


@dataclass(frozen=True)
class ConditionalSDFSplit:
    fold: int
    train_dates: list[pd.Timestamp]
    test_dates: list[pd.Timestamp]


@dataclass(frozen=True)
class ConditionalSDFFeatureGroups:
    surface: list[str]
    tabular: list[str]
    state: list[str]

    @property
    def ordered(self) -> list[str]:
        return [*self.surface, *self.tabular, *self.state]


def classify_conditional_sdf_features(feature_cols: list[str]) -> ConditionalSDFFeatureGroups:
    surface: list[str] = []
    tabular: list[str] = []
    state: list[str] = []
    for col in dict.fromkeys(feature_cols):
        if col in SURFACE_FEATURE_NAMES or col.startswith("surface_ae_") or col.startswith(("svi_", "ssvi_")):
            surface.append(col)
        elif col in STATE_FEATURE_NAMES or col.startswith(STATE_FEATURE_PREFIXES):
            state.append(col)
        else:
            tabular.append(col)
    return ConditionalSDFFeatureGroups(surface=surface, tabular=tabular, state=state)


def select_conditional_sdf_features(frame: pd.DataFrame, min_nonmissing: float = 0.65) -> list[str]:
    candidates = list(DEFAULT_FEATURE_CANDIDATES)
    candidates.extend(sorted(col for col in frame.columns if col.startswith("surface_ae_")))
    selected = []
    for col in dict.fromkeys(candidates):
        if col not in frame:
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().mean() >= min_nonmissing and numeric.nunique(dropna=True) > 2:
            selected.append(col)
    return selected


def prepare_conditional_sdf_frame(
    frame: pd.DataFrame,
    feature_cols: list[str],
    return_col: str = "next_ret",
    min_assets_per_month: int = 50,
    state_cols: list[str] | None = None,
    winsor_limits: tuple[float, float] = (0.005, 0.995),
) -> pd.DataFrame:
    required = {"date", "permno", return_col}.union(feature_cols)
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing conditional SDF columns: {sorted(missing)}")
    data = frame[["date", "permno", return_col, *feature_cols]].copy()
    data["date"] = pd.to_datetime(data["date"]).dt.to_period("M").dt.to_timestamp()
    data["permno"] = pd.to_numeric(data["permno"], errors="coerce")
    data[return_col] = pd.to_numeric(data[return_col], errors="coerce")
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "permno", return_col])
    state_set = set(state_cols or [])
    for col in feature_cols:
        global_median = data[col].median(skipna=True)
        if not np.isfinite(global_median):
            global_median = 0.0

        def _fill_month(s: pd.Series) -> pd.Series:
            month_median = s.median(skipna=True) if s.notna().any() else global_median
            return s.fillna(month_median)

        data[col] = data.groupby("date")[col].transform(_fill_month)
        data[col] = data[col].fillna(global_median)
    for col in feature_cols:
        if col in state_set:
            continue

        def _scale(s: pd.Series) -> pd.Series:
            if winsor_limits:
                low, high = winsor_limits
                if 0.0 <= low < high <= 1.0 and s.notna().sum() > 10:
                    lo = s.quantile(low)
                    hi = s.quantile(high)
                    s = s.clip(lower=lo, upper=hi)
            std = s.std(ddof=0)
            if not np.isfinite(std) or std == 0:
                return pd.Series(0.0, index=s.index)
            return (s - s.mean()) / std

        data[col] = data.groupby("date")[col].transform(_scale)
    counts = data.groupby("date")["permno"].transform("nunique")
    data = data[counts >= min_assets_per_month].copy()
    return data.sort_values(["date", "permno"]).reset_index(drop=True)


def make_walk_forward_sdf_splits(
    dates: list[pd.Timestamp],
    min_train_months: int,
    test_months: int,
) -> list[ConditionalSDFSplit]:
    clean_dates = sorted(pd.Timestamp(item) for item in dates)
    splits: list[ConditionalSDFSplit] = []
    start = int(min_train_months)
    fold = 1
    while start < len(clean_dates):
        end = min(start + int(test_months), len(clean_dates))
        splits.append(ConditionalSDFSplit(fold=fold, train_dates=clean_dates[:start], test_dates=clean_dates[start:end]))
        start = end
        fold += 1
    return splits


def month_arrays(
    frame: pd.DataFrame,
    feature_cols: list[str],
    return_col: str = "next_ret",
) -> list[tuple[pd.Timestamp, np.ndarray, np.ndarray, np.ndarray]]:
    rows = []
    for date, group in frame.groupby("date", sort=True):
        x = group[feature_cols].to_numpy(dtype=np.float32)
        r = pd.to_numeric(group[return_col], errors="coerce").to_numpy(dtype=np.float32)
        ids = pd.to_numeric(group["permno"], errors="coerce").to_numpy(dtype=np.int64)
        ok = np.isfinite(x).all(axis=1) & np.isfinite(r) & np.isfinite(ids)
        if ok.any():
            rows.append((pd.Timestamp(date), x[ok], r[ok], ids[ok]))
    return rows


def sdf_pricing_error_summary(errors: pd.Series | np.ndarray) -> dict[str, float | int]:
    values = pd.to_numeric(pd.Series(errors), errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"n": 0, "mean_abs": np.nan, "rms": np.nan}
    return {
        "n": int(len(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values**2))),
    }
