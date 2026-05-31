from __future__ import annotations

import numpy as np
import pandas as pd


def top_abs_attributions(attributions: pd.DataFrame, value_col: str = "integrated_gradient", top_n: int = 20) -> pd.DataFrame:
    required = {"feature", value_col}
    missing = required.difference(attributions.columns)
    if missing:
        raise KeyError(f"Missing attribution columns: {sorted(missing)}")
    frame = attributions.copy()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    out = (
        frame.dropna(subset=[value_col])
        .groupby("feature", as_index=False)
        .agg(
            mean_attribution=(value_col, "mean"),
            mean_abs_attribution=(value_col, lambda s: float(np.mean(np.abs(s)))),
            attribution_vol=(value_col, "std"),
        )
        .sort_values("mean_abs_attribution", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return out


def latent_factor_correlations(
    latent_factors: pd.DataFrame,
    state_controls: pd.DataFrame,
    factor_prefix: str = "sdf_factor_",
) -> pd.DataFrame:
    left = latent_factors.copy()
    right = state_controls.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.to_period("M").dt.to_timestamp()
    date_col = "month" if "month" in right.columns and "date" not in right.columns else "date"
    right["date"] = pd.to_datetime(right[date_col]).dt.to_period("M").dt.to_timestamp()
    merged = left.merge(right.drop(columns=["month"], errors="ignore"), on="date", how="inner")
    factor_cols = [col for col in merged.columns if col.startswith(factor_prefix)]
    state_cols = [
        col
        for col in merged.columns
        if col not in {"date", "fold", *factor_cols}
        and pd.api.types.is_numeric_dtype(merged[col])
    ]
    rows = []
    for factor in factor_cols:
        for state in state_cols:
            sample = merged[[factor, state]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sample) < 6 or sample[factor].std(ddof=0) == 0 or sample[state].std(ddof=0) == 0:
                continue
            rows.append(
                {
                    "latent_factor": factor,
                    "state_variable": state,
                    "correlation": float(sample[factor].corr(sample[state])),
                    "nobs": int(len(sample)),
                }
            )
    columns = ["latent_factor", "state_variable", "correlation", "nobs"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def beta_characteristic_correlations(
    asset_panel: pd.DataFrame,
    characteristic_panel: pd.DataFrame,
    beta_prefix: str = "sdf_beta_",
) -> pd.DataFrame:
    left = asset_panel.copy()
    right = characteristic_panel.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.to_period("M").dt.to_timestamp()
    right["date"] = pd.to_datetime(right["date"]).dt.to_period("M").dt.to_timestamp()
    beta_cols = [col for col in left.columns if col.startswith(beta_prefix)]
    characteristic_cols = [
        "mean_iv",
        "atm_iv",
        "put_call_iv_spread",
        "momentum_12_2",
        "book_to_market_comp",
        "gross_profitability",
        "investment",
        "beta_252d",
        "idio_vol_252d",
        "vix_eom",
        "fred_unrate",
        "fred_cpi_yoy",
        "fred_indpro_yoy",
        "fred_10y2y_spread",
        "bls_cpi_u_yoy",
        "eia_wti_spot_return",
    ]
    characteristic_cols = [col for col in characteristic_cols if col in right]
    merged = left[["date", "permno", *beta_cols]].merge(
        right[["date", "permno", *characteristic_cols]].drop_duplicates(["date", "permno"]),
        on=["date", "permno"],
        how="inner",
    )
    rows = []
    for beta in beta_cols:
        for char in characteristic_cols:
            sample = merged[[beta, char]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sample) < 100 or sample[beta].std(ddof=0) == 0 or sample[char].std(ddof=0) == 0:
                continue
            rows.append(
                {
                    "latent_beta": beta,
                    "characteristic": char,
                    "correlation": float(sample[beta].corr(sample[char])),
                    "nobs": int(len(sample)),
                }
            )
    columns = ["latent_beta", "characteristic", "correlation", "nobs"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
