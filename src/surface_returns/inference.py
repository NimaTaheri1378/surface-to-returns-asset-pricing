from __future__ import annotations

import numpy as np
import pandas as pd


def to_month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp()


def normalize_factor_units(factors: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = factors.copy()
    for col in columns:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if out[col].abs().median(skipna=True) > 0.5:
                out[col] = out[col] / 100.0
    return out


def newey_west_ols(y: pd.Series, x: pd.DataFrame, lags: int = 6) -> dict[str, object]:
    import statsmodels.api as sm

    frame = pd.concat([pd.Series(y, name="y"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) <= x.shape[1] + 2:
        return {"status": "SKIPPED_TOO_FEW_OBSERVATIONS", "nobs": int(len(frame))}
    y_clean = frame["y"].astype(float)
    x_clean = sm.add_constant(frame.drop(columns=["y"]).astype(float), has_constant="add")
    model = sm.OLS(y_clean, x_clean).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
    return {
        "status": "PASS",
        "nobs": int(model.nobs),
        "params": {str(key): float(value) for key, value in model.params.items()},
        "tstats": {str(key): float(value) for key, value in model.tvalues.items()},
        "pvalues": {str(key): float(value) for key, value in model.pvalues.items()},
        "rsquared": float(model.rsquared),
        "residuals": pd.Series(model.resid, index=frame.index),
    }


def factor_alpha_table(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    return_cols: list[str],
    factor_cols: list[str],
    date_col: str = "date",
    lags: int = 6,
) -> pd.DataFrame:
    frame = returns.copy()
    ff = factors.copy()
    frame[date_col] = to_month_start(frame[date_col])
    ff[date_col] = to_month_start(ff[date_col])
    factor_cols = [col for col in factor_cols if col in ff]
    conflict_cols = [col for col in factor_cols + ["rf"] if col in frame.columns]
    if conflict_cols:
        frame = frame.drop(columns=conflict_cols)
    ff = normalize_factor_units(ff, factor_cols + (["rf"] if "rf" in ff else []))
    merged = frame.merge(ff[[date_col] + factor_cols + (["rf"] if "rf" in ff else [])], on=date_col, how="inner")
    rows = []
    residuals = {}
    for col in return_cols:
        if col not in merged:
            continue
        y = pd.to_numeric(merged[col], errors="coerce")
        if "rf" in merged and not col.endswith("_ls"):
            y = y - pd.to_numeric(merged["rf"], errors="coerce")
        fit = newey_west_ols(y, merged[factor_cols], lags=lags)
        if fit["status"] != "PASS":
            rows.append({"portfolio": col, "status": fit["status"], "nobs": fit.get("nobs", 0)})
            continue
        residuals[col] = fit["residuals"]
        alpha = fit["params"].get("const", np.nan)
        rows.append(
            {
                "portfolio": col,
                "status": "PASS",
                "nobs": fit["nobs"],
                "alpha_monthly": alpha,
                "alpha_annualized": (1.0 + alpha) ** 12 - 1.0 if np.isfinite(alpha) else np.nan,
                "alpha_t_newey_west": fit["tstats"].get("const", np.nan),
                "alpha_p_newey_west": fit["pvalues"].get("const", np.nan),
                "rsquared": fit["rsquared"],
            }
        )
        for factor in factor_cols:
            rows[-1][f"beta_{factor}"] = fit["params"].get(factor, np.nan)
            rows[-1][f"t_{factor}"] = fit["tstats"].get(factor, np.nan)
    return pd.DataFrame(rows)


def grs_test(asset_returns: pd.DataFrame, factors: pd.DataFrame, asset_cols: list[str], factor_cols: list[str]) -> dict[str, float | int | None]:
    from scipy.stats import f as f_dist

    import statsmodels.api as sm

    left = asset_returns.copy()
    right = factors.copy()
    left["date"] = to_month_start(left["date"])
    right["date"] = to_month_start(right["date"])
    data = left.merge(right[["date"] + factor_cols + (["rf"] if "rf" in right else [])], on="date", how="inner")
    factor_cols = [col for col in factor_cols if col in data]
    data = normalize_factor_units(data, factor_cols + (["rf"] if "rf" in data else []))
    cols = ["date"] + asset_cols + factor_cols + (["rf"] if "rf" in data else [])
    data = data[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return {"status": "FAILED_NO_DATA", "nobs": 0}
    y = data[asset_cols].astype(float)
    if "rf" in data:
        y = y.sub(data["rf"].astype(float), axis=0)
    x = sm.add_constant(data[factor_cols].astype(float), has_constant="add")
    alphas = []
    residual_matrix = []
    for col in asset_cols:
        fit = sm.OLS(y[col], x).fit()
        alphas.append(float(fit.params["const"]))
        residual_matrix.append(fit.resid.to_numpy())
    eps = np.column_stack(residual_matrix)
    factors_matrix = data[factor_cols].astype(float).to_numpy()
    t_obs, n_assets = eps.shape
    k_factors = len(factor_cols)
    if t_obs <= n_assets + k_factors:
        return {"status": "SKIPPED_TOO_FEW_OBSERVATIONS", "nobs": int(t_obs)}
    sigma = np.cov(eps, rowvar=False, ddof=k_factors + 1)
    omega = np.cov(factors_matrix, rowvar=False, ddof=1)
    if omega.ndim == 0:
        omega = np.array([[float(omega)]])
    alpha_vec = np.asarray(alphas).reshape(-1, 1)
    mu_f = factors_matrix.mean(axis=0).reshape(-1, 1)
    sigma_inv = np.linalg.pinv(sigma)
    omega_inv = np.linalg.pinv(omega)
    numerator = float(np.asarray(alpha_vec.T @ sigma_inv @ alpha_vec).reshape(-1)[0])
    denominator = float(1.0 + np.asarray(mu_f.T @ omega_inv @ mu_f).reshape(-1)[0])
    grs = ((t_obs - n_assets - k_factors) / n_assets) * numerator / denominator
    pvalue = float(f_dist.sf(grs, n_assets, t_obs - n_assets - k_factors))
    hj_distance = float(np.sqrt(max(numerator, 0.0)))
    return {
        "status": "PASS",
        "nobs": int(t_obs),
        "n_assets": int(n_assets),
        "n_factors": int(k_factors),
        "grs_stat": float(grs),
        "grs_pvalue": pvalue,
        "hj_distance_monthly": hj_distance,
        "hj_distance_annualized": float(hj_distance * np.sqrt(12.0)),
        "mean_abs_alpha_monthly": float(np.mean(np.abs(alphas))),
    }


def moving_block_bootstrap_summary(
    returns: pd.Series,
    block_length: int = 6,
    n_boot: int = 1000,
    seed: int = 1378,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    ret = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(ret)
    if n == 0:
        return {"status": "FAILED_NO_RETURNS", "nobs": 0}
    block_length = max(1, min(int(block_length), n))
    means = []
    sharpes = []
    for _ in range(int(n_boot)):
        sample = []
        while len(sample) < n:
            start = int(rng.integers(0, n))
            block = [ret[(start + offset) % n] for offset in range(block_length)]
            sample.extend(block)
        sample_arr = np.asarray(sample[:n], dtype=float)
        mean = float(sample_arr.mean())
        std = float(sample_arr.std(ddof=1)) if n > 1 else np.nan
        means.append(mean)
        sharpes.append(mean / std * np.sqrt(12.0) if np.isfinite(std) and std > 0 else np.nan)
    means_arr = np.asarray(means)
    sharpes_arr = np.asarray(sharpes)
    return {
        "status": "PASS",
        "nobs": int(n),
        "block_length": int(block_length),
        "n_boot": int(n_boot),
        "mean_monthly": float(ret.mean()),
        "mean_monthly_ci_low": float(np.nanquantile(means_arr, 0.025)),
        "mean_monthly_ci_high": float(np.nanquantile(means_arr, 0.975)),
        "sharpe": float(ret.mean() / ret.std(ddof=1) * np.sqrt(12.0)) if n > 1 and ret.std(ddof=1) > 0 else np.nan,
        "sharpe_ci_low": float(np.nanquantile(sharpes_arr, 0.025)),
        "sharpe_ci_high": float(np.nanquantile(sharpes_arr, 0.975)),
    }


def decile_returns_from_predictions(
    predictions: pd.DataFrame,
    score_col: str = "pred",
    return_col: str = "next_ret",
    quantiles: int = 10,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["date"] = to_month_start(frame["date"])
    rows = []
    for date, group in frame.dropna(subset=[score_col, return_col]).groupby("date", sort=True):
        if len(group) < quantiles * 5:
            continue
        ranks = group[score_col].rank(method="first", pct=True)
        decile = np.ceil(ranks * quantiles).clip(1, quantiles).astype(int)
        means = group.assign(decile=decile).groupby("decile")[return_col].mean()
        row = {"date": date}
        for idx in range(1, quantiles + 1):
            row[f"decile_{idx}"] = float(means.get(idx, np.nan))
        row["decile_ls"] = row[f"decile_{quantiles}"] - row["decile_1"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
