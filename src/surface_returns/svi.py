from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd

from surface_returns.surfaces import clean_option_quotes


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SSVIParams:
    rho: float
    eta: float
    lam: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(x) / sqrt(2.0)))


def _normal_icdf(p: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(p)


def raw_svi_total_variance(k: np.ndarray | pd.Series, params: SVIParams) -> np.ndarray:
    x = np.asarray(k, dtype=float) - params.m
    return params.a + params.b * (params.rho * x + np.sqrt(x * x + params.sigma * params.sigma))


def ssvi_phi(theta: np.ndarray | float, params: SSVIParams) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float).clip(1e-10, None)
    return params.eta * np.power(theta_arr, -params.lam)


def ssvi_total_variance(k: np.ndarray | pd.Series, theta: np.ndarray | float, params: SSVIParams) -> np.ndarray:
    k_arr = np.asarray(k, dtype=float)
    theta_arr = np.asarray(theta, dtype=float).clip(1e-10, None)
    phi = ssvi_phi(theta_arr, params)
    term = phi * k_arr + params.rho
    return 0.5 * theta_arr * (
        1.0
        + params.rho * phi * k_arr
        + np.sqrt(term * term + 1.0 - params.rho * params.rho)
    )


def black_call_price_normalized(k: np.ndarray, total_variance: np.ndarray) -> np.ndarray:
    """Black call price with forward normalized to 1 and strike exp(k)."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_variance, dtype=float).clip(1e-12, None)
    vol_sqrt = np.sqrt(w)
    d1 = (-k + 0.5 * w) / vol_sqrt
    d2 = d1 - vol_sqrt
    return _norm_cdf(d1) - np.exp(k) * _norm_cdf(d2)


def black_call_delta_from_k(k: np.ndarray, total_variance: np.ndarray) -> np.ndarray:
    w = np.asarray(total_variance, dtype=float).clip(1e-12, None)
    d1 = (-np.asarray(k, dtype=float) + 0.5 * w) / np.sqrt(w)
    return np.asarray(_norm_cdf(d1), dtype=float)


def infer_log_moneyness(options: pd.DataFrame) -> pd.Series:
    """Infer log strike/forward with a documented fallback when forward is unavailable."""
    frame = options.copy()
    strike = pd.to_numeric(frame["strike_price"], errors="coerce")
    forward_cols = [
        "forward_price",
        "forward",
        "spot_price",
        "sec_price",
        "underlying_price",
        "close",
        "prc",
    ]
    for col in forward_cols:
        if col in frame:
            forward = pd.to_numeric(frame[col], errors="coerce")
            if forward.notna().sum() > 0:
                return np.log(strike / forward.abs())
    if "delta" in frame:
        delta_gap = (pd.to_numeric(frame["delta"], errors="coerce").abs() - 0.5).abs()
        keys = [col for col in ["date", "secid", "dte"] if col in frame]
        atm = pd.Series(index=frame.index, dtype=float)
        for _, group in frame.assign(_delta_gap=delta_gap).groupby(keys, dropna=False):
            best_idx = group["_delta_gap"].idxmin()
            atm.loc[group.index] = strike.loc[best_idx]
        return np.log(strike / atm)
    return np.log(strike / strike.median())


def svi_static_arbitrage_checks(params: SVIParams, k_min: float = -1.0, k_max: float = 1.0) -> dict[str, bool | float]:
    grid = np.linspace(k_min, k_max, 301)
    w = raw_svi_total_variance(grid, params)
    calls = black_call_price_normalized(grid, w)
    strikes = np.exp(grid)
    slopes = np.diff(calls) / np.diff(strikes)
    slope_diffs = np.diff(slopes)
    min_second = float(np.nanmin(slope_diffs)) if len(slope_diffs) else float("nan")
    min_total_variance = float(np.nanmin(w)) if len(w) else float("nan")
    slope_left = params.b * (params.rho - 1.0)
    slope_right = params.b * (params.rho + 1.0)
    return {
        "positive_total_variance": bool(min_total_variance > 0),
        "convex_call_slice": bool(min_second >= -1e-6),
        "svi_slope_bounds": bool(abs(slope_left) <= 2.0 + 1e-8 and abs(slope_right) <= 2.0 + 1e-8),
        "min_total_variance": min_total_variance,
        "min_call_slope_difference": min_second,
    }


def _initial_params(k: np.ndarray, w: np.ndarray) -> SVIParams:
    a = float(np.nanpercentile(w, 10))
    b = float(max(1e-4, (np.nanpercentile(w, 90) - np.nanpercentile(w, 10)) / 2.0))
    m = float(np.nanmedian(k))
    sigma = float(max(0.05, np.nanstd(k)))
    return SVIParams(a=max(a, 1e-6), b=b, rho=-0.2, m=m, sigma=sigma)


def fit_svi_slice(quotes: pd.DataFrame, min_quotes: int = 8) -> tuple[SVIParams | None, dict[str, object]]:
    frame = clean_option_quotes(quotes).reset_index(drop=True)
    if frame.empty or len(frame) < min_quotes:
        return None, {"status": "SKIPPED_TOO_FEW_QUOTES", "quotes": int(len(frame))}
    frame = frame.dropna(subset=["impl_volatility", "dte", "strike_price"]).copy()
    if len(frame) < min_quotes:
        return None, {"status": "SKIPPED_TOO_FEW_VALID_QUOTES", "quotes": int(len(frame))}
    frame["log_moneyness"] = infer_log_moneyness(frame)
    frame["tau"] = pd.to_numeric(frame["dte"], errors="coerce") / 365.25
    frame["total_variance"] = pd.to_numeric(frame["impl_volatility"], errors="coerce").pow(2) * frame["tau"]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_moneyness", "total_variance"])
    if len(frame) < min_quotes:
        return None, {"status": "SKIPPED_TOO_FEW_FINITE_QUOTES", "quotes": int(len(frame))}

    k = frame["log_moneyness"].to_numpy(dtype=float)
    w = frame["total_variance"].to_numpy(dtype=float)
    weight = np.ones(len(frame), dtype=float)
    if "quoted_spread_pct" in frame:
        spread = pd.to_numeric(frame["quoted_spread_pct"], errors="coerce").fillna(frame["quoted_spread_pct"].median())
        weight = 1.0 / np.sqrt(spread.clip(0.01, 1.0).to_numpy(dtype=float))

    init = _initial_params(k, w)
    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        return None, {"status": "FAILED_SCIPY_UNAVAILABLE", "error": type(exc).__name__}

    k_min, k_max = float(np.nanmin(k) - 0.25), float(np.nanmax(k) + 0.25)
    dense_k = np.linspace(k_min, k_max, 101)

    def unpack(theta: np.ndarray) -> SVIParams:
        return SVIParams(
            a=float(theta[0]),
            b=float(theta[1]),
            rho=float(theta[2]),
            m=float(theta[3]),
            sigma=float(theta[4]),
        )

    def residual(theta: np.ndarray) -> np.ndarray:
        params = unpack(theta)
        fitted = raw_svi_total_variance(k, params)
        out = [(fitted - w) * weight]
        dense_w = raw_svi_total_variance(dense_k, params)
        calls = black_call_price_normalized(dense_k, dense_w)
        dense_strikes = np.exp(dense_k)
        slopes = np.diff(calls) / np.diff(dense_strikes)
        slope_diffs = np.diff(slopes)
        penalties = [
            np.minimum(dense_w - 1e-8, 0.0) * 1_000.0,
            np.minimum(slope_diffs + 1e-7, 0.0) * 1_000.0,
            np.array(
                [
                    max(0.0, abs(params.b * (1.0 + params.rho)) - 2.0),
                    max(0.0, abs(params.b * (1.0 - params.rho)) - 2.0),
                ]
            )
            * 100.0,
        ]
        out.extend(penalties)
        return np.concatenate(out)

    lower = np.array([1e-8, 1e-8, -0.999, k_min - 0.5, 1e-4])
    upper = np.array([max(float(np.nanmax(w) * 3.0), 0.05), 2.0, 0.999, k_max + 0.5, 5.0])
    x0 = np.array([init.a, init.b, init.rho, init.m, init.sigma], dtype=float)
    x0 = np.minimum(np.maximum(x0, lower + 1e-10), upper - 1e-10)
    result = least_squares(
        residual,
        x0=x0,
        bounds=(lower, upper),
        max_nfev=600,
        ftol=1e-8,
        xtol=1e-8,
    )
    params = unpack(result.x)
    checks = svi_static_arbitrage_checks(params, k_min=k_min, k_max=k_max)
    rmse = float(np.sqrt(np.nanmean((raw_svi_total_variance(k, params) - w) ** 2)))
    diagnostics: dict[str, object] = {
        "status": "PASS" if checks["positive_total_variance"] and checks["convex_call_slice"] else "FAILED_NO_ARB_CHECK",
        "quotes": int(len(frame)),
        "rmse_total_variance": rmse,
        "optimizer_success": bool(result.success),
        **checks,
    }
    return params, diagnostics


def solve_k_for_call_delta(params: SVIParams, target_delta: float, low: float = -2.0, high: float = 2.0) -> float:
    target_delta = float(np.clip(target_delta, 1e-4, 1 - 1e-4))
    lo, hi = low, high
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        w_mid = raw_svi_total_variance(np.array([mid]), params)
        delta_mid = float(black_call_delta_from_k(np.array([mid]), w_mid)[0])
        if delta_mid > target_delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _fit_svi_group(
    date: object,
    secid: object,
    group: pd.DataFrame,
    maturities: list[int],
    deltas: list[float],
    min_quotes: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []
    slice_outputs: list[dict[str, object]] = []
    for maturity in maturities:
        width = max(21, int(round(maturity * 0.35)))
        near = group[(group["dte"] - maturity).abs() <= width].copy()
        if len(near) < min_quotes:
            near = group.iloc[(group["dte"] - maturity).abs().argsort()[: max(min_quotes, 20)]].copy()
        try:
            params, diagnostics = fit_svi_slice(near, min_quotes=min_quotes)
        except Exception as exc:
            params = None
            diagnostics = {
                "status": "FAILED_FIT_EXCEPTION",
                "quotes": int(len(near)),
                "error": type(exc).__name__,
                "error_message": str(exc)[:300],
            }
        diag = {
            "date": date,
            "secid": secid,
            "target_dte": maturity,
            **diagnostics,
        }
        if params is not None:
            diag.update({f"svi_{key}": value for key, value in params.to_dict().items()})
        diag_rows.append(diag)
        if params is None:
            continue
        for delta in deltas:
            k_target = solve_k_for_call_delta(params, delta)
            total_variance = float(raw_svi_total_variance(np.array([k_target]), params)[0])
            slice_outputs.append(
                {
                    "date": date,
                    "secid": secid,
                    "target_dte": maturity,
                    "target_call_delta": float(delta),
                    "target_log_moneyness": float(k_target),
                    "total_variance": total_variance,
                    "impl_volatility": float(sqrt(max(total_variance / (maturity / 365.25), 1e-12))),
                    "svi_status": diagnostics["status"],
                }
            )
    if slice_outputs:
        out = pd.DataFrame(slice_outputs).sort_values(["target_call_delta", "target_dte"])
        adjusted = False
        for _delta, delta_group in out.groupby("target_call_delta"):
            idx = delta_group.sort_values("target_dte").index
            original = out.loc[idx, "total_variance"].to_numpy(dtype=float)
            monotone = np.maximum.accumulate(original)
            if np.any(monotone > original + 1e-10):
                adjusted = True
            out.loc[idx, "total_variance"] = monotone
            out.loc[idx, "impl_volatility"] = [
                sqrt(max(tv / (dte / 365.25), 1e-12))
                for tv, dte in zip(monotone, out.loc[idx, "target_dte"], strict=True)
            ]
        out["calendar_adjusted"] = adjusted
        rows.extend(out.to_dict("records"))
    return rows, diag_rows


def _fit_svi_group_task(args: tuple[object, object, pd.DataFrame, list[int], list[float], int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return _fit_svi_group(*args)


def fit_svi_surface_grid(
    options: pd.DataFrame,
    maturities: list[int],
    deltas: list[float],
    min_quotes: int = 8,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = clean_option_quotes(options)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    tasks = [
        (date, secid, group.copy(), maturities, deltas, min_quotes)
        for (date, secid), group in frame.groupby(["date", "secid"], dropna=False)
    ]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            outputs = list(executor.map(_fit_svi_group_task, tasks))
    else:
        outputs = [_fit_svi_group_task(task) for task in tasks]
    rows = [row for group_rows, _diag_rows in outputs for row in group_rows]
    diag_rows = [row for _group_rows, group_diag_rows in outputs for row in group_diag_rows]
    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def ssvi_no_arbitrage_bounds(theta: float, phi: float, rho: float) -> dict[str, bool | float]:
    left = theta * phi * (1.0 + abs(rho))
    right = theta * phi * phi * (1.0 + abs(rho))
    return {
        "ssvi_calendar_butterfly_bound_1": float(left),
        "ssvi_calendar_butterfly_bound_2": float(right),
        "ssvi_bounds_pass": bool(left < 4.0 and right < 4.0),
    }


def ssvi_surface_no_arbitrage_checks(
    theta_by_maturity: dict[int, float],
    params: SSVIParams,
    k_min: float = -2.0,
    k_max: float = 2.0,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    maturities = sorted(theta_by_maturity)
    theta = np.array([theta_by_maturity[item] for item in maturities], dtype=float)
    k_grid = np.linspace(k_min, k_max, 151)
    total_variance = np.vstack([ssvi_total_variance(k_grid, value, params) for value in theta])
    phi = ssvi_phi(theta, params)
    bounds = [ssvi_no_arbitrage_bounds(float(t), float(p), params.rho) for t, p in zip(theta, phi, strict=True)]
    bound_1 = np.array([item["ssvi_calendar_butterfly_bound_1"] for item in bounds], dtype=float)
    bound_2 = np.array([item["ssvi_calendar_butterfly_bound_2"] for item in bounds], dtype=float)
    if len(theta) > 1:
        calendar_diff = np.diff(total_variance, axis=0)
        theta_diff = np.diff(theta)
        theta_phi_diff = np.diff(theta * phi)
        min_calendar_diff = float(np.nanmin(calendar_diff))
        min_theta_diff = float(np.nanmin(theta_diff))
        min_theta_phi_diff = float(np.nanmin(theta_phi_diff))
    else:
        min_calendar_diff = np.nan
        min_theta_diff = np.nan
        min_theta_phi_diff = np.nan
    return {
        "positive_total_variance": bool(np.nanmin(total_variance) > 0),
        "theta_monotone": bool(len(theta) <= 1 or min_theta_diff >= -tolerance),
        "theta_phi_monotone": bool(len(theta) <= 1 or min_theta_phi_diff >= -tolerance),
        "calendar_monotone_grid": bool(len(theta) <= 1 or min_calendar_diff >= -tolerance),
        "ssvi_bounds_pass": bool(np.nanmax(bound_1) <= 4.0 + tolerance and np.nanmax(bound_2) <= 4.0 + tolerance),
        "min_total_variance": float(np.nanmin(total_variance)),
        "min_calendar_total_variance_diff": min_calendar_diff,
        "min_theta_diff": min_theta_diff,
        "min_theta_phi_diff": min_theta_phi_diff,
        "max_ssvi_bound_1": float(np.nanmax(bound_1)),
        "max_ssvi_bound_2": float(np.nanmax(bound_2)),
    }


def _initial_ssvi_theta(group: pd.DataFrame, maturities: list[int]) -> np.ndarray:
    theta = []
    for maturity in maturities:
        slice_df = group[group["target_dte"].eq(maturity)].copy()
        if slice_df.empty:
            theta.append(np.nan)
            continue
        atm = slice_df.iloc[(slice_df["target_call_delta"] - 0.5).abs().argsort()[:1]]
        theta.append(float(pd.to_numeric(atm["total_variance"], errors="coerce").iloc[0]))
    theta_arr = np.asarray(theta, dtype=float)
    if not np.isfinite(theta_arr).any():
        theta_arr = np.full(len(maturities), 0.04, dtype=float)
    median_theta = float(np.nanmedian(theta_arr[np.isfinite(theta_arr)]))
    theta_arr = np.where(np.isfinite(theta_arr), theta_arr, median_theta)
    return np.maximum.accumulate(np.clip(theta_arr, 1e-5, None))


def fit_ssvi_surface(
    grid: pd.DataFrame,
    min_points: int = 12,
    k_min: float = -2.0,
    k_max: float = 2.0,
) -> tuple[dict[str, object] | None, pd.DataFrame, dict[str, object]]:
    required = {"target_dte", "target_log_moneyness", "total_variance"}
    missing = required.difference(grid.columns)
    if missing:
        return None, pd.DataFrame(), {"status": "FAILED_MISSING_COLUMNS", "missing": sorted(missing)}
    frame = grid.copy()
    frame["target_dte"] = pd.to_numeric(frame["target_dte"], errors="coerce")
    frame["target_log_moneyness"] = pd.to_numeric(frame["target_log_moneyness"], errors="coerce")
    frame["total_variance"] = pd.to_numeric(frame["total_variance"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_dte", "target_log_moneyness", "total_variance"])
    frame = frame[frame["total_variance"] > 0].copy()
    if len(frame) < min_points or frame["target_dte"].nunique() < 2:
        return None, pd.DataFrame(), {
            "status": "SKIPPED_TOO_FEW_POINTS",
            "points": int(len(frame)),
            "maturities": int(frame["target_dte"].nunique()),
        }
    maturities = sorted(int(item) for item in frame["target_dte"].dropna().unique())
    maturity_index = {maturity: idx for idx, maturity in enumerate(maturities)}
    obs_maturity_index = frame["target_dte"].astype(int).map(maturity_index).to_numpy(dtype=int)
    k_obs = frame["target_log_moneyness"].to_numpy(dtype=float)
    w_obs = frame["total_variance"].to_numpy(dtype=float)
    theta0 = _initial_ssvi_theta(frame, maturities)
    if len(theta0) != len(maturities):
        theta0 = np.resize(theta0, len(maturities))

    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        return None, pd.DataFrame(), {"status": "FAILED_SCIPY_UNAVAILABLE", "error": type(exc).__name__}

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, SSVIParams]:
        theta_m = np.asarray(theta[: len(maturities)], dtype=float)
        params = SSVIParams(rho=float(theta[-3]), eta=float(theta[-2]), lam=float(theta[-1]))
        return theta_m, params

    def residual(theta: np.ndarray) -> np.ndarray:
        theta_m, params = unpack(theta)
        fitted = ssvi_total_variance(k_obs, theta_m[obs_maturity_index], params)
        scale = np.sqrt(np.maximum(w_obs, 1e-6))
        out = [(fitted - w_obs) / scale]
        phi = ssvi_phi(theta_m, params)
        if len(theta_m) > 1:
            out.append(np.minimum(np.diff(theta_m), 0.0) * 1_000.0)
            out.append(np.minimum(np.diff(theta_m * phi), 0.0) * 1_000.0)
            k_grid = np.linspace(k_min, k_max, 41)
            surfaces = np.vstack([ssvi_total_variance(k_grid, value, params) for value in theta_m])
            out.append(np.minimum(np.diff(surfaces, axis=0).ravel(), 0.0) * 500.0)
        bound_1 = theta_m * phi * (1.0 + abs(params.rho))
        bound_2 = theta_m * phi * phi * (1.0 + abs(params.rho))
        out.append(np.maximum(bound_1 - 3.999, 0.0) * 500.0)
        out.append(np.maximum(bound_2 - 3.999, 0.0) * 500.0)
        return np.concatenate(out)

    lower_theta = np.full(len(maturities), 1e-6, dtype=float)
    upper_theta = np.full(len(maturities), max(float(np.nanmax(w_obs) * 4.0), 0.10), dtype=float)
    x0 = np.concatenate([theta0, np.array([-0.30, 1.0, 0.30], dtype=float)])
    lower = np.concatenate([lower_theta, np.array([-0.999, 1e-4, 1e-4], dtype=float)])
    upper = np.concatenate([upper_theta, np.array([0.999, 20.0, 0.499], dtype=float)])
    x0 = np.minimum(np.maximum(x0, lower + 1e-10), upper - 1e-10)
    result = least_squares(
        residual,
        x0=x0,
        bounds=(lower, upper),
        max_nfev=800,
        ftol=1e-8,
        xtol=1e-8,
    )
    theta_fit_raw, params = unpack(result.x)
    theta_fit = np.maximum.accumulate(theta_fit_raw)
    theta_by_maturity = {maturity: float(theta_fit[idx]) for maturity, idx in maturity_index.items()}
    checks = ssvi_surface_no_arbitrage_checks(theta_by_maturity, params, k_min=k_min, k_max=k_max)
    fitted = ssvi_total_variance(k_obs, theta_fit[obs_maturity_index], params)
    rmse = float(np.sqrt(np.nanmean((fitted - w_obs) ** 2)))
    output_grid = frame.copy()
    output_grid["ssvi_total_variance"] = fitted
    output_grid["ssvi_impl_volatility"] = np.sqrt(
        np.maximum(output_grid["ssvi_total_variance"] / (output_grid["target_dte"] / 365.25), 1e-12)
    )
    output_grid["ssvi_residual_total_variance"] = output_grid["ssvi_total_variance"] - output_grid["total_variance"]
    pass_checks = all(
        bool(checks[key])
        for key in [
            "positive_total_variance",
            "theta_monotone",
            "theta_phi_monotone",
            "calendar_monotone_grid",
            "ssvi_bounds_pass",
        ]
    )
    diagnostics: dict[str, object] = {
        "status": "PASS" if pass_checks else "FAILED_NO_ARB_CHECK",
        "points": int(len(frame)),
        "maturities": int(len(maturities)),
        "rmse_total_variance": rmse,
        "optimizer_success": bool(result.success),
        "theta_projection_max_abs": float(np.nanmax(np.abs(theta_fit - theta_fit_raw))),
        **params.to_dict(),
        **{f"theta_{maturity}d": theta_by_maturity[maturity] for maturity in maturities},
        **checks,
    }
    return {"params": params, "theta_by_maturity": theta_by_maturity}, output_grid, diagnostics
