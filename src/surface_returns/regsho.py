from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd

from surface_returns.trading_costs import standardize_ticker


SEC_CATEGORY_A_URL = "https://www.sec.gov/spotlight/shopilot/currentpilota41305.txt"
SEC_PILOT_ORDER_URL = "https://www.sec.gov/files/rules/other/34-50104.htm"
REGSHO_PILOT_START = pd.Timestamp("2005-05-01")
REGSHO_PILOT_END = pd.Timestamp("2006-04-01")


@dataclass(frozen=True)
class OLSResult:
    params: dict[str, float]
    se_hc1: dict[str, float]
    se_cluster_date: dict[str, float]
    se_cluster_entity: dict[str, float]
    nobs: int
    r2: float

    def tstat(self, name: str, se: str = "cluster_date") -> float:
        se_map = {
            "hc1": self.se_hc1,
            "cluster_date": self.se_cluster_date,
            "cluster_entity": self.se_cluster_entity,
        }[se]
        denom = se_map.get(name, np.nan)
        return float(self.params.get(name, np.nan) / denom) if np.isfinite(denom) and denom > 0 else np.nan


def fetch_text(url: str, timeout: int = 60) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 surface-to-returns-research",
            "Accept": "text/plain,text/html,*/*",
            "Connection": "close",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def parse_category_a_pilot_text(text: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(text), sep="|", dtype=str)
    lower = {col.lower(): col for col in frame.columns}
    if not {"symbol", "security_name"}.issubset(lower):
        raise ValueError("Category A pilot text does not look pipe-delimited.")
    out = frame.rename(columns={lower["symbol"]: "ticker", lower["security_name"]: "security_name"})
    exchange_col = lower.get("exchange")
    keep_cols = ["ticker", "security_name"] + ([exchange_col] if exchange_col else [])
    out = out[keep_cols].rename(columns={exchange_col: "exchange"} if exchange_col else {})
    out["ticker"] = out["ticker"].map(standardize_ticker)
    out = out[out["ticker"].notna()].copy()
    out["pilot_category"] = "A"
    return out.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def parse_appendix_a_from_order_html(text: str) -> pd.DataFrame:
    clean = re.sub(r"<[^>]+>", "\n", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in clean.splitlines()]
    rows = []
    in_appendix = False
    for line in lines:
        if line == "APPENDIX A":
            in_appendix = True
            continue
        if not in_appendix:
            continue
        if line.startswith("17 CFR") or line.startswith('"Short sale"'):
            break
        match = re.match(r"^([A-Z][A-Z0-9.]{0,6})\s+(.+)$", line)
        if not match:
            continue
        ticker = standardize_ticker(match.group(1))
        name = match.group(2).strip()
        if ticker and ticker not in {"TICKER", "SYMBOL"} and len(name) > 2:
            rows.append({"ticker": ticker, "security_name": name, "pilot_category": "A"})
    return pd.DataFrame(rows).drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def load_category_a_pilot_list() -> tuple[pd.DataFrame, str]:
    try:
        return parse_category_a_pilot_text(fetch_text(SEC_CATEGORY_A_URL)), SEC_CATEGORY_A_URL
    except Exception:
        return parse_appendix_a_from_order_html(fetch_text(SEC_PILOT_ORDER_URL)), SEC_PILOT_ORDER_URL


def month_standardize(frame: pd.DataFrame, col: str) -> pd.Series:
    values = pd.to_numeric(frame[col], errors="coerce")

    def _scale(s: pd.Series) -> pd.Series:
        std = pd.to_numeric(s, errors="coerce").std(ddof=0)
        try:
            std_value = float(std)
        except (TypeError, ValueError):
            std_value = np.nan
        if not np.isfinite(std_value) or std_value == 0:
            return pd.Series(0.0, index=s.index)
        return (s - pd.to_numeric(s, errors="coerce").mean()) / std_value

    return values.groupby(pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()).transform(_scale)


def prepare_regsho_did_frame(
    panel: pd.DataFrame,
    pilot_list: pd.DataFrame,
    signal_col: str = "put_call_iv_spread",
    return_col: str = "next_ret",
    window_start: str = "2004-01-01",
    window_end: str = "2006-12-31",
    controls: list[str] | None = None,
) -> pd.DataFrame:
    controls = controls or []
    required = {"date", "permno", "ticker", signal_col, return_col}.union(controls)
    missing = required.difference(panel.columns)
    if missing:
        raise KeyError(f"Missing Reg SHO DID columns: {sorted(missing)}")
    data = panel[["date", "permno", "ticker", signal_col, return_col, *controls]].copy()
    data["date"] = pd.to_datetime(data["date"]).dt.to_period("M").dt.to_timestamp()
    data = data[data["date"].between(pd.Timestamp(window_start), pd.Timestamp(window_end))].copy()
    data["ticker"] = data["ticker"].map(standardize_ticker)
    pilot_tickers = set(pilot_list["ticker"].map(standardize_ticker).dropna())
    data["regsho_pilot"] = data["ticker"].isin(pilot_tickers).astype(float)
    data["regsho_post"] = data["date"].between(REGSHO_PILOT_START, REGSHO_PILOT_END).astype(float)
    data["surface_signal"] = month_standardize(data, signal_col)
    data[return_col] = pd.to_numeric(data[return_col], errors="coerce")
    for col in controls:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        global_median = data[col].median(skipna=True)
        if not np.isfinite(global_median):
            global_median = 0.0
        data[col] = data[col].fillna(data.groupby("date")[col].transform("median")).fillna(global_median)
        data[f"{col}_z"] = month_standardize(data, col)
    data["pilot_post"] = data["regsho_pilot"] * data["regsho_post"]
    data["signal_x_pilot"] = data["surface_signal"] * data["regsho_pilot"]
    data["signal_x_post"] = data["surface_signal"] * data["regsho_post"]
    data["signal_x_pilot_x_post"] = data["surface_signal"] * data["regsho_pilot"] * data["regsho_post"]
    needed = [return_col, "surface_signal", "signal_x_pilot", "signal_x_post", "signal_x_pilot_x_post", "pilot_post"]
    needed += [f"{col}_z" for col in controls]
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=needed + ["permno", "date"]).reset_index(drop=True)


def two_way_demean(frame: pd.DataFrame, cols: list[str], entity_col: str, time_col: str, iterations: int = 20) -> pd.DataFrame:
    out = frame[cols].astype(float).copy()
    for _ in range(iterations):
        out -= out.groupby(frame[entity_col]).transform("mean")
        out -= out.groupby(frame[time_col]).transform("mean")
    return out


def _cluster_se(x: np.ndarray, resid: np.ndarray, xtx_inv: np.ndarray, clusters: pd.Series) -> np.ndarray:
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    score = x * resid.reshape(-1, 1)
    for _, idx in clusters.groupby(clusters).groups.items():
        summed = score[list(idx)].sum(axis=0).reshape(-1, 1)
        meat += summed @ summed.T
    g = clusters.nunique()
    n, k = x.shape
    correction = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov = correction * xtx_inv @ meat @ xtx_inv
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def fit_regsho_did(
    frame: pd.DataFrame,
    return_col: str = "next_ret",
    controls: list[str] | None = None,
) -> OLSResult:
    controls = controls or []
    regressors = ["surface_signal", "signal_x_pilot", "signal_x_post", "signal_x_pilot_x_post", "pilot_post"]
    regressors += [f"{col}_z" for col in controls]
    cols = [return_col, *regressors]
    demeaned = two_way_demean(frame, cols, entity_col="permno", time_col="date")
    y = demeaned[return_col].to_numpy(dtype=float)
    x = demeaned[regressors].to_numpy(dtype=float)
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y = y[keep]
    x = x[keep]
    used = frame.loc[keep].reset_index(drop=True)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    resid = y - fitted
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = (x * resid.reshape(-1, 1)).T @ (x * resid.reshape(-1, 1))
    cov_hc1 = (n / max(n - k, 1)) * xtx_inv @ meat @ xtx_inv
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / total if total > 0 else np.nan
    return OLSResult(
        params=dict(zip(regressors, beta, strict=True)),
        se_hc1=dict(zip(regressors, np.sqrt(np.maximum(np.diag(cov_hc1), 0.0)), strict=True)),
        se_cluster_date=dict(zip(regressors, _cluster_se(x, resid, xtx_inv, used["date"]), strict=True)),
        se_cluster_entity=dict(zip(regressors, _cluster_se(x, resid, xtx_inv, used["permno"]), strict=True)),
        nobs=int(n),
        r2=float(r2),
    )


def event_study_surface_spreads(
    frame: pd.DataFrame,
    return_col: str = "next_ret",
    quantiles: int = 3,
) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("date", sort=True):
        for pilot, sub in group.groupby("regsho_pilot"):
            sub = sub.copy()
            if sub["surface_signal"].nunique() < quantiles or len(sub) < quantiles * 5:
                continue
            sub["bucket"] = pd.qcut(sub["surface_signal"], quantiles, labels=False, duplicates="drop")
            if sub["bucket"].nunique() < quantiles:
                continue
            high = sub.loc[sub["bucket"].eq(sub["bucket"].max()), return_col].mean()
            low = sub.loc[sub["bucket"].eq(sub["bucket"].min()), return_col].mean()
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "regsho_pilot": int(pilot),
                    "high_minus_low": float(high - low),
                    "n_assets": int(len(sub)),
                    "post": int(pd.Timestamp(date) >= REGSHO_PILOT_START and pd.Timestamp(date) <= REGSHO_PILOT_END),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["event_month"] = (out["date"].dt.year - 2005) * 12 + (out["date"].dt.month - 5)
    return out
