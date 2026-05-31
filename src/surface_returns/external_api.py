from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd


FRED_SERIES = {
    "UNRATE": "fred_unrate",
    "CPIAUCSL": "fred_cpi",
    "INDPRO": "fred_indpro",
    "DGS10": "fred_dgs10",
    "DGS2": "fred_dgs2",
    "T10Y3M": "fred_t10y3m",
    "NFCI": "fred_nfci",
}

BLS_SERIES = {
    "LNS14000000": "bls_unrate",
    "CES0500000003": "bls_avg_hourly_earnings",
    "CUUR0000SA0": "bls_cpi_u",
}

SECRET_ENV_NAMES = [
    "FRED_API_KEY",
    "BLS_API_KEY",
    "BEA_API_KEY",
    "EIA_API_KEY",
    "SEC_EDGAR_USER_AGENT",
]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 surface-to-returns-research",
    "Accept": "application/json,text/csv,text/plain,*/*",
    "Connection": "close",
}


@dataclass(frozen=True)
class SourceStatus:
    source: str
    status: str
    rows: int = 0
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "status": self.status,
            "rows": int(self.rows),
            "detail": self.detail,
        }


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or "=" not in clean:
            continue
        key, value = clean.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def month_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp()


def _urlopen_text(
    url: str,
    timeout: int = 60,
    headers: dict[str, str] | None = None,
    retries: int = 1,
) -> str:
    merged_headers = dict(DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            request = urllib.request.Request(url, headers=merged_headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < max(1, retries):
                time.sleep(1.5 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("urlopen failed without exception")


def _urlopen_json(url: str, timeout: int = 60, headers: dict[str, str] | None = None) -> dict[str, object]:
    return json.loads(_urlopen_text(url, timeout=timeout, headers=headers))


def sanitize_detail(detail: object) -> str:
    text = str(detail)
    for name in SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value and len(value) >= 4:
            text = text.replace(value, "<redacted>")
    return text[:240]


def fetch_fred_series_api(series_id: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    query = urllib.parse.urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }
    )
    data = _urlopen_json(f"https://api.stlouisfed.org/fred/series/observations?{query}", timeout=90)
    observations = data.get("observations", [])
    rows = []
    for item in observations if isinstance(observations, list) else []:
        rows.append({"date": item.get("date"), series_id: item.get("value")})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["date", series_id])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[series_id] = pd.to_numeric(out[series_id], errors="coerce")
    return out.dropna(subset=["date"])


def fetch_fred_series_csv(series_id: str, start: str, end: str) -> pd.DataFrame:
    query = urllib.parse.urlencode({"id": series_id})
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{query}"
    text = _urlopen_text(url, timeout=90, retries=2)
    frame = pd.read_csv(StringIO(text))
    date_col = "observation_date" if "observation_date" in frame else "DATE"
    value_col = series_id if series_id in frame else frame.columns[-1]
    out = frame[[date_col, value_col]].rename(columns={date_col: "date", value_col: series_id})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[series_id] = pd.to_numeric(out[series_id], errors="coerce")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out = out[out["date"].between(start_ts, end_ts)].dropna(subset=["date"])
    return out


def fetch_fred_series(series_id: str, start: str, end: str, api_key: str | None = None) -> tuple[pd.DataFrame, str]:
    errors = []
    if api_key:
        try:
            return fetch_fred_series_api(series_id, start, end, api_key), "official API"
        except Exception as exc:
            errors.append(sanitize_detail(f"official API {type(exc).__name__}: {exc}"))
    try:
        return fetch_fred_series_csv(series_id, start, end), "fredgraph CSV"
    except Exception as exc:
        errors.append(sanitize_detail(f"fredgraph CSV {type(exc).__name__}: {exc}"))
    raise RuntimeError("; ".join(errors))


def fetch_fred_controls(start: str, end: str) -> tuple[pd.DataFrame, SourceStatus]:
    frames = []
    routes = []
    api_key = os.environ.get("FRED_API_KEY")
    for series_id, col in FRED_SERIES.items():
        item, route = fetch_fred_series(series_id, start, end, api_key=api_key)
        routes.append(route)
        item = item.rename(columns={series_id: col})
        item["month"] = month_key(item["date"])
        item = item.sort_values("date").drop_duplicates("month", keep="last")[["month", col]]
        frames.append(item)
    out = frames[0]
    for item in frames[1:]:
        out = out.merge(item, on="month", how="outer")
    out = out.sort_values("month").reset_index(drop=True)
    if {"fred_dgs10", "fred_dgs2"}.issubset(out.columns):
        out["fred_10y2y_spread"] = out["fred_dgs10"] - out["fred_dgs2"]
    if "fred_cpi" in out:
        out["fred_cpi_yoy"] = out["fred_cpi"].pct_change(12)
    if "fred_indpro" in out:
        out["fred_indpro_yoy"] = out["fred_indpro"].pct_change(12)
    detail = ",".join(sorted(set(routes)))
    return out, SourceStatus("FRED", "PASS", rows=len(out), detail=detail)


def _bls_post(payload: dict[str, object]) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "surface-to-returns-research"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def _year_chunks(start_year: int, end_year: int, width: int = 10) -> list[tuple[int, int]]:
    return [(year, min(year + width - 1, end_year)) for year in range(start_year, end_year + 1, width)]


def fetch_bls_controls(start_year: int, end_year: int, registration_key: str | None = None) -> tuple[pd.DataFrame, SourceStatus]:
    rows = []
    for chunk_start, chunk_end in _year_chunks(start_year, end_year):
        payload: dict[str, object] = {
            "seriesid": list(BLS_SERIES.keys()),
            "startyear": str(chunk_start),
            "endyear": str(chunk_end),
        }
        if registration_key:
            payload["registrationkey"] = registration_key
        result = _bls_post(payload)
        if result.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(str(result.get("message") or result.get("status")))
        for series in result.get("Results", {}).get("series", []):
            series_id = series.get("seriesID")
            col = BLS_SERIES.get(series_id)
            if not col:
                continue
            for item in series.get("data", []):
                period = item.get("period", "")
                if not str(period).startswith("M"):
                    continue
                year = int(item["year"])
                month = int(str(period)[1:])
                rows.append({"month": pd.Timestamp(year=year, month=month, day=1), col: item.get("value")})
    if not rows:
        return pd.DataFrame(columns=["month"]), SourceStatus("BLS", "EMPTY", rows=0)
    frame = pd.DataFrame(rows)
    for col in BLS_SERIES.values():
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    out = frame.groupby("month", as_index=False).last().sort_values("month")
    if "bls_cpi_u" in out:
        out["bls_cpi_u_yoy"] = out["bls_cpi_u"].pct_change(12)
    return out, SourceStatus("BLS", "PASS", rows=len(out), detail="public API v2")


def fetch_bea_controls(start_year: int, end_year: int, api_key: str | None) -> tuple[pd.DataFrame, SourceStatus]:
    if not api_key:
        return pd.DataFrame(columns=["month"]), SourceStatus("BEA", "SKIPPED_MISSING_ENV", detail="BEA_API_KEY")
    params = {
        "UserID": api_key,
        "method": "GetData",
        "datasetname": "NIPA",
        "TableName": "T10101",
        "LineNumber": "1",
        "Frequency": "Q",
        "Year": "ALL",
        "ResultFormat": "JSON",
    }
    data = _urlopen_json(f"https://apps.bea.gov/api/data/?{urllib.parse.urlencode(params)}", timeout=90)
    rows = data.get("BEAAPI", {}).get("Results", {}).get("Data", [])  # type: ignore[union-attr]
    parsed = []
    for row in rows if isinstance(rows, list) else []:
        period = str(row.get("TimePeriod", ""))
        if "Q" not in period:
            continue
        year_text, quarter_text = period.split("Q", 1)
        try:
            year = int(year_text)
            quarter = int(quarter_text)
        except ValueError:
            continue
        if year < start_year or year > end_year:
            continue
        value = pd.to_numeric(str(row.get("DataValue", "")).replace(",", ""), errors="coerce")
        parsed.append({"month": pd.Timestamp(year=year, month=quarter * 3, day=1), "bea_gdp": value})
    if not parsed:
        return pd.DataFrame(columns=["month"]), SourceStatus("BEA", "EMPTY", rows=0, detail="NIPA T10101 line 1")
    quarterly = pd.DataFrame(parsed).dropna(subset=["bea_gdp"]).sort_values("month")
    quarterly["bea_gdp_yoy"] = quarterly["bea_gdp"].pct_change(4)
    months = pd.DataFrame({"month": pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")})
    out = pd.merge_asof(months, quarterly, on="month", direction="backward")
    return out, SourceStatus("BEA", "PASS", rows=len(out), detail="NIPA T10101 line 1, current vintage")


def fetch_eia_controls_v2(start_year: int, end_year: int, api_key: str) -> tuple[pd.DataFrame, SourceStatus]:
    params = [
        ("api_key", api_key),
        ("frequency", "monthly"),
        ("data[0]", "value"),
        ("facets[series][]", "RWTC"),
        ("start", f"{start_year}-01"),
        ("end", f"{end_year}-12"),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("offset", "0"),
        ("length", "5000"),
    ]
    data = _urlopen_json(
        f"https://api.eia.gov/v2/petroleum/pri/spt/data/?{urllib.parse.urlencode(params)}",
        timeout=90,
    )
    rows = data.get("response", {}).get("data", [])  # type: ignore[union-attr]
    parsed = []
    for item in rows if isinstance(rows, list) else []:
        period = str(item.get("period", ""))
        if len(period) < 7:
            continue
        try:
            month = pd.Timestamp(f"{period[:7]}-01")
        except ValueError:
            continue
        parsed.append({"month": month, "eia_wti_spot": item.get("value")})
    if not parsed:
        return pd.DataFrame(columns=["month"]), SourceStatus("EIA", "EMPTY", rows=0, detail="v2 petroleum/pri/spt RWTC")
    out = pd.DataFrame(parsed).sort_values("month")
    out["eia_wti_spot"] = pd.to_numeric(out["eia_wti_spot"], errors="coerce")
    out["eia_wti_spot_return"] = out["eia_wti_spot"].pct_change()
    return out, SourceStatus("EIA", "PASS", rows=len(out), detail="v2 petroleum/pri/spt RWTC")


def fetch_eia_controls_legacy(start_year: int, end_year: int, api_key: str) -> tuple[pd.DataFrame, SourceStatus]:
    params = {"api_key": api_key, "series_id": "PET.RWTC.M"}
    data = _urlopen_json(f"https://api.eia.gov/series/?{urllib.parse.urlencode(params)}", timeout=90)
    series = data.get("series", [])
    rows = series[0].get("data", []) if isinstance(series, list) and series else []
    parsed = []
    for period, value in rows if isinstance(rows, list) else []:
        period_text = str(period)
        if len(period_text) < 6:
            continue
        try:
            year = int(period_text[:4])
            month = int(period_text[4:6])
        except ValueError:
            continue
        if start_year <= year <= end_year:
            parsed.append({"month": pd.Timestamp(year=year, month=month, day=1), "eia_wti_spot": value})
    if not parsed:
        return pd.DataFrame(columns=["month"]), SourceStatus("EIA", "EMPTY", rows=0, detail="legacy PET.RWTC.M")
    out = pd.DataFrame(parsed).sort_values("month")
    out["eia_wti_spot"] = pd.to_numeric(out["eia_wti_spot"], errors="coerce")
    out["eia_wti_spot_return"] = out["eia_wti_spot"].pct_change()
    return out, SourceStatus("EIA", "PASS", rows=len(out), detail="legacy PET.RWTC.M")


def fetch_eia_controls(start_year: int, end_year: int, api_key: str | None) -> tuple[pd.DataFrame, SourceStatus]:
    if not api_key:
        return pd.DataFrame(columns=["month"]), SourceStatus("EIA", "SKIPPED_MISSING_ENV", detail="EIA_API_KEY")
    try:
        return fetch_eia_controls_v2(start_year, end_year, api_key)
    except Exception as exc:
        v2_error = sanitize_detail(f"v2 {type(exc).__name__}: {exc}")
    frame, status = fetch_eia_controls_legacy(start_year, end_year, api_key)
    return frame, SourceStatus(status.source, status.status, rows=status.rows, detail=f"{status.detail}; fallback after {v2_error}")


def fetch_sec_edgar_status(user_agent: str | None) -> SourceStatus:
    if not user_agent:
        return SourceStatus("SEC_EDGAR", "SKIPPED_MISSING_ENV", detail="SEC_EDGAR_USER_AGENT")
    data = _urlopen_json(
        "https://www.sec.gov/files/company_tickers.json",
        timeout=90,
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    )
    rows = len(data) if isinstance(data, dict) else 0
    return SourceStatus("SEC_EDGAR", "PASS" if rows else "EMPTY", rows=rows, detail="company_tickers metadata")


def build_external_controls(start_year: int, end_year: int) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"
    statuses: list[SourceStatus] = []
    frames: list[pd.DataFrame] = []
    try:
        fred, status = fetch_fred_controls(start, end)
        frames.append(fred)
        statuses.append(status)
    except Exception as exc:
        statuses.append(SourceStatus("FRED", "FAILED", detail=sanitize_detail(f"{type(exc).__name__}: {exc}")))
    try:
        bls, status = fetch_bls_controls(start_year, end_year, os.environ.get("BLS_API_KEY"))
        frames.append(bls)
        statuses.append(status)
    except Exception as exc:
        statuses.append(SourceStatus("BLS", "FAILED", detail=sanitize_detail(f"{type(exc).__name__}: {exc}")))
    for source, fetcher, key_name in [
        ("BEA", fetch_bea_controls, "BEA_API_KEY"),
        ("EIA", fetch_eia_controls, "EIA_API_KEY"),
    ]:
        try:
            frame, status = fetcher(start_year, end_year, os.environ.get(key_name))
            if not frame.empty and status.status == "PASS":
                frames.append(frame)
            statuses.append(status)
        except Exception as exc:
            statuses.append(SourceStatus(source, "FAILED", detail=sanitize_detail(f"{type(exc).__name__}: {exc}")))
    try:
        statuses.append(fetch_sec_edgar_status(os.environ.get("SEC_EDGAR_USER_AGENT")))
    except Exception as exc:
        statuses.append(SourceStatus("SEC_EDGAR", "FAILED", detail=sanitize_detail(f"{type(exc).__name__}: {exc}")))
    if not frames:
        return pd.DataFrame(columns=["month"]), [status.to_dict() for status in statuses]
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="month", how="outer")
    out = out.sort_values("month").reset_index(drop=True)
    return out, [status.to_dict() for status in statuses]
