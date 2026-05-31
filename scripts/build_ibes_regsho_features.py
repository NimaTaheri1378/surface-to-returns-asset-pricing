from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from surface_returns.analyst_short import (
    aggregate_ibes_estimates,
    aggregate_short_volume,
    expand_ibes_link_to_months,
    merge_ibes_to_panel,
    merge_short_volume_to_panel,
)
from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs
from surface_returns.trading_costs import standardize_ticker
from surface_returns.wrds_helpers import connect_wrds, list_tables_safe


def panel_path(root: Path, explicit: str | None = None) -> Path:
    if explicit:
        return root / explicit
    candidates = [
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_taq_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_panel.parquet",
    ]
    return next((path for path in candidates if path.exists()), candidates[-1])


def chunked(values: list[object], size: int):
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


def literals(values: list[object]) -> str:
    clean = []
    for value in values:
        if isinstance(value, str):
            item = standardize_ticker(value)
            if item:
                clean.append("'" + item.replace("'", "''") + "'")
        else:
            clean.append(str(int(value)))
    return ", ".join(clean) if clean else "''"


def query_ibes_links(conn, permnos: list[int], start: str, end: str) -> pd.DataFrame:
    parts = []
    for chunk in chunked(permnos, 1000):
        sql = f"""
            select ticker, permno, ncusip, sdate, edate, score
            from wrdsapps_link_crsp_ibes.ibcrsphist
            where permno in ({literals(chunk)})
              and sdate <= '{end}'
              and coalesce(edate, '2099-12-31') >= '{start}'
        """
        part = conn.raw_sql(sql, date_cols=["sdate", "edate"])
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def query_ibes_stats(conn, tickers: list[str], start: str, end: str) -> pd.DataFrame:
    parts = []
    for chunk in chunked(tickers, 500):
        sql = f"""
            select ticker, statpers, measure, fpi, numest, numup, numdown, meanest, stdev
            from ibes.statsumu_epsus
            where statpers between '{start}' and '{end}'
              and measure = 'EPS'
              and ticker in ({literals(chunk)})
        """
        part = conn.raw_sql(sql, date_cols=["statpers"])
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def query_crsp_ticker_map(conn, permnos: list[int], start: str, end: str, panel_keys: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for chunk in chunked(permnos, 1000):
        sql = f"""
            select permno, namedt, nameenddt, ticker
            from crsp.stocknames
            where permno in ({literals(chunk)})
              and namedt <= '{end}'
              and coalesce(nameenddt, '2099-12-31') >= '{start}'
              and ticker is not null
        """
        part = conn.raw_sql(sql, date_cols=["namedt", "nameenddt"])
        if not part.empty:
            parts.append(part)
    names = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if names.empty:
        return pd.DataFrame(columns=["permno", "date", "ticker"])
    keys = panel_keys[["permno", "date"]].drop_duplicates().copy()
    keys["date"] = pd.to_datetime(keys["date"]).dt.to_period("M").dt.to_timestamp()
    names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
    names["nameenddt"] = pd.to_datetime(names["nameenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    merged = keys.merge(names, on="permno", how="left")
    merged = merged[(merged["namedt"] <= merged["date"]) & (merged["date"] <= merged["nameenddt"])]
    merged["ticker"] = merged["ticker"].map(standardize_ticker)
    return merged.dropna(subset=["ticker"]).drop_duplicates(["permno", "date"], keep="last")[["permno", "date", "ticker"]]


def query_short_volume(conn, tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, str | None]:
    libs = conn.list_libraries()
    if "wrds_shortvolume" in libs and "wrds_shortvolume" in list_tables_safe(conn, "wrds_shortvolume"):
        table = "wrds_shortvolume.wrds_shortvolume"
    elif "wrds_shortvolume_samp" in libs:
        table = "wrds_shortvolume_samp.wrds_shortvolume_samp"
    else:
        return pd.DataFrame(), None
    parts = []
    for chunk in chunked(tickers, 500):
        sql = f"""
            select *
            from {table}
            where date between '{start}' and '{end}'
              and symbol in ({literals(chunk)})
        """
        part = conn.raw_sql(sql, date_cols=["date"])
        if not part.empty:
            parts.append(part)
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default=None)
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--sample-permnos", type=int, default=0)
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    panel_file = panel_path(root, args.panel)
    if not panel_file.exists():
        raise FileNotFoundError(panel_file)
    panel = pd.read_parquet(panel_file)
    panel["date"] = pd.to_datetime(panel["date"]).dt.to_period("M").dt.to_timestamp()
    panel = panel[panel["date"].dt.year.between(args.start_year, args.end_year)].copy()
    if args.sample_permnos:
        keep = panel.groupby("permno").size().sort_values(ascending=False).head(args.sample_permnos).index
        panel = panel[panel["permno"].isin(keep)].copy()
    permnos = sorted(panel["permno"].dropna().astype(int).unique().tolist())
    start = f"{args.start_year}-01-01"
    end = f"{args.end_year}-12-31"

    conn = connect_wrds()
    try:
        links = query_ibes_links(conn, permnos, start, end)
        ibes_map = expand_ibes_link_to_months(links, panel[["permno", "date"]]) if not links.empty else pd.DataFrame()
        tickers = sorted(ibes_map["ticker"].dropna().unique().tolist()) if not ibes_map.empty else []
        ibes_raw = query_ibes_stats(conn, tickers, start, end) if tickers else pd.DataFrame()
        ibes_monthly = aggregate_ibes_estimates(ibes_raw)
        panel_ibes = merge_ibes_to_panel(panel, ibes_monthly, links) if not links.empty else panel.copy()

        ticker_map = query_crsp_ticker_map(conn, permnos, start, end, panel[["permno", "date"]])
        short_raw, short_table = query_short_volume(conn, sorted(ticker_map["ticker"].dropna().unique()), start, end)
        short_monthly = aggregate_short_volume(short_raw)
        out = merge_short_volume_to_panel(panel_ibes, short_monthly, ticker_map) if not ticker_map.empty else panel_ibes
    finally:
        conn.close()

    out_path = root / "data" / "processed" / "panel" / "surface_characteristic_state_ibes_regsho_panel.parquet"
    out.to_parquet(out_path, index=False)
    ibes_cols = [col for col in out.columns if col.startswith("ibes_")]
    short_cols = [col for col in out.columns if col.startswith("regsho_")]
    manifest = {
        "status": "PASS",
        "panel": str(panel_file.relative_to(root)),
        "panel_rows": int(len(out)),
        "sample_permnos": args.sample_permnos,
        "ibes_link_rows": int(len(links)),
        "ibes_raw_rows": int(len(ibes_raw)),
        "ibes_columns": ibes_cols,
        "ibes_missingness": {col: float(out[col].isna().mean()) for col in ibes_cols},
        "short_volume_table": short_table,
        "short_raw_rows": int(len(short_raw)),
        "short_columns": short_cols,
        "short_missingness": {col: float(out[col].isna().mean()) for col in short_cols},
        "artifacts": {"panel": str(out_path.relative_to(root))},
    }
    write_json_atomic(dirs["manifests"] / "ibes_regsho_manifest.json", manifest)
    print("ibes_regsho_status=PASS")
    print(f"panel_rows={manifest['panel_rows']} ibes_raw={manifest['ibes_raw_rows']} short_raw={manifest['short_raw_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
