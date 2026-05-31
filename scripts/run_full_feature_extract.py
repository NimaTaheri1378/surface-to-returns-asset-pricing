from __future__ import annotations

import argparse
import calendar
import json
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from surface_returns.manifest import read_json, utc_now_iso, write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs
from surface_returns.surfaces import clean_option_quotes, fixed_surface_grid, surface_features
from surface_returns.wrds_helpers import TableRef, connect_wrds


def month_bounds(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return int(len(frame))


def raw_sql(conn, sql: str) -> pd.DataFrame:
    return conn.raw_sql(sql, date_cols=["date", "exdate", "sdate", "edate"])


def month_end_dates(conn, option_ref: TableRef, start_date: str, end_date: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    option_date_df = raw_sql(
        conn,
        f"select max(date) as date from {option_ref.qualified} "
        f"where date between '{start_date}' and '{end_date}'",
    )
    crsp_date_df = raw_sql(
        conn,
        "select max(date) as date from crsp.msf "
        f"where date between '{start_date}' and '{end_date}'",
    )
    option_date = pd.Timestamp(option_date_df["date"].iloc[0])
    crsp_date = pd.Timestamp(crsp_date_df["date"].iloc[0])
    return (None if pd.isna(option_date) else option_date, None if pd.isna(crsp_date) else crsp_date)


def query_month_options(conn, option_ref: TableRef, option_date: pd.Timestamp, crsp_date: pd.Timestamp) -> pd.DataFrame:
    min_dte = option_date + pd.Timedelta(days=10)
    max_dte = option_date + pd.Timedelta(days=365)
    sql = f"""
        select
            o.secid,
            l.permno,
            o.date,
            o.exdate,
            o.cp_flag,
            o.strike_price,
            o.best_bid,
            o.best_offer,
            o.impl_volatility,
            o.delta,
            o.volume,
            o.open_interest
        from {option_ref.qualified} as o
        join wrdsapps.opcrsphist as l
          on o.secid = l.secid
         and l.permno is not null
         and (l.sdate is null or l.sdate <= o.date)
         and (l.edate is null or o.date <= l.edate)
        join crsp.msf as m
          on m.permno = l.permno
         and m.date = '{crsp_date.date()}'
        join crsp.msenames as n
          on n.permno = m.permno
         and n.namedt <= m.date
         and m.date <= coalesce(n.nameendt, '2099-12-31')
        where o.date = '{option_date.date()}'
          and o.exdate between '{min_dte.date()}' and '{max_dte.date()}'
          and o.impl_volatility is not null
          and o.best_bid > 0
          and o.best_offer > o.best_bid
          and n.shrcd in (10, 11)
          and n.exchcd in (1, 2, 3)
          and abs(m.prc) >= 5
          and abs(m.prc) * m.shrout * 1000 >= 250000000
    """
    return raw_sql(conn, sql)


def query_month_crsp(conn, start_date: str, end_date: str) -> pd.DataFrame:
    sql = f"""
        select m.permno, m.date, m.ret, m.prc, m.shrout, m.vol,
               n.shrcd, n.exchcd, n.siccd, n.ticker
        from crsp.msf as m
        join crsp.msenames as n
          on n.permno = m.permno
         and n.namedt <= m.date
         and m.date <= coalesce(n.nameendt, '2099-12-31')
        where m.date between '{start_date}' and '{end_date}'
          and n.shrcd in (10, 11)
          and n.exchcd in (1, 2, 3)
    """
    return raw_sql(conn, sql)


def build_month_features(options: pd.DataFrame, crsp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cleaned = clean_option_quotes(options)
    base_features = surface_features(cleaned)
    if base_features.empty:
        return base_features, pd.DataFrame(), pd.DataFrame()
    secid_permno = cleaned[["date", "secid", "permno"]].dropna().drop_duplicates()
    features = base_features.merge(secid_permno, on=["date", "secid"], how="left")
    features = features.merge(crsp[["permno", "date", "ret", "prc", "shrout", "vol"]], on=["permno", "date"], how="left")
    features["market_equity"] = features["prc"].abs() * features["shrout"] * 1000
    grid = fixed_surface_grid(
        cleaned,
        maturities=[30, 60, 90, 180, 270],
        deltas=[0.10, 0.25, 0.50, 0.75, 0.90],
    )
    if not grid.empty:
        grid = grid.merge(secid_permno, on=["date", "secid"], how="left")
    return features, grid, cleaned


def smoke_gate(root: Path) -> None:
    manifest_path = root / "manifests" / "wrds_smoke_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Cannot run full extraction: missing WRDS smoke manifest.")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise RuntimeError(f"Cannot run full extraction: smoke status is {manifest.get('status')!r}.")
    if not manifest.get("scale_gate", {}).get("approved_for_full_scale"):
        raise RuntimeError("Cannot run full extraction: scale gate did not approve full-scale run.")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"created_utc": utc_now_iso(), **payload}, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    smoke_gate(root)
    run_manifest_dir = root / "manifests" / "run_full"
    progress_log = root / "logs" / "full_feature_extract_progress.jsonl"
    months = [(year, month) for year in range(args.start_year, args.end_year + 1) for month in range(1, 13)]
    if args.max_shards:
        months = months[: args.max_shards]

    conn = connect_wrds()
    completed = 0
    failed = 0
    skipped = 0
    try:
        for year, month in months:
            shard_id = f"{year:04d}_{month:02d}"
            shard_manifest = run_manifest_dir / f"{shard_id}.json"
            feature_path = root / "data" / "processed" / "full" / "surface_features" / f"year={year:04d}" / f"month={month:02d}.parquet"
            grid_path = root / "data" / "processed" / "full" / "surface_grid" / f"year={year:04d}" / f"month={month:02d}.parquet"
            crsp_path = root / "data" / "processed" / "full" / "crsp_monthly" / f"year={year:04d}" / f"month={month:02d}.parquet"
            raw_path = root / "data" / "raw" / "full" / "option_quotes" / f"year={year:04d}" / f"month={month:02d}.parquet"
            if args.resume and shard_manifest.exists() and feature_path.exists() and grid_path.exists():
                prior = read_json(shard_manifest)
                if prior.get("status") == "PASS":
                    skipped += 1
                    continue
            option_ref = TableRef("optionm", f"opprcd{year:04d}")
            start_date, end_date = month_bounds(year, month)
            payload: dict[str, Any] = {
                "status": "STARTED",
                "year": year,
                "month": month,
                "option_table": option_ref.qualified,
                "start_date": start_date,
                "end_date": end_date,
            }
            try:
                option_date, crsp_date = month_end_dates(conn, option_ref, start_date, end_date)
                payload["option_date"] = str(option_date.date()) if option_date else None
                payload["crsp_date"] = str(crsp_date.date()) if crsp_date else None
                if option_date is None or crsp_date is None:
                    payload["status"] = "SKIPPED_NO_MONTH_END_DATE"
                    write_json_atomic(shard_manifest, payload)
                    append_jsonl(progress_log, payload)
                    skipped += 1
                    continue
                crsp = query_month_crsp(conn, start_date, end_date)
                options = query_month_options(conn, option_ref, option_date, crsp_date)
                features, grid, cleaned = build_month_features(options, crsp)
                payload["row_counts"] = {
                    "crsp_monthly": write_parquet_atomic(crsp, crsp_path),
                    "option_quotes_cleaned": int(len(cleaned)),
                    "surface_features": write_parquet_atomic(features, feature_path),
                    "surface_grid": write_parquet_atomic(grid, grid_path),
                }
                if args.keep_raw:
                    payload["row_counts"]["option_quotes_raw"] = write_parquet_atomic(options, raw_path)
                payload["artifacts"] = {
                    "surface_features": str(feature_path.relative_to(root)),
                    "surface_grid": str(grid_path.relative_to(root)),
                    "crsp_monthly": str(crsp_path.relative_to(root)),
                }
                payload["status"] = "PASS" if len(features) > 0 and len(grid) > 0 else "EMPTY_FEATURES"
                write_json_atomic(shard_manifest, payload)
                append_jsonl(progress_log, payload)
                if payload["status"] == "PASS":
                    completed += 1
                else:
                    failed += 1
            except Exception as exc:
                payload["status"] = "FAILED"
                payload["error"] = type(exc).__name__
                payload["error_message"] = str(exc)[:1000]
                payload["traceback_tail"] = traceback.format_exc().splitlines()[-20:]
                write_json_atomic(shard_manifest, payload)
                append_jsonl(progress_log, payload)
                failed += 1
                if args.fail_fast:
                    raise
        summary = {
            "status": "PASS" if completed > 0 and failed == 0 else "PARTIAL" if completed > 0 else "FAILED",
            "start_year": args.start_year,
            "end_year": args.end_year,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "keep_raw": bool(args.keep_raw),
            "progress_log": str(progress_log.relative_to(root)),
        }
        write_json_atomic(run_manifest_dir / "summary.json", summary)
        print("full_feature_extract_status=" + summary["status"])
        print(f"completed={completed} failed={failed} skipped={skipped}")
        return 0 if summary["status"] in {"PASS", "PARTIAL"} and completed > 0 else 2
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
