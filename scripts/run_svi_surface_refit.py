from __future__ import annotations

import argparse
import calendar
import traceback
from pathlib import Path

import pandas as pd

from surface_returns.manifest import utc_now_iso, write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs
from surface_returns.svi import fit_svi_surface_grid
from surface_returns.wrds_helpers import TableRef, connect_wrds


def month_bounds(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"


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


def query_month_options(
    conn,
    option_ref: TableRef,
    option_date: pd.Timestamp,
    crsp_date: pd.Timestamp,
    max_securities: int,
) -> pd.DataFrame:
    min_dte = option_date + pd.Timedelta(days=10)
    max_dte = option_date + pd.Timedelta(days=365)
    limit_clause = ""
    security_filter = ""
    if max_securities > 0:
        security_filter = f"""
          and o.secid in (
              select secid from (
                  select o2.secid, count(*) as n_quotes
                  from {option_ref.qualified} as o2
                  join wrdsapps.opcrsphist as l2
                    on o2.secid = l2.secid
                   and l2.permno is not null
                   and (l2.sdate is null or l2.sdate <= o2.date)
                   and (l2.edate is null or o2.date <= l2.edate)
                  join crsp.msf as m2
                    on m2.permno = l2.permno
                   and m2.date = '{crsp_date.date()}'
                  join crsp.msenames as n2
                    on n2.permno = m2.permno
                   and n2.namedt <= m2.date
                   and m2.date <= coalesce(n2.nameendt, '2099-12-31')
                  where o2.date = '{option_date.date()}'
                    and o2.exdate between '{min_dte.date()}' and '{max_dte.date()}'
                    and o2.impl_volatility is not null
                    and o2.best_bid > 0
                    and o2.best_offer > o2.best_bid
                    and n2.shrcd in (10, 11)
                    and n2.exchcd in (1, 2, 3)
                    and abs(m2.prc) >= 5
                    and abs(m2.prc) * m2.shrout * 1000 >= 250000000
                  group by o2.secid
                  order by n_quotes desc
                  limit {max_securities}
              ) ranked
          )
        """
    elif max_securities == 0:
        limit_clause = ""
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
            o.open_interest,
            m.prc,
            m.shrout,
            m.vol,
            n.siccd,
            n.ticker
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
          {security_filter}
        {limit_clause}
    """
    return raw_sql(conn, sql)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return int(len(frame))


def run_month(args: argparse.Namespace, conn, root: Path, year: int, month: int) -> dict[str, object]:
    option_ref = TableRef("optionm", f"opprcd{year:04d}")
    start_date, end_date = month_bounds(year, month)
    manifest: dict[str, object] = {
        "status": "STARTED",
        "created_utc": utc_now_iso(),
        "year": year,
        "month": month,
        "option_table": option_ref.qualified,
        "max_securities": args.max_securities,
        "maturities": args.maturities,
        "deltas": args.deltas,
        "workers": args.workers,
    }
    option_date, crsp_date = month_end_dates(conn, option_ref, start_date, end_date)
    manifest["option_date"] = str(option_date.date()) if option_date is not None else None
    manifest["crsp_date"] = str(crsp_date.date()) if crsp_date is not None else None
    if option_date is None or crsp_date is None:
        manifest["status"] = "SKIPPED_NO_MONTH_END_DATE"
        return manifest

    options = query_month_options(conn, option_ref, option_date, crsp_date, args.max_securities)
    grid, diagnostics = fit_svi_surface_grid(
        options,
        args.maturities,
        args.deltas,
        min_quotes=args.min_quotes,
        workers=args.workers,
    )
    secid_permno = options[["date", "secid", "permno"]].dropna().drop_duplicates()
    if not grid.empty:
        grid = grid.merge(secid_permno, on=["date", "secid"], how="left")
    if not diagnostics.empty:
        diagnostics = diagnostics.merge(secid_permno, on=["date", "secid"], how="left")

    grid_path = root / "data" / "processed" / "full" / "svi_surface_grid" / f"year={year:04d}" / f"month={month:02d}.parquet"
    diag_path = root / "data" / "processed" / "full" / "svi_surface_diagnostics" / f"year={year:04d}" / f"month={month:02d}.parquet"
    manifest["row_counts"] = {
        "option_quotes": int(len(options)),
        "svi_grid": write_parquet_atomic(grid, grid_path),
        "svi_diagnostics": write_parquet_atomic(diagnostics, diag_path),
    }
    status_counts = diagnostics["status"].value_counts(dropna=False).to_dict() if "status" in diagnostics else {}
    manifest["diagnostic_status_counts"] = {str(key): int(value) for key, value in status_counts.items()}
    manifest["calendar_adjusted_rows"] = int(grid.get("calendar_adjusted", pd.Series(dtype=bool)).sum()) if not grid.empty else 0
    manifest["artifacts"] = {
        "svi_grid": str(grid_path.relative_to(root)),
        "svi_diagnostics": str(diag_path.relative_to(root)),
    }
    manifest["status"] = "PASS" if len(grid) > 0 and status_counts.get("PASS", 0) > 0 else "PARTIAL"
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--months", type=int, nargs="*", default=[12])
    parser.add_argument("--max-securities", type=int, default=100)
    parser.add_argument("--min-quotes", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--maturities", type=int, nargs="+", default=[30, 60, 90, 180, 270])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.10, 0.25, 0.50, 0.75, 0.90])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    ensure_project_dirs(root)
    manifest_dir = root / "manifests" / "svi_refit"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    conn = connect_wrds()
    statuses: list[str] = []
    try:
        for year in range(args.start_year, args.end_year + 1):
            months = args.months if args.months else list(range(1, 13))
            for month in months:
                manifest_path = manifest_dir / f"{year:04d}_{month:02d}.json"
                if args.resume and manifest_path.exists():
                    continue
                try:
                    manifest = run_month(args, conn, root, year, month)
                except Exception as exc:
                    manifest = {
                        "status": "FAILED",
                        "created_utc": utc_now_iso(),
                        "year": year,
                        "month": month,
                        "error": type(exc).__name__,
                        "error_message": str(exc)[:1000],
                        "traceback_tail": traceback.format_exc().splitlines()[-20:],
                    }
                statuses.append(str(manifest["status"]))
                write_json_atomic(manifest_path, manifest)
                print(f"svi_refit {year}-{month:02d} status={manifest['status']}", flush=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    summary = {
        "status": "PASS" if statuses and all(status == "PASS" for status in statuses) else "PARTIAL",
        "created_utc": utc_now_iso(),
        "months": len(statuses),
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
    }
    write_json_atomic(manifest_dir / "summary.json", summary)
    print("svi_refit_status=" + summary["status"])
    return 0 if summary["status"] in {"PASS", "PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
