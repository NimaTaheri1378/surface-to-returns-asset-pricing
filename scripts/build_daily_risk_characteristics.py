from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from surface_returns.characteristics import (
    collapse_daily_risk_to_panel_months,
    compute_daily_risk_characteristics,
)
from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs
from surface_returns.wrds_helpers import connect_wrds


def safe_suffix(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    if text and not text.startswith("_"):
        text = "_" + text
    return text


def chunked(values: list[int], size: int) -> Iterable[list[int]]:
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


def query_ff_daily(conn, start: str, end: str) -> tuple[pd.DataFrame, str]:
    try:
        factors = conn.raw_sql(
            "select date, mktrf, rf from ff_all.factors_daily "
            f"where date between '{start}' and '{end}'",
            date_cols=["date"],
        )
        if not factors.empty:
            return factors, "ff_all.factors_daily"
    except Exception:
        pass
    market = conn.raw_sql(
        "select date, vwretd from crsp.dsi " f"where date between '{start}' and '{end}'",
        date_cols=["date"],
    )
    return market, "crsp.dsi"


def query_crsp_daily_chunk(conn, start: str, end: str, permnos: list[int]) -> pd.DataFrame:
    if not permnos:
        return pd.DataFrame(columns=["permno", "date", "ret"])
    permno_sql = ", ".join(str(int(item)) for item in permnos)
    sql = f"""
        select permno, date, ret
        from crsp.dsf
        where date between '{start}' and '{end}'
          and permno in ({permno_sql})
          and ret is not null
    """
    return conn.raw_sql(sql, date_cols=["date"])


def build_year_daily_risk(
    conn,
    panel_year: pd.DataFrame,
    year: int,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    query_start = f"{year - 1}-01-01"
    query_end = f"{year}-12-31"
    factors, factor_source = query_ff_daily(conn, query_start, query_end)
    permnos = sorted(int(item) for item in panel_year["permno"].dropna().unique())
    daily_parts: list[pd.DataFrame] = []
    for permno_chunk in chunked(permnos, chunk_size):
        daily_chunk = query_crsp_daily_chunk(conn, query_start, query_end, permno_chunk)
        if not daily_chunk.empty:
            daily_parts.append(daily_chunk)
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame(columns=["permno", "date", "ret"])
    risk_daily = compute_daily_risk_characteristics(daily, factors)
    risk_monthly = collapse_daily_risk_to_panel_months(risk_daily, panel_year[["permno", "date"]])
    risk_monthly = risk_monthly[pd.to_datetime(risk_monthly["date"]).dt.year.eq(year)].copy()
    stats = {
        "year": year,
        "factor_source": factor_source,
        "target_permnos": len(permnos),
        "daily_rows": int(len(daily)),
        "factor_rows": int(len(factors)),
        "monthly_rows": int(len(risk_monthly)),
        "beta_nonmissing": int(risk_monthly["beta_252d"].notna().sum()) if "beta_252d" in risk_monthly else 0,
        "idio_nonmissing": int(risk_monthly["idio_vol_252d"].notna().sum()) if "idio_vol_252d" in risk_monthly else 0,
    }
    return risk_monthly, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--sample-permnos", type=int, default=0, help="Optional smallest smoke universe size.")
    parser.add_argument("--output-suffix", default="", help="Safe suffix for smoke outputs, e.g. _smoke.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--panel",
        default="data/processed/panel/surface_characteristic_state_panel.parquet",
        help="Panel to augment with daily risk characteristics, relative to project root.",
    )
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    panel_path = root / args.panel
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing panel for daily risk merge: {panel_path}")

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    target = panel[
        panel["date"].dt.year.between(args.start_year, args.end_year)
        & panel["permno"].notna()
    ].copy()
    if args.sample_permnos:
        chosen = (
            target.groupby("permno")["date"]
            .count()
            .sort_values(ascending=False)
            .head(args.sample_permnos)
            .index.astype(int)
            .tolist()
        )
        target = target[target["permno"].isin(chosen)].copy()

    suffix = safe_suffix(args.output_suffix) or (f"_sample{args.sample_permnos}" if args.sample_permnos else "")
    shard_dir = root / "data" / "processed" / "panel" / f"daily_risk{suffix}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = dirs["manifests"] / f"daily_risk{suffix}"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    conn = connect_wrds()
    year_stats: list[dict[str, object]] = []
    shards: list[Path] = []
    try:
        for year in range(args.start_year, args.end_year + 1):
            panel_year = target[target["date"].dt.year.eq(year)].copy()
            if panel_year.empty:
                stats = {"year": year, "status": "SKIP_EMPTY_PANEL"}
                year_stats.append(stats)
                write_json_atomic(manifest_dir / f"{year}.json", stats)
                continue
            shard_path = shard_dir / f"daily_risk_{year}.parquet"
            if args.resume and shard_path.exists():
                risk_year = pd.read_parquet(shard_path)
                stats = {
                    "year": year,
                    "status": "PASS_RESUMED",
                    "monthly_rows": int(len(risk_year)),
                    "beta_nonmissing": int(risk_year["beta_252d"].notna().sum()) if "beta_252d" in risk_year else 0,
                    "idio_nonmissing": int(risk_year["idio_vol_252d"].notna().sum()) if "idio_vol_252d" in risk_year else 0,
                }
            else:
                risk_year, stats = build_year_daily_risk(conn, panel_year, year, args.chunk_size)
                stats["status"] = "PASS"
                tmp_path = shard_path.with_suffix(".parquet.tmp")
                risk_year.to_parquet(tmp_path, index=False)
                tmp_path.replace(shard_path)
            year_stats.append(stats)
            shards.append(shard_path)
            write_json_atomic(manifest_dir / f"{year}.json", stats)
            print(
                f"daily_risk_year={year} status={stats['status']} "
                f"rows={stats.get('monthly_rows', 0)} beta_nonmissing={stats.get('beta_nonmissing', 0)}"
            )
    except Exception as exc:
        manifest = {
            "status": "BLOCKED_DAILY_RISK",
            "error": type(exc).__name__,
            "error_message": str(exc)[:1000],
            "completed_years": year_stats,
        }
        write_json_atomic(dirs["manifests"] / "daily_risk_manifest.json", manifest)
        print("daily_risk_status=BLOCKED_DAILY_RISK")
        return 31
    finally:
        conn.close()

    risk_all = pd.concat((pd.read_parquet(path) for path in shards), ignore_index=True) if shards else pd.DataFrame()
    risk_path = root / "data" / "processed" / "panel" / f"daily_risk_characteristics{suffix}.parquet"
    risk_all.to_parquet(risk_path, index=False)
    merged = panel.merge(risk_all, on=["permno", "date"], how="left")
    output_path = root / "data" / "processed" / "panel" / f"surface_characteristic_state_daily_risk_panel{suffix}.parquet"
    merged.to_parquet(output_path, index=False)

    risk_cols = [
        col
        for col in ["beta_252d", "idio_vol_252d", "total_vol_252d", "mkt_corr_252d", "daily_obs_252d"]
        if col in merged
    ]
    manifest = {
        "status": "PASS",
        "sample_permnos": args.sample_permnos,
        "output_suffix": suffix,
        "years": year_stats,
        "risk_rows": int(len(risk_all)),
        "panel_rows": int(len(merged)),
        "date_min": str(merged["date"].min().date()) if not merged.empty else None,
        "date_max": str(merged["date"].max().date()) if not merged.empty else None,
        "risk_columns": risk_cols,
        "missingness": {col: float(merged[col].isna().mean()) for col in risk_cols},
        "artifacts": {
            "risk_table": str(risk_path.relative_to(root)),
            "panel": str(output_path.relative_to(root)),
        },
    }
    write_json_atomic(dirs["manifests"] / f"daily_risk_manifest{suffix}.json", manifest)
    print("daily_risk_status=PASS")
    print(f"panel_rows={manifest['panel_rows']} risk_columns={len(risk_cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
