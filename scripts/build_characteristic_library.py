from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from surface_returns.characteristics import (
    compute_compustat_annual_characteristics,
    compute_crsp_monthly_characteristics,
    merge_compustat_characteristics,
)
from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs
from surface_returns.wrds_helpers import connect_wrds


def parquet_files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern))


def read_many(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def query_compustat(conn, start_year: int, end_year: int) -> pd.DataFrame:
    sql = f"""
        select gvkey, datadate, fyear, at, ceq, seq, txditc, pstk, pstkrv, pstkl,
               sale, revt, cogs, xsga, xint, ni, ib, oancf, capx, act, che, lct,
               dlc, dltt, txp, dp, csho, prcc_f
        from comp.funda
        where indfmt='INDL'
          and datafmt='STD'
          and popsrc='D'
          and consol='C'
          and datadate between '{start_year - 2}-01-01' and '{end_year}-12-31'
    """
    return conn.raw_sql(sql, date_cols=["datadate"])


def query_ccm(conn, start_year: int, end_year: int) -> pd.DataFrame:
    sql = f"""
        select gvkey, lpermno as permno, linkdt, linkenddt, linktype, linkprim
        from crsp.ccmxpf_lnkhist
        where lpermno is not null
          and linktype in ('LU', 'LC', 'LS')
          and linkdt <= '{end_year}-12-31'
          and coalesce(linkenddt, '2099-12-31') >= '{start_year - 2}-01-01'
    """
    return conn.raw_sql(sql, date_cols=["linkdt", "linkenddt"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--skip-wrds", action="store_true", help="Build only CRSP-monthly characteristics from cached shards.")
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    panel_path = root / "data" / "processed" / "panel" / "surface_features_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing surface panel: {panel_path}")
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])

    crsp_paths = parquet_files(root, "data/processed/full/crsp_monthly/year=*/month=*.parquet")
    crsp = read_many(crsp_paths)
    crsp_chars = compute_crsp_monthly_characteristics(crsp)
    out = panel.merge(crsp_chars, on=["permno", "date"], how="left", suffixes=("", "_crsp_char"))

    comp_rows = 0
    ccm_rows = 0
    if not args.skip_wrds:
        try:
            conn = connect_wrds()
            try:
                funda = query_compustat(conn, args.start_year, args.end_year)
                ccm = query_ccm(conn, args.start_year, args.end_year)
            finally:
                conn.close()
            comp_chars = compute_compustat_annual_characteristics(funda)
            comp_rows = int(len(comp_chars))
            ccm_rows = int(len(ccm))
            out = merge_compustat_characteristics(out, comp_chars, ccm)
        except Exception as exc:
            manifest = {
                "status": "BLOCKED_WRDS_CHARACTERISTICS",
                "error": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "crsp_characteristic_rows": int(len(crsp_chars)),
            }
            write_json_atomic(dirs["manifests"] / "characteristic_library_manifest.json", manifest)
            print("characteristic_library_status=BLOCKED_WRDS_CHARACTERISTICS")
            return 23

    char_cols = [
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
        "book_equity",
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
        "book_to_market_comp",
    ]
    available = [col for col in char_cols if col in out]
    output_path = root / "data" / "processed" / "panel" / "surface_characteristic_panel.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    missingness = {col: float(out[col].isna().mean()) for col in available}
    manifest = {
        "status": "PASS",
        "panel_rows": int(len(out)),
        "unique_permnos": int(out["permno"].nunique()) if "permno" in out else 0,
        "date_min": str(out["date"].min().date()) if not out.empty else None,
        "date_max": str(out["date"].max().date()) if not out.empty else None,
        "crsp_monthly_shards": len(crsp_paths),
        "crsp_characteristic_rows": int(len(crsp_chars)),
        "compustat_characteristic_rows": comp_rows,
        "ccm_link_rows": ccm_rows,
        "characteristics": available,
        "missingness": missingness,
        "artifacts": {"panel": str(output_path.relative_to(root))},
    }
    write_json_atomic(dirs["manifests"] / "characteristic_library_manifest.json", manifest)
    print("characteristic_library_status=PASS")
    print(f"panel_rows={manifest['panel_rows']} characteristics={len(available)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
