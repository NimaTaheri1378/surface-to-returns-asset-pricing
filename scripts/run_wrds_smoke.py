from __future__ import annotations

import argparse
import calendar
import traceback
from pathlib import Path

import pandas as pd

from surface_returns.config import load_yaml
from surface_returns.figures import write_smoke_surface_figure
from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs
from surface_returns.surfaces import fixed_surface_grid, surface_features
from surface_returns.wrds_helpers import (
    TableRef,
    candidate_tables,
    choose_table,
    connect_wrds,
    describe_columns,
    list_tables_safe,
    select_existing,
)


def month_bounds(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"


def write_parquet(frame: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return int(len(frame))


def safe_raw_sql(conn, sql: str) -> pd.DataFrame:
    return conn.raw_sql(sql, date_cols=["date", "datadate", "exdate", "linkdt", "linkenddt"])


def detect_inventory(conn, libraries: list[str]) -> dict[str, list[str]]:
    visible = set(conn.list_libraries())
    inventory: dict[str, list[str]] = {}
    for lib in libraries:
        inventory[lib] = list_tables_safe(conn, lib) if lib in visible else []
    return inventory


def option_price_table(inventory: dict[str, list[str]], smoke_year: int) -> TableRef | None:
    tables = inventory.get("optionm", [])
    table = choose_table(
        tables,
        preferred=[f"opprcd{smoke_year}", "opprcd"],
        contains=["opprcd"],
    )
    return TableRef("optionm", table) if table else None


def direct_option_crsp_link(conn, inventory: dict[str, list[str]]) -> tuple[TableRef | None, list[str]]:
    candidates = [
        ("wrdsapps", ["opcrsphist", "optionm_crsp_link", "optm_crsp_link", "optionm_crsp", "opcrsp_link"]),
        ("optionm", ["opcrsphist", "optionm_crsp_link", "crsp_link", "opcrsp_link"]),
    ]
    for lib, preferred in candidates:
        table_names = candidate_tables(inventory.get(lib, []), preferred=preferred, contains=["crsp"])
        if lib == "wrdsapps":
            table_names.extend(
                table
                for table in candidate_tables(inventory.get(lib, []), preferred=["opcrsphist"], contains=["opcrsp"])
                if table not in table_names
            )
        for table in table_names:
            ref = TableRef(lib, table)
            columns = describe_columns(conn, ref)
            if {"secid", "permno"}.issubset(set(columns)):
                return ref, columns
    return None, []


def run_smoke(root: Path, config_path: Path) -> int:
    root = assert_approved_root(root)
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    cfg = load_yaml(config_path)
    sample = cfg["sample"]
    year = int(sample["smoke_year"])
    month = int(sample["smoke_month"])
    start_date, end_date = month_bounds(year, month)
    manifest: dict[str, object] = {
        "status": "STARTED",
        "smoke_year": year,
        "smoke_month": month,
        "tables": {},
        "row_counts": {},
        "artifacts": {},
        "warnings": [],
    }

    try:
        conn = connect_wrds()
    except Exception as exc:
        manifest["status"] = "BLOCKED_WRDS_AUTH"
        manifest["error"] = type(exc).__name__
        manifest["error_message"] = str(exc)[:500]
        write_json_atomic(dirs["manifests"] / "wrds_smoke_manifest.json", manifest)
        print("wrds_smoke_status=BLOCKED_WRDS_AUTH")
        return 23

    try:
        core_libs = list(cfg["wrds"]["core_libraries"])
        link_libs = list(cfg["wrds"].get("optional_link_libraries", []))
        inventory = detect_inventory(conn, sorted(set(core_libs + link_libs)))
        manifest["visible_library_count"] = len(conn.list_libraries())
        manifest["inventory"] = {lib: tables[:50] for lib, tables in inventory.items()}

        option_ref = option_price_table(inventory, year)
        crsp_msf = TableRef("crsp", choose_table(inventory.get("crsp", []), ["msf"]) or "")
        crsp_names = TableRef(
            "crsp",
            choose_table(inventory.get("crsp", []), ["msenames", "stocknames"], contains=["name"]) or "",
        )
        comp_funda = TableRef("comp", choose_table(inventory.get("comp", []), ["funda"]) or "")
        ccm_link = TableRef(
            "crsp",
            choose_table(inventory.get("crsp", []), ["ccmxpf_lnkhist"], contains=["lnkhist"]) or "",
        )

        required_refs = {
            "option_prices": option_ref,
            "crsp_msf": crsp_msf if crsp_msf.table else None,
            "crsp_names": crsp_names if crsp_names.table else None,
            "comp_funda": comp_funda if comp_funda.table else None,
            "ccm_link": ccm_link if ccm_link.table else None,
        }
        manifest["tables"] = {
            key: ref.qualified if ref else None for key, ref in required_refs.items()
        }
        missing = [key for key, ref in required_refs.items() if ref is None]
        if missing:
            manifest["status"] = "BLOCKED_SCHEMA_MISSING"
            manifest["missing_required_tables"] = missing
            write_json_atomic(dirs["manifests"] / "wrds_smoke_manifest.json", manifest)
            print("wrds_smoke_status=BLOCKED_SCHEMA_MISSING")
            return 24

        option_cols = describe_columns(conn, option_ref)
        selected_option_cols = select_existing(
            option_cols,
            [
                "secid",
                "date",
                "exdate",
                "cp_flag",
                "strike_price",
                "best_bid",
                "best_offer",
                "impl_volatility",
                "delta",
                "volume",
                "open_interest",
            ],
        )
        if not {"secid", "date", "exdate", "impl_volatility"}.issubset(set(selected_option_cols)):
            manifest["status"] = "BLOCKED_OPTION_COLUMNS"
            manifest["option_columns"] = option_cols
            write_json_atomic(dirs["manifests"] / "wrds_smoke_manifest.json", manifest)
            print("wrds_smoke_status=BLOCKED_OPTION_COLUMNS")
            return 25

        option_date_sql = (
            f"select max(date) as date from {option_ref.qualified} "
            f"where date between '{start_date}' and '{end_date}'"
        )
        option_date_df = safe_raw_sql(conn, option_date_sql)
        option_date = pd.Timestamp(option_date_df["date"].iloc[0])
        if pd.isna(option_date):
            manifest["status"] = "BLOCKED_NO_OPTION_DATE"
            write_json_atomic(dirs["manifests"] / "wrds_smoke_manifest.json", manifest)
            print("wrds_smoke_status=BLOCKED_NO_OPTION_DATE")
            return 26
        max_dte = option_date + pd.Timedelta(days=365)
        min_dte = option_date + pd.Timedelta(days=10)
        option_sql = (
            f"select {', '.join(selected_option_cols)} from {option_ref.qualified} "
            f"where date = '{option_date.date()}' "
            f"and exdate between '{min_dte.date()}' and '{max_dte.date()}' "
            f"and impl_volatility is not null "
        )
        if {"best_bid", "best_offer"}.issubset(set(selected_option_cols)):
            option_sql += "and best_bid > 0 and best_offer > best_bid "
        option_sql += f"limit {int(sample['max_option_rows'])}"
        options = safe_raw_sql(conn, option_sql)
        manifest["row_counts"]["option_quotes"] = write_parquet(
            options, dirs["raw_smoke"] / "option_quotes.parquet"
        )

        features = surface_features(options)
        grid = fixed_surface_grid(options, maturities=[30, 60, 90, 180, 270], deltas=[0.10, 0.25, 0.50, 0.75, 0.90])
        manifest["row_counts"]["surface_features"] = write_parquet(
            features, dirs["processed_smoke"] / "surface_features.parquet"
        )
        manifest["row_counts"]["surface_grid"] = write_parquet(
            grid, dirs["processed_smoke"] / "surface_grid.parquet"
        )
        fig_paths = write_smoke_surface_figure(options, dirs["figures_smoke"] / "option_surface_slice")
        manifest["artifacts"]["smoke_figures"] = [str(path.relative_to(root)) for path in fig_paths]

        crsp_sql = (
            "select m.permno, m.date, m.ret, m.prc, m.shrout, "
            "n.shrcd, n.exchcd, n.siccd, n.ticker, n.comnam "
            f"from {crsp_msf.qualified} as m "
            f"left join {crsp_names.qualified} as n "
            "on m.permno = n.permno and n.namedt <= m.date "
            "and m.date <= coalesce(n.nameendt, '2099-12-31') "
            f"where m.date between '{start_date}' and '{end_date}' "
            "and n.shrcd in (10, 11) and n.exchcd in (1, 2, 3) "
            f"limit {int(sample['max_crsp_rows'])}"
        )
        crsp = safe_raw_sql(conn, crsp_sql)
        manifest["row_counts"]["crsp_monthly"] = write_parquet(crsp, dirs["raw_smoke"] / "crsp_monthly.parquet")

        comp_sql = (
            "select gvkey, datadate, fyear, at, ceq, seq, txditc, pstk, sale, ni, capx "
            f"from {comp_funda.qualified} "
            "where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C' "
            f"and datadate between '{year - 2}-01-01' and '{year}-12-31' "
            f"limit {int(sample['max_comp_rows'])}"
        )
        comp = safe_raw_sql(conn, comp_sql)
        manifest["row_counts"]["comp_funda"] = write_parquet(comp, dirs["raw_smoke"] / "comp_funda.parquet")

        ccm_sql = (
            "select gvkey, lpermno as permno, linkdt, linkenddt, linktype, linkprim "
            f"from {ccm_link.qualified} "
            "where lpermno is not null and linktype in ('LU', 'LC') "
            f"and linkdt <= '{end_date}' and coalesce(linkenddt, '2099-12-31') >= '{year - 2}-01-01' "
            "limit 10000"
        )
        ccm = safe_raw_sql(conn, ccm_sql)
        manifest["row_counts"]["ccm_link"] = write_parquet(ccm, dirs["raw_smoke"] / "ccm_link.parquet")

        link_ref, link_cols = direct_option_crsp_link(conn, inventory)
        manifest["tables"]["option_crsp_link"] = link_ref.qualified if link_ref else None
        if link_ref:
            cols = select_existing(link_cols, ["secid", "permno", "sdate", "edate", "score", "ticker", "cusip"])
            link_sql = f"select {', '.join(cols)} from {link_ref.qualified} limit 50000"
            link = safe_raw_sql(conn, link_sql)
            manifest["row_counts"]["option_crsp_link"] = write_parquet(
                link, dirs["raw_smoke"] / "option_crsp_link.parquet"
            )
            link_for_merge = link[["secid", "permno"]].dropna().drop_duplicates()
            merged = features.merge(link_for_merge, on="secid", how="inner")
            merged = merged.merge(crsp[["permno", "date", "ret"]], on=["permno", "date"], how="inner")
            manifest["row_counts"]["linked_feature_rows"] = write_parquet(
                merged, dirs["processed_smoke"] / "linked_features.parquet"
            )
        else:
            manifest["warnings"].append("No direct OptionMetrics-CRSP link table with secid and permno detected.")
            manifest["row_counts"]["option_crsp_link"] = 0
            manifest["row_counts"]["linked_feature_rows"] = 0

        pass_conditions = [
            manifest["row_counts"].get("option_quotes", 0) > 0,
            manifest["row_counts"].get("surface_features", 0) > 0,
            manifest["row_counts"].get("crsp_monthly", 0) > 0,
            len(fig_paths) > 0,
            manifest["row_counts"].get("linked_feature_rows", 0) > 0,
        ]
        manifest["status"] = "PASS" if all(pass_conditions) else "SMOKE_PARTIAL_BLOCKED"
        manifest["scale_gate"] = {
            "approved_for_full_scale": manifest["status"] == "PASS",
            "reason": "all smoke checks passed" if manifest["status"] == "PASS" else "missing required linked rows or artifacts",
        }
        write_json_atomic(dirs["manifests"] / "wrds_smoke_manifest.json", manifest)
        print(f"wrds_smoke_status={manifest['status']}")
        print(f"manifest={dirs['manifests'] / 'wrds_smoke_manifest.json'}")
        return 0 if manifest["status"] == "PASS" else 3
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = type(exc).__name__
        manifest["error_message"] = str(exc)[:1000]
        manifest["traceback_tail"] = traceback.format_exc().splitlines()[-20:]
        write_json_atomic(dirs["manifests"] / "wrds_smoke_manifest.json", manifest)
        print("wrds_smoke_status=FAILED")
        print(f"error={type(exc).__name__}")
        return 1
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yml")
    args = parser.parse_args()
    root = assert_approved_root(Path.cwd())
    return run_smoke(root, root / args.config)


if __name__ == "__main__":
    raise SystemExit(main())
