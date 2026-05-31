from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs
from surface_returns.state_controls import build_state_control_table, merge_state_controls
from surface_returns.wrds_helpers import connect_wrds


def query_tables(conn, start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = f"{start_year - 1}-01-01"
    end = f"{end_year}-12-31"
    ff = conn.raw_sql(
        f"select date, mktrf, smb, hml, rf, umd from ff_all.factors_monthly "
        f"where date between '{start}' and '{end}'",
        date_cols=["date"],
    )
    ff5 = conn.raw_sql(
        f"select date, rmw, cma from ff_all.fivefactors_monthly where date between '{start}' and '{end}'",
        date_cols=["date"],
    )
    frb = conn.raw_sql(
        "select date, aaa, baa, fedfunds, mswp1, mswp10, d_tcmnom_y20, ltiit "
        "from frb_all.rates_monthly "
        f"where date between '{start}' and '{end}'",
        date_cols=["date"],
    )
    cboe = conn.raw_sql(
        f"select date, vix, vxo, vxn, vxd from cboe_all.cboe where date between '{start}' and '{end}'",
        date_cols=["date"],
    )
    return ff, ff5, frb, cboe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--panel",
        default="data/processed/panel/surface_characteristic_panel.parquet",
        help="Panel to augment with state controls, relative to project root.",
    )
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    panel_path = root / args.panel
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing panel for state-control merge: {panel_path}")

    conn = connect_wrds()
    try:
        ff, ff5, frb, cboe = query_tables(conn, args.start_year, args.end_year)
    finally:
        conn.close()
    state = build_state_control_table(ff, ff5, frb, cboe)
    state_path = root / "data" / "processed" / "states" / "monthly_state_controls.parquet"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state.to_parquet(state_path, index=False)

    panel = pd.read_parquet(panel_path)
    merged = merge_state_controls(panel, state)
    output_path = root / "data" / "processed" / "panel" / "surface_characteristic_state_panel.parquet"
    merged.to_parquet(output_path, index=False)

    state_cols = [col for col in state.columns if col != "month"]
    manifest = {
        "status": "PASS",
        "ff_rows": int(len(ff)),
        "ff5_rows": int(len(ff5)),
        "frb_rows": int(len(frb)),
        "cboe_rows": int(len(cboe)),
        "state_months": int(len(state)),
        "panel_rows": int(len(merged)),
        "state_columns": state_cols,
        "state_missingness": {col: float(merged[col].isna().mean()) for col in state_cols if col in merged},
        "artifacts": {
            "state_table": str(state_path.relative_to(root)),
            "panel": str(output_path.relative_to(root)),
        },
    }
    write_json_atomic(dirs["manifests"] / "state_controls_manifest.json", manifest)
    print("state_controls_status=PASS")
    print(f"panel_rows={manifest['panel_rows']} state_columns={len(state_cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
