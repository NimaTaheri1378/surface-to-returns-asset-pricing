from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from surface_returns.external_api import build_external_controls, load_env_file
from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs
from surface_returns.state_controls import merge_state_controls


def candidate_panel_paths(root: Path) -> list[Path]:
    return [
        root / "data" / "processed" / "panel" / "surface_characteristic_state_ibes_regsho_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_taq_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_features_panel.parquet",
    ]


def lag_for_availability(controls: pd.DataFrame, lag_months: int, start_year: int, end_year: int) -> pd.DataFrame:
    out = controls.copy()
    out["month"] = pd.to_datetime(out["month"]).dt.to_period("M").dt.to_timestamp()
    if lag_months:
        out["month"] = out["month"] + pd.DateOffset(months=lag_months)
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-01")
    out = out[out["month"].between(start, end)].copy()
    return out.sort_values("month").drop_duplicates("month", keep="last").reset_index(drop=True)


def merge_state_tables(base_state: pd.DataFrame, external_state: pd.DataFrame) -> pd.DataFrame:
    external = external_state.copy()
    external["month"] = pd.to_datetime(external["month"]).dt.to_period("M").dt.to_timestamp()
    if base_state.empty:
        return external
    base = base_state.copy()
    base["month"] = pd.to_datetime(base["month"]).dt.to_period("M").dt.to_timestamp()
    overlaps = sorted(set(base.columns).intersection(external.columns).difference({"month"}))
    if overlaps:
        external = external.rename(columns={col: f"external_{col}" for col in overlaps})
    return base.merge(external, on="month", how="outer").sort_values("month").reset_index(drop=True)


def panel_output_path(panel_path: Path) -> Path:
    stem = panel_path.stem
    if stem.endswith("_panel"):
        stem = f"{stem[:-6]}_external_panel"
    else:
        stem = f"{stem}_external"
    return panel_path.with_name(f"{stem}.parquet")


def coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in frame.columns:
        if col == "month":
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        rows.append(
            {
                "column": col,
                "nonmissing": int(values.notna().sum()),
                "missing_share": float(values.isna().mean()) if len(values) else 1.0,
                "min": float(values.min()) if values.notna().any() else None,
                "max": float(values.max()) if values.notna().any() else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_share", "column"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--availability-lag-months", type=int, default=1)
    parser.add_argument(
        "--panel",
        default=None,
        help="Panel to augment with external controls, relative to project root. Defaults to the richest available panel.",
    )
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    load_env_file(root / ".env.local")

    raw_controls, statuses = build_external_controls(args.start_year, args.end_year)
    raw_controls["month"] = pd.to_datetime(raw_controls["month"]).dt.to_period("M").dt.to_timestamp()
    raw_controls = raw_controls.sort_values("month").reset_index(drop=True)
    signal_controls = lag_for_availability(
        raw_controls,
        lag_months=args.availability_lag_months,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    external_dir = root / "data" / "external"
    state_dir = root / "data" / "processed" / "states"
    panel_dir = root / "data" / "processed" / "panel"
    report_dir = root / "outputs" / "reports" / "external"
    for path in [external_dir, state_dir, panel_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)

    raw_path = external_dir / "monthly_external_api_controls_raw.parquet"
    signal_path = external_dir / "monthly_external_api_controls.parquet"
    coverage_path = report_dir / "monthly_external_api_coverage.csv"
    raw_controls.to_parquet(raw_path, index=False)
    signal_controls.to_parquet(signal_path, index=False)
    coverage = coverage_table(signal_controls)
    coverage.to_csv(coverage_path, index=False)

    base_state_path = state_dir / "monthly_state_controls.parquet"
    base_state = pd.read_parquet(base_state_path) if base_state_path.exists() else pd.DataFrame()
    merged_state = merge_state_tables(base_state, signal_controls)
    state_path = state_dir / "monthly_state_controls_external.parquet"
    merged_state.to_parquet(state_path, index=False)

    candidate_panels = candidate_panel_paths(root)
    panel_path = root / args.panel if args.panel else next((path for path in candidate_panels if path.exists()), candidate_panels[-1])
    panel_artifact = None
    panel_rows = 0
    if panel_path.exists() and not signal_controls.empty:
        panel = pd.read_parquet(panel_path)
        merged_panel = merge_state_controls(panel, signal_controls)
        panel_artifact_path = panel_output_path(panel_path)
        merged_panel.to_parquet(panel_artifact_path, index=False)
        panel_artifact = str(panel_artifact_path.relative_to(root))
        panel_rows = int(len(merged_panel))

    control_passes = [
        status["source"]
        for status in statuses
        if status.get("status") == "PASS" and status.get("source") in {"FRED", "BLS", "BEA", "EIA"}
    ]
    manifest = {
        "status": "PASS" if control_passes and not signal_controls.empty else "FAILED",
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
        "availability_lag_months": int(args.availability_lag_months),
        "source_statuses": statuses,
        "control_sources_passed": control_passes,
        "raw_rows": int(len(raw_controls)),
        "signal_rows": int(len(signal_controls)),
        "signal_columns": [col for col in signal_controls.columns if col != "month"],
        "state_rows": int(len(merged_state)),
        "panel": str(panel_path.relative_to(root)) if panel_path.exists() else None,
        "panel_rows": panel_rows,
        "artifacts": {
            "raw_controls": str(raw_path.relative_to(root)),
            "signal_controls": str(signal_path.relative_to(root)),
            "state_controls_external": str(state_path.relative_to(root)),
            "coverage_csv": str(coverage_path.relative_to(root)),
            "panel_external": panel_artifact,
        },
    }
    write_json_atomic(dirs["manifests"] / "external_api_controls_manifest.json", manifest)
    print(f"external_api_controls_status={manifest['status']}")
    print(f"sources={','.join(control_passes)} signal_rows={manifest['signal_rows']} panel_rows={panel_rows}")
    return 0 if manifest["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
