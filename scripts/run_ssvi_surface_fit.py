from __future__ import annotations

import argparse
import calendar
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from surface_returns.manifest import utc_now_iso, write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs
from surface_returns.svi import fit_ssvi_surface


def month_iter(start_year: int, end_year: int, months: list[int]) -> list[tuple[int, int]]:
    out = []
    for year in range(start_year, end_year + 1):
        use_months = months or list(range(1, 13))
        for month in use_months:
            if 1 <= month <= 12:
                out.append((year, month))
    return out


def input_grid_path(root: Path, year: int, month: int) -> Path:
    return root / "data" / "processed" / "full" / "svi_surface_grid" / f"year={year:04d}" / f"month={month:02d}.parquet"


def output_paths(root: Path, year: int, month: int) -> tuple[Path, Path]:
    grid_path = root / "data" / "processed" / "full" / "ssvi_surface_grid" / f"year={year:04d}" / f"month={month:02d}.parquet"
    diag_path = root / "data" / "processed" / "full" / "ssvi_surface_diagnostics" / f"year={year:04d}" / f"month={month:02d}.parquet"
    return grid_path, diag_path


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return int(len(frame))


def fit_one_surface(task: tuple[object, object, object, pd.DataFrame, int]) -> tuple[pd.DataFrame, dict[str, object]]:
    date, secid, permno, group, min_points = task
    _fit, fitted_grid, diagnostics = fit_ssvi_surface(group, min_points=min_points)
    base = {"date": date, "secid": secid, "permno": permno}
    diag = {**base, **diagnostics}
    if not fitted_grid.empty:
        fitted_grid = fitted_grid.copy()
        fitted_grid["date"] = date
        fitted_grid["secid"] = secid
        fitted_grid["permno"] = permno
    return fitted_grid, diag


def run_month(root: Path, year: int, month: int, args: argparse.Namespace) -> dict[str, object]:
    source = input_grid_path(root, year, month)
    manifest: dict[str, object] = {
        "status": "STARTED",
        "created_utc": utc_now_iso(),
        "year": year,
        "month": month,
        "source": str(source.relative_to(root)),
        "workers": args.workers,
        "min_points": args.min_points,
    }
    if not source.exists():
        manifest["status"] = "SKIPPED_MISSING_SVI_GRID"
        return manifest
    frame = pd.read_parquet(source)
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    if args.max_securities > 0:
        keep = (
            frame.groupby(["date", "secid"], dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(args.max_securities)
            .index
        )
        key_frame = pd.DataFrame(list(keep), columns=["date", "secid"])
        frame = frame.merge(key_frame, on=["date", "secid"], how="inner")
    tasks = []
    for (date, secid), group in frame.groupby(["date", "secid"], dropna=False):
        permno = group["permno"].dropna().iloc[0] if "permno" in group and group["permno"].notna().any() else None
        tasks.append((date, secid, permno, group.copy(), args.min_points))
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            outputs = list(executor.map(fit_one_surface, tasks))
    else:
        outputs = [fit_one_surface(task) for task in tasks]
    grids = [item for item, _diag in outputs if not item.empty]
    diagnostics = pd.DataFrame([diag for _grid, diag in outputs])
    fitted = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()
    grid_path, diag_path = output_paths(root, year, month)
    status_counts = diagnostics["status"].value_counts(dropna=False).to_dict() if "status" in diagnostics else {}
    pass_count = int(status_counts.get("PASS", 0))
    manifest["row_counts"] = {
        "input_grid": int(len(frame)),
        "surfaces": int(len(tasks)),
        "ssvi_grid": write_parquet_atomic(fitted, grid_path),
        "ssvi_diagnostics": write_parquet_atomic(diagnostics, diag_path),
    }
    manifest["diagnostic_status_counts"] = {str(key): int(value) for key, value in status_counts.items()}
    manifest["pass_share"] = float(pass_count / len(tasks)) if tasks else 0.0
    manifest["mean_rmse_total_variance"] = (
        float(pd.to_numeric(diagnostics.get("rmse_total_variance"), errors="coerce").mean())
        if not diagnostics.empty and "rmse_total_variance" in diagnostics
        else None
    )
    manifest["no_arbitrage_pass_counts"] = {
        key: int(pd.Series(diagnostics.get(key, pd.Series(dtype=bool))).fillna(False).astype(bool).sum())
        for key in [
            "positive_total_variance",
            "theta_monotone",
            "theta_phi_monotone",
            "calendar_monotone_grid",
            "ssvi_bounds_pass",
        ]
    }
    manifest["artifacts"] = {
        "ssvi_grid": str(grid_path.relative_to(root)),
        "ssvi_diagnostics": str(diag_path.relative_to(root)),
    }
    manifest["status"] = "PASS" if tasks and pass_count == len(tasks) else ("PARTIAL" if pass_count else "FAILED")
    return manifest


def summarize(manifests: list[dict[str, object]]) -> dict[str, object]:
    statuses = [str(item.get("status")) for item in manifests]
    surfaces = sum(int(item.get("row_counts", {}).get("surfaces", 0)) for item in manifests)
    pass_surfaces = sum(int(item.get("diagnostic_status_counts", {}).get("PASS", 0)) for item in manifests)
    return {
        "status": "PASS" if statuses and all(status == "PASS" for status in statuses) else "PARTIAL",
        "created_utc": utc_now_iso(),
        "months": len(manifests),
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "surfaces": int(surfaces),
        "pass_surfaces": int(pass_surfaces),
        "pass_share": float(pass_surfaces / surfaces) if surfaces else 0.0,
        "calendar": "SSVI power-law phi with fitted monotone theta and grid calendar checks",
        "butterfly": "Gatheral-Jacquier SSVI bounds theta*phi*(1+|rho|)<4 and theta*phi^2*(1+|rho|)<4",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--months", type=int, nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-points", type=int, default=12)
    parser.add_argument("--max-securities", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    ensure_project_dirs(root)
    manifest_dir = root / "manifests" / "ssvi_fit"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifests = []
    for year, month in month_iter(args.start_year, args.end_year, args.months):
        calendar.monthrange(year, month)
        manifest_path = manifest_dir / f"{year:04d}_{month:02d}.json"
        if args.resume and manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifests.append(existing)
            continue
        manifest = run_month(root, year, month, args)
        write_json_atomic(manifest_path, manifest)
        manifests.append(manifest)
        print(
            f"ssvi_fit {year}-{month:02d} status={manifest['status']} "
            f"pass_share={manifest.get('pass_share', 0.0):.3f}",
            flush=True,
        )

    summary = summarize(manifests)
    write_json_atomic(manifest_dir / "summary.json", summary)
    print("ssvi_fit_status=" + summary["status"])
    print(f"surfaces={summary['surfaces']} pass_share={summary['pass_share']:.4f}")
    return 0 if summary["status"] in {"PASS", "PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
