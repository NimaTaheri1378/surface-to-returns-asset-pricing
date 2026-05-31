from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs


def parquet_files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern))


def read_many(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def add_next_month_returns(features: pd.DataFrame, crsp: pd.DataFrame) -> pd.DataFrame:
    crsp = crsp[["permno", "date", "ret"]].copy()
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp = crsp.sort_values(["permno", "date"])
    crsp["next_ret"] = crsp.groupby("permno")["ret"].shift(-1)
    crsp_next = crsp[["permno", "date", "next_ret"]]
    panel = features.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.merge(crsp_next, on=["permno", "date"], how="left")


def write_coverage_figure(panel: pd.DataFrame, output_path: Path) -> list[Path]:
    if panel.empty:
        return []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly = (
        panel.assign(month=lambda frame: pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .agg(
            securities=("permno", "nunique"),
            median_contracts=("n_contracts", "median"),
            mean_iv=("mean_iv", "mean"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(monthly["month"], monthly["securities"], color="#1f77b4", linewidth=1.6)
    axes[0].set_ylabel("Securities")
    axes[1].plot(monthly["month"], monthly["median_contracts"], color="#2ca02c", linewidth=1.6)
    axes[1].set_ylabel("Median contracts")
    axes[2].plot(monthly["month"], monthly["mean_iv"], color="#d62728", linewidth=1.6)
    axes[2].set_ylabel("Mean IV")
    axes[2].set_xlabel("Month")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Full-Sample Surface Feature Coverage")
    fig.tight_layout()
    png = output_path.with_suffix(".png")
    svg = output_path.with_suffix(".svg")
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close(fig)
    return [png, svg]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    feature_paths = parquet_files(root, "data/processed/full/surface_features/year=*/month=*.parquet")
    crsp_paths = parquet_files(root, "data/processed/full/crsp_monthly/year=*/month=*.parquet")
    if not feature_paths or not crsp_paths:
        print("feature_panel_status=BLOCKED_NO_FULL_SHARDS")
        return 10
    summary_path = root / "manifests" / "run_full" / "summary.json"
    if summary_path.exists() and not args.allow_partial:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") not in {"PASS"}:
            print(f"feature_panel_status=BLOCKED_FULL_STATUS_{summary.get('status')}")
            return 11
    features = read_many(feature_paths)
    crsp = read_many(crsp_paths)
    panel = add_next_month_returns(features, crsp)
    panel_path = root / "data" / "processed" / "panel" / "surface_features_panel.parquet"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path, index=False)
    figure_paths = write_coverage_figure(panel, root / "outputs" / "figures" / "full" / "surface_feature_coverage")
    manifest = {
        "status": "PASS",
        "feature_shards": len(feature_paths),
        "crsp_shards": len(crsp_paths),
        "panel_rows": int(len(panel)),
        "unique_permnos": int(panel["permno"].nunique()) if "permno" in panel else 0,
        "date_min": str(pd.to_datetime(panel["date"]).min().date()) if not panel.empty else None,
        "date_max": str(pd.to_datetime(panel["date"]).max().date()) if not panel.empty else None,
        "next_ret_nonmissing": int(panel["next_ret"].notna().sum()) if "next_ret" in panel else 0,
        "artifacts": {
            "panel": str(panel_path.relative_to(root)),
            "figures": [str(path.relative_to(root)) for path in figure_paths],
        },
    }
    write_json_atomic(dirs["manifests"] / "feature_panel_manifest.json", manifest)
    print("feature_panel_status=PASS")
    print(f"panel_rows={manifest['panel_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
