from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs


CHECK_COLUMNS = [
    "positive_total_variance",
    "theta_monotone",
    "theta_phi_monotone",
    "calendar_monotone_grid",
    "ssvi_bounds_pass",
]


def load_diagnostics(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("data/processed/full/ssvi_surface_diagnostics/year=*/month=*.parquet"))
    if not paths:
        raise FileNotFoundError("No SSVI diagnostic shards found.")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    return frame


def write_figure(monthly: pd.DataFrame, diagnostics: pd.DataFrame, output_prefix: Path) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].plot(monthly["date"], monthly["pass_share"], color="#4c78a8", linewidth=1.6)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Monthly SSVI Pass Share")
    axes[0].set_ylabel("Share")
    axes[0].grid(True, alpha=0.25)

    rmse = pd.to_numeric(diagnostics["rmse_total_variance"], errors="coerce").dropna()
    axes[1].hist(rmse, bins=40, color="#f58518", alpha=0.85)
    axes[1].set_title("SSVI Fit Error")
    axes[1].set_xlabel("RMSE total variance")
    axes[1].set_ylabel("Surfaces")
    axes[1].grid(True, axis="y", alpha=0.25)

    pass_counts = diagnostics[CHECK_COLUMNS].fillna(False).astype(bool).sum().sort_values()
    axes[2].barh(pass_counts.index, pass_counts.values, color="#54a24b")
    axes[2].set_title("No-Arbitrage Checks")
    axes[2].set_xlabel("Passing surfaces")
    axes[2].grid(True, axis="x", alpha=0.25)

    fig.suptitle("Global SSVI No-Arbitrage Diagnostics")
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    diagnostics = load_diagnostics(root)
    report_dir = root / "outputs" / "reports" / "surfaces"
    report_dir.mkdir(parents=True, exist_ok=True)

    monthly = (
        diagnostics.assign(_pass=diagnostics["status"].eq("PASS"))
        .groupby("date", as_index=False)
        .agg(
            surfaces=("status", "size"),
            pass_surfaces=("_pass", "sum"),
            pass_share=("_pass", "mean"),
            mean_rmse_total_variance=("rmse_total_variance", "mean"),
            max_theta_projection_abs=("theta_projection_max_abs", "max"),
        )
    )
    monthly_path = report_dir / "ssvi_monthly_summary.csv"
    aggregate_path = report_dir / "ssvi_fit_aggregate.json"
    monthly.to_csv(monthly_path, index=False)

    aggregate = {
        "status": "PASS" if diagnostics["status"].eq("PASS").all() else "PARTIAL",
        "surfaces": int(len(diagnostics)),
        "months": int(diagnostics["date"].nunique()),
        "pass_surfaces": int(diagnostics["status"].eq("PASS").sum()),
        "pass_share": float(diagnostics["status"].eq("PASS").mean()),
        "mean_rmse_total_variance": float(pd.to_numeric(diagnostics["rmse_total_variance"], errors="coerce").mean()),
        "median_rmse_total_variance": float(pd.to_numeric(diagnostics["rmse_total_variance"], errors="coerce").median()),
        "max_theta_projection_abs": float(pd.to_numeric(diagnostics["theta_projection_max_abs"], errors="coerce").max()),
        "mean_theta_projection_abs": float(pd.to_numeric(diagnostics["theta_projection_max_abs"], errors="coerce").mean()),
        "no_arbitrage_pass_counts": {
            col: int(diagnostics[col].fillna(False).astype(bool).sum()) for col in CHECK_COLUMNS
        },
    }
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    figures = write_figure(monthly, diagnostics, root / "outputs" / "figures" / "full" / "ssvi_no_arbitrage_diagnostics")
    manifest = {
        "status": aggregate["status"],
        "surfaces": aggregate["surfaces"],
        "months": aggregate["months"],
        "pass_share": aggregate["pass_share"],
        "artifacts": {
            "monthly_csv": str(monthly_path.relative_to(root)),
            "aggregate_json": str(aggregate_path.relative_to(root)),
            "figures": [str(path.relative_to(root)) for path in figures],
        },
    }
    write_json_atomic(dirs["manifests"] / "ssvi_summary_manifest.json", manifest)
    print("ssvi_summary_status=" + str(manifest["status"]))
    print(f"surfaces={manifest['surfaces']} pass_share={manifest['pass_share']:.4f}")
    return 0 if manifest["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
