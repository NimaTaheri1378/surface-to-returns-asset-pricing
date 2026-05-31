from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from surface_returns.interpretation import (
    beta_characteristic_correlations,
    latent_factor_correlations,
    top_abs_attributions,
)
from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs


def load_existing(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def candidate_panel(root: Path) -> Path:
    candidates = [
        root / "data" / "processed" / "panel" / "surface_characteristic_state_ibes_regsho_external_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_taq_external_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_external_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_external_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_ibes_regsho_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_taq_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_panel.parquet",
    ]
    return next((path for path in candidates if path.exists()), candidates[-1])


def write_interpretation_figure(
    attributions: pd.DataFrame,
    factor_corrs: pd.DataFrame,
    beta_corrs: pd.DataFrame,
    output_prefix: Path,
) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    if not attributions.empty:
        plot = attributions.head(15).sort_values("mean_abs_attribution")
        axes[0].barh(plot["feature"], plot["mean_abs_attribution"], color="#4c78a8")
    axes[0].set_title("Integrated Gradients")
    axes[0].set_xlabel("Mean absolute attribution")
    axes[0].grid(True, axis="x", alpha=0.25)

    if not factor_corrs.empty:
        plot = factor_corrs.head(15).copy()
        plot["label"] = plot["latent_factor"] + " / " + plot["state_variable"]
        plot = plot.sort_values("correlation")
        axes[1].barh(plot["label"], plot["correlation"], color="#f58518")
    axes[1].set_title("Latent Factor State Links")
    axes[1].set_xlabel("Correlation")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].grid(True, axis="x", alpha=0.25)

    if not beta_corrs.empty:
        plot = beta_corrs.head(15).copy()
        plot["label"] = plot["latent_beta"] + " / " + plot["characteristic"]
        plot = plot.sort_values("correlation")
        axes[2].barh(plot["label"], plot["correlation"], color="#54a24b")
    axes[2].set_title("Latent Beta Characteristics")
    axes[2].set_xlabel("Correlation")
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].grid(True, axis="x", alpha=0.25)
    fig.suptitle("Conditional SDF Interpretation")
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    report_dir = root / "outputs" / "reports" / "interpretation"
    report_dir.mkdir(parents=True, exist_ok=True)

    sdf_dir = root / "outputs" / "reports" / "sdf"
    attr = load_existing(sdf_dir / "conditional_autoencoder_sdf_integrated_gradients.csv")
    assets = load_existing(sdf_dir / "conditional_autoencoder_sdf_assets.parquet")
    factors = load_existing(sdf_dir / "conditional_autoencoder_sdf_factors.csv")
    state_candidates = [
        root / "data" / "processed" / "states" / "monthly_state_controls_external.parquet",
        root / "data" / "processed" / "states" / "monthly_state_controls.parquet",
    ]
    state_path = next((path for path in state_candidates if path.exists()), state_candidates[-1])
    states = load_existing(state_path, required=False)
    panel = load_existing(candidate_panel(root))

    top_attr = top_abs_attributions(attr, top_n=args.top_n)
    factor_corrs = latent_factor_correlations(factors, states) if not states.empty else pd.DataFrame()
    beta_corrs = beta_characteristic_correlations(assets, panel)

    attr_path = report_dir / "top_integrated_gradients.csv"
    factor_path = report_dir / "latent_factor_state_correlations.csv"
    beta_path = report_dir / "latent_beta_characteristic_correlations.csv"
    top_attr.to_csv(attr_path, index=False)
    factor_corrs.to_csv(factor_path, index=False)
    beta_corrs.to_csv(beta_path, index=False)
    figures = write_interpretation_figure(
        top_attr,
        factor_corrs,
        beta_corrs,
        root / "outputs" / "figures" / "full" / "conditional_sdf_interpretation",
    )
    manifest = {
        "status": "PASS",
        "top_attributions": int(len(top_attr)),
        "latent_factor_correlations": int(len(factor_corrs)),
        "latent_beta_correlations": int(len(beta_corrs)),
        "artifacts": {
            "top_integrated_gradients": str(attr_path.relative_to(root)),
            "latent_factor_state_correlations": str(factor_path.relative_to(root)),
            "latent_beta_characteristic_correlations": str(beta_path.relative_to(root)),
            "figures": [str(path.relative_to(root)) for path in figures],
        },
    }
    write_json_atomic(dirs["manifests"] / "model_interpretation_manifest.json", manifest)
    print("model_interpretation_status=PASS")
    print(
        f"top_attributions={manifest['top_attributions']} "
        f"factor_corrs={manifest['latent_factor_correlations']} "
        f"beta_corrs={manifest['latent_beta_correlations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
