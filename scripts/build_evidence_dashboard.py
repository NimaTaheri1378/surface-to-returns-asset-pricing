from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def main() -> int:
    root = Path.cwd()
    manifests = root / "manifests"
    figures = root / "outputs" / "figures" / "full"
    figures.mkdir(parents=True, exist_ok=True)

    proposal = pd.read_csv(root / "outputs" / "reports" / "backtests" / "proposal_buffered_portfolio.csv")
    proposal["date"] = pd.to_datetime(proposal["date"])
    sdf_monthly = pd.read_csv(root / "outputs" / "reports" / "sdf" / "conditional_autoencoder_sdf_monthly.csv")
    sdf_monthly["date"] = pd.to_datetime(sdf_monthly["date"])
    attr = pd.read_csv(root / "outputs" / "reports" / "interpretation" / "top_integrated_gradients.csv")
    shap_path = root / "outputs" / "reports" / "interpretation" / "tree_shap_feature_importance.csv"
    shap = pd.read_csv(shap_path) if shap_path.exists() else pd.DataFrame()
    external_path = root / "outputs" / "reports" / "external" / "monthly_external_api_coverage.csv"
    external = pd.read_csv(external_path) if external_path.exists() else pd.DataFrame()
    ff5 = pd.read_csv(root / "outputs" / "reports" / "inference" / "ff5_proposal_alphas.csv")

    extraction = read_json(manifests / "run_full" / "summary.json")
    gpu = read_json(manifests / "gpu_return_model_manifest.json")
    portfolio = read_json(manifests / "proposal_portfolio_manifest.json")
    inference = read_json(manifests / "asset_pricing_inference_manifest.json")
    sdf = read_json(manifests / "conditional_autoencoder_sdf_manifest.json")
    external_manifest = read_json(manifests / "external_api_controls_manifest.json")
    ssvi = read_json(manifests / "ssvi_fit" / "summary.json")
    taq = read_json(manifests / "taq_cost_manifest_calib_2015_2024_m12_s50.json")
    regsho = read_json(manifests / "regsho_pilot_did_manifest.json")

    plot = proposal.sort_values("date").copy()
    wealth_cols = {
        "Beta-neutral gross": "beta_neutral_gross_return",
        "Net fixed cost": "net_return",
        "Net TAQ-calibrated": "taq_net_return",
    }
    for label, col in wealth_cols.items():
        if col in plot:
            plot[label] = (1.0 + plot[col].fillna(0.0)).cumprod()

    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    ax = axes[0, 0]
    for label in wealth_cols:
        if label in plot:
            ax.plot(plot["date"], plot[label], linewidth=1.8, label=label)
    ax.set_title("Buffered Portfolio Wealth")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    alpha = ff5[ff5["portfolio"].isin(["proposal_beta_neutral_ls", "proposal_net_ls", "proposal_taq_net_ls"])].copy()
    if not alpha.empty:
        alpha["annualized_alpha"] = alpha["alpha_monthly"] * 12.0
        alpha = alpha.sort_values("annualized_alpha")
        ax.barh(alpha["portfolio"], alpha["annualized_alpha"], color="#4c78a8")
        ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("FF5+UMD Annualized Alpha")
    ax.set_xlabel("Alpha")
    ax.grid(True, axis="x", alpha=0.25)

    ax = axes[1, 0]
    sdf_plot = sdf_monthly.sort_values("date")
    ax.plot(sdf_plot["date"], sdf_plot["rms_pricing_error"], label="Pricing RMS", color="#e45756")
    ax.plot(sdf_plot["date"], sdf_plot["rms_reconstruction_error"], label="Reconstruction RMS", color="#54a24b")
    ax.set_title("Conditional Autoencoder SDF Errors")
    ax.set_ylabel("Monthly RMS")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    if not attr.empty:
        top = attr.head(10).sort_values("mean_abs_attribution")
        ax.barh(top["feature"], top["mean_abs_attribution"], color="#f58518")
    ax.set_title("Top SDF Integrated Gradients")
    ax.set_xlabel("Mean absolute attribution")
    ax.grid(True, axis="x", alpha=0.25)

    ax = axes[0, 2]
    if not external.empty:
        coverage = external.head(10).sort_values("nonmissing")
        ax.barh(coverage["column"], coverage["nonmissing"], color="#72b7b2")
    ax.set_title("External API Coverage")
    ax.set_xlabel("Nonmissing months")
    ax.grid(True, axis="x", alpha=0.25)

    ax = axes[1, 2]
    if not shap.empty:
        top = shap.head(10).sort_values("mean_abs_shap")
        ax.barh(top["feature"], top["mean_abs_shap"], color="#b279a2")
    ax.set_title("TreeSHAP Importance")
    ax.set_xlabel("Mean absolute SHAP")
    ax.grid(True, axis="x", alpha=0.25)

    external_sources = ",".join(external_manifest.get("control_sources_passed", [])) or "none"
    shap_top = shap["feature"].iloc[0] if not shap.empty else "n/a"
    regsho_t = (
        regsho.get("regression", {})
        .get("target_coefficient", {})
        .get("t_cluster_date")
    )
    summary_top = (
        f"Extraction: {extraction.get('completed')} completed, {extraction.get('skipped')} skipped, "
        f"{extraction.get('failed')} failed | "
        f"GPU rank IC: {gpu.get('mean_rank_ic'):.3f} | "
        f"Proposal net mean: {pct(portfolio.get('net', {}).get('mean_monthly_return'))}/mo | "
        f"FF5+UMD net alpha t: {inference.get('proposal_net_ff5_alpha_t'):.2f} | "
        f"SDF OOS months: {sdf.get('oos_months')} | "
        f"SSVI pass: {100*ssvi.get('pass_share', 0):.1f}%"
    )
    summary_bottom = (
        f"Reg SHO triple t: {num(regsho_t)} | "
        f"External API: {external_sources} | "
        f"Top SHAP: {shap_top} | "
        f"TAQ exact/calibrated: {100*taq.get('exact_taq_share', 0):.2f}%/"
        f"{100*taq.get('calibrated_taq_share', 0):.2f}%"
    )
    fig.suptitle("Surface-to-Returns Full-Sample Evidence Dashboard", fontsize=16)
    fig.text(0.5, 0.027, summary_top, ha="center", va="bottom", fontsize=8.6)
    fig.text(0.5, 0.011, summary_bottom, ha="center", va="bottom", fontsize=8.6)
    fig.tight_layout(rect=[0, 0.065, 1, 0.95])
    for suffix in [".png", ".svg"]:
        fig.savefig(figures / f"evidence_dashboard{suffix}", dpi=240)
    plt.close(fig)
    print("evidence_dashboard_status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
