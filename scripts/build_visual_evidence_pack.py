from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from surface_returns.paths import assert_approved_root, ensure_project_dirs


PALETTE = {
    "green": "#2f7d32",
    "teal": "#4c9f9f",
    "blue": "#4c78a8",
    "orange": "#f28e2b",
    "red": "#c45a42",
    "purple": "#8f4f9f",
    "gray": "#4f5661",
    "light": "#f4f6f8",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def wealth_index(frame: pd.DataFrame, return_col: str) -> pd.Series:
    returns = pd.to_numeric(frame[return_col], errors="coerce").fillna(0.0)
    return (1.0 + returns).cumprod()


def status_counts(readiness: pd.DataFrame) -> dict[str, int]:
    if readiness.empty or "status" not in readiness:
        return {}
    return readiness["status"].value_counts().sort_index().to_dict()


def external_detail(external: dict) -> str:
    skipped = [
        item.get("source")
        for item in external.get("source_statuses", [])
        if str(item.get("status", "")).startswith("SKIPPED")
    ]
    if skipped:
        return f"skipped {','.join(str(item) for item in skipped)}"
    return "BEA/EIA/SEC loaded"


def plot_status_cards(ax, manifests: dict[str, dict], readiness: pd.DataFrame) -> None:
    ax.axis("off")
    counts = status_counts(readiness)
    extraction = manifests["extraction"]
    chars = manifests["chars"]
    ssvi = manifests["ssvi"]
    sdf = manifests["sdf"]
    gpu = manifests["gpu"]
    portfolio = manifests["portfolio"]
    regsho = manifests["regsho"]
    external = manifests["external"]
    reg_t = (
        regsho.get("regression", {})
        .get("target_coefficient", {})
        .get("t_cluster_date")
    )
    cards = [
        ("Readiness", f"{counts.get('PASS', 0)} pass\n{counts.get('BLOCKED', 0)} blocked, {counts.get('NEGATIVE_RESULT', 0)} negative", PALETTE["green"]),
        ("Sample", f"{extraction.get('completed')} shards\n1996-2024 usable", PALETTE["blue"]),
        ("No-arbitrage", f"{100 * safe_float(ssvi.get('pass_share'), 0):.1f}% SSVI pass\n{ssvi.get('surfaces')} surfaces", PALETTE["teal"]),
        ("Characteristics", f"{len(chars.get('characteristics', []))} CRSP/Compustat\n{chars.get('panel_rows')} rows", PALETTE["purple"]),
        ("GPU signal", f"rank IC {safe_float(gpu.get('mean_rank_ic'), 0):.3f}\nspread {safe_float(gpu.get('mean_top_bottom_return'), 0):.3f}/mo", PALETTE["red"]),
        ("Portfolio", f"net {100 * safe_float(portfolio.get('net', {}).get('mean_monthly_return'), 0):.2f}%/mo\nTAQ {100 * safe_float(portfolio.get('taq_net', {}).get('mean_monthly_return'), 0):.2f}%/mo", PALETTE["red"]),
        ("SDF", f"{sdf.get('oos_months')} OOS months\nRMS {safe_float(sdf.get('pricing_error', {}).get('rms'), 0):.3f}", PALETTE["orange"]),
        ("Mechanism", f"Reg SHO t {safe_float(reg_t, 0):.2f}\nweak evidence", PALETTE["gray"]),
        ("External", f"{','.join(external.get('control_sources_passed', []))}\n{external_detail(external)}", PALETTE["purple"]),
    ]
    cols = 3
    rows = 3
    for idx, (title, body, color) in enumerate(cards):
        row = idx // cols
        col = idx % cols
        x0 = col / cols + 0.012
        y0 = 1.0 - (row + 1) / rows + 0.035
        width = 1 / cols - 0.024
        height = 1 / rows - 0.07
        rect = plt.Rectangle(
            (x0, y0),
            width,
            height,
            transform=ax.transAxes,
            facecolor=PALETTE["light"],
            edgecolor="#d2d7dd",
            linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                0.012,
                height,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
            )
        )
        ax.text(x0 + 0.025, y0 + height - 0.05, title, transform=ax.transAxes, fontsize=10, weight="bold", va="top")
        ax.text(x0 + 0.025, y0 + height - 0.14, body, transform=ax.transAxes, fontsize=9, va="top", linespacing=1.25)


def plot_wealth(ax, proposal: pd.DataFrame) -> None:
    plot = proposal.sort_values("date").copy()
    lines = {
        "Gross beta-neutral": ("beta_neutral_gross_return", PALETTE["blue"]),
        "Net fixed cost": ("net_return", PALETTE["orange"]),
        "Net TAQ-calibrated": ("taq_net_return", PALETTE["red"]),
    }
    for label, (col, color) in lines.items():
        if col in plot:
            ax.plot(plot["date"], wealth_index(plot, col), label=label, color=color, linewidth=1.8)
    ax.axhline(1.0, color="#30343b", linewidth=0.8, alpha=0.7)
    ax.set_title("Portfolio Evidence: Wealth")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, fontsize=8, loc="lower left")


def plot_alpha(ax, ff5: pd.DataFrame) -> None:
    target = ["proposal_beta_neutral_ls", "proposal_net_ls", "proposal_taq_net_ls"]
    data = ff5[ff5["portfolio"].isin(target)].copy()
    data["annualized_alpha"] = pd.to_numeric(data["alpha_monthly"], errors="coerce") * 12.0
    data["t"] = pd.to_numeric(data["alpha_t_newey_west"], errors="coerce")
    data = data.sort_values("annualized_alpha")
    colors = [PALETTE["red"] if value < 0 else PALETTE["green"] for value in data["annualized_alpha"]]
    ax.barh(data["portfolio"], data["annualized_alpha"], color=colors)
    for idx, row in data.reset_index(drop=True).iterrows():
        ax.text(row["annualized_alpha"], idx, f"  t={row['t']:.2f}", va="center", fontsize=8)
    ax.axvline(0, color="#30343b", linewidth=0.8)
    ax.set_title("FF5+UMD Alpha")
    ax.set_xlabel("Annualized alpha")
    ax.grid(True, axis="x", alpha=0.24)


def plot_ssvi(ax, ssvi_monthly: pd.DataFrame) -> None:
    frame = ssvi_monthly.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    ax.plot(frame["date"], frame["pass_share"], color=PALETTE["green"], linewidth=1.6, label="Pass share")
    ax.set_ylim(0.95, 1.005)
    ax.set_ylabel("Pass share")
    ax2 = ax.twinx()
    ax2.plot(frame["date"], frame["mean_rmse_total_variance"], color=PALETTE["purple"], alpha=0.75, linewidth=1.1, label="Fit RMSE")
    ax2.set_ylabel("RMSE")
    ax.set_title("SSVI No-arbitrage Diagnostics")
    ax.grid(True, alpha=0.20)


def plot_sdf(ax, sdf_monthly: pd.DataFrame) -> None:
    frame = sdf_monthly.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    ax.plot(frame["date"], frame["rms_pricing_error"], color=PALETTE["red"], linewidth=1.5, label="Pricing")
    ax.plot(frame["date"], frame["rms_reconstruction_error"], color=PALETTE["teal"], linewidth=1.5, label="Reconstruction")
    ax.set_title("Conditional Autoencoder SDF")
    ax.set_ylabel("Monthly RMS")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, fontsize=8)


def plot_importance(ax, shap: pd.DataFrame, title: str, value_col: str) -> None:
    if shap.empty:
        ax.axis("off")
        return
    top = shap.head(10).sort_values(value_col)
    ax.barh(top["feature"], top[value_col], color=PALETTE["purple"])
    ax.set_title(title)
    ax.set_xlabel("Mean absolute value")
    ax.grid(True, axis="x", alpha=0.24)


def plot_readiness(ax, readiness: pd.DataFrame) -> None:
    label_map = {
        "External APIs": "External APIs",
        "Factor alphas, HJ/GRS, Newey-West, bootstrap inference": "Inference",
        "GPU walk-forward return model": "GPU return model",
        "Sector-balanced beta-hedged buffered portfolio": "Buffered portfolio",
        "Reg SHO pilot mechanism test": "Reg SHO DID",
        "WRDS extraction and usable sample": "Extraction",
        "Raw SVI no-arbitrage refits": "Raw SVI",
        "Global SSVI calendar and butterfly constraints": "SSVI",
    }
    plot = readiness.sort_values(["score", "requirement"]).head(8).copy()
    plot["label"] = plot["requirement"].map(label_map).fillna(plot["requirement"])
    status_color = {
        "PASS": PALETTE["green"],
        "PASS_WEAK_RESULT": "#7aa95c",
        "NEGATIVE_RESULT": PALETTE["red"],
        "BLOCKED": PALETTE["purple"],
        "PARTIAL": PALETTE["orange"],
    }
    ax.barh(plot["label"], plot["score"], color=[status_color.get(s, PALETTE["gray"]) for s in plot["status"]])
    for idx, row in plot.reset_index(drop=True).iterrows():
        label = str(row["status"]).replace("_", " ")
        if row["status"] == "PASS":
            label = "PASS"
        ax.text(float(row["score"]) + 0.025, idx, label, va="center", fontsize=7.5)
    ax.set_xlim(0, 1.28)
    ax.set_title("Readiness Bottlenecks")
    ax.set_xlabel("Score")
    ax.grid(True, axis="x", alpha=0.24)


def plot_regsho(ax, coeffs: pd.DataFrame) -> None:
    terms = ["surface_signal", "signal_x_pilot", "signal_x_post", "signal_x_pilot_x_post"]
    data = coeffs[coeffs["term"].isin(terms)].copy()
    labels = {
        "surface_signal": "surface",
        "signal_x_pilot": "signal x pilot",
        "signal_x_post": "signal x post",
        "signal_x_pilot_x_post": "triple",
    }
    data["coefficient"] = pd.to_numeric(data["coefficient"], errors="coerce")
    data["t_cluster_date"] = pd.to_numeric(data["t_cluster_date"], errors="coerce")
    data = data.sort_values("coefficient")
    data["label"] = data["term"].map(labels).fillna(data["term"])
    ax.barh(data["label"], data["coefficient"], color=[PALETTE["red"] if v < 0 else PALETTE["green"] for v in data["coefficient"]])
    for idx, row in data.reset_index(drop=True).iterrows():
        ax.text(row["coefficient"], idx, f"  t={row['t_cluster_date']:.2f}", va="center", fontsize=8)
    ax.axvline(0, color="#30343b", linewidth=0.8)
    ax.set_title("Reg SHO Mechanism")
    ax.set_xlabel("Coefficient")
    ax.grid(True, axis="x", alpha=0.24)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-root-guard", action="store_true")
    args = parser.parse_args()

    root = Path.cwd() if args.no_root_guard else assert_approved_root(Path.cwd())
    ensure_project_dirs(root)
    reports = root / "outputs" / "reports"
    figures = root / "outputs" / "figures" / "full"
    figures.mkdir(parents=True, exist_ok=True)
    manifests = root / "manifests"

    proposal = pd.read_csv(reports / "backtests" / "proposal_buffered_portfolio.csv")
    proposal["date"] = pd.to_datetime(proposal["date"])
    ff5 = pd.read_csv(reports / "inference" / "ff5_proposal_alphas.csv")
    ssvi = pd.read_csv(reports / "surfaces" / "ssvi_monthly_summary.csv")
    sdf = pd.read_csv(reports / "sdf" / "conditional_autoencoder_sdf_monthly.csv")
    shap = pd.read_csv(reports / "interpretation" / "tree_shap_feature_importance.csv")
    readiness = pd.read_csv(reports / "readiness" / "proposal_readiness_audit.csv")
    regsho = pd.read_csv(reports / "regsho" / "regsho_pilot_did_coefficients.csv")

    manifest_map = {
        "extraction": read_json(manifests / "run_full" / "summary.json"),
        "chars": read_json(manifests / "characteristic_library_manifest.json"),
        "ssvi": read_json(manifests / "ssvi_fit" / "summary.json"),
        "sdf": read_json(manifests / "conditional_autoencoder_sdf_manifest.json"),
        "gpu": read_json(manifests / "gpu_return_model_manifest.json"),
        "portfolio": read_json(manifests / "proposal_portfolio_manifest.json"),
        "regsho": read_json(manifests / "regsho_pilot_did_manifest.json"),
        "external": read_json(manifests / "external_api_controls_manifest.json"),
    }

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig = plt.figure(figsize=(18, 13), facecolor="white")
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1.05, 1.25, 1.15, 1.15], hspace=0.55, wspace=0.45)
    fig.suptitle("Surface-to-Returns Proposal Evidence Pack", fontsize=18, weight="bold", y=0.982)
    fig.text(
        0.5,
        0.955,
        "Implementation evidence, no-arbitrage diagnostics, model interpretation, and disciplined return results",
        ha="center",
        fontsize=10.5,
        color=PALETTE["gray"],
    )

    plot_status_cards(fig.add_subplot(gs[0, :]), manifest_map, readiness)
    plot_wealth(fig.add_subplot(gs[1, 0]), proposal)
    plot_alpha(fig.add_subplot(gs[1, 1]), ff5)
    plot_readiness(fig.add_subplot(gs[1, 2]), readiness)
    plot_ssvi(fig.add_subplot(gs[2, 0]), ssvi)
    plot_sdf(fig.add_subplot(gs[2, 1]), sdf)
    plot_importance(fig.add_subplot(gs[2, 2]), shap, "TreeSHAP Importance", "mean_abs_shap")
    plot_regsho(fig.add_subplot(gs[3, 0]), regsho)
    plot_importance(fig.add_subplot(gs[3, 1]), pd.read_csv(reports / "interpretation" / "top_integrated_gradients.csv"), "SDF Integrated Gradients", "mean_abs_attribution")

    ax = fig.add_subplot(gs[3, 2])
    ax.axis("off")
    notes = [
        "Current Interpretation",
        "1. Surface construction and SSVI no-arbitrage gates pass at full usable scale.",
        "2. Full-characteristic return prediction is weak to negative in the latest refresh.",
        "3. The SDF and interpretation stack are implemented and refreshed on the expanded panel.",
        f"4. External controls: {external_detail(manifest_map['external'])}.",
    ]
    y = 0.96
    for idx, line in enumerate(notes):
        ax.text(
            0.02,
            y,
            line,
            transform=ax.transAxes,
            fontsize=11 if idx == 0 else 9.5,
            weight="bold" if idx == 0 else "normal",
            va="top",
            color="#20242a",
            wrap=True,
        )
        y -= 0.14 if idx == 0 else 0.18

    fig.text(
        0.01,
        0.012,
        "Generated from non-raw reports and manifests. Negative return evidence is shown as evidence discipline, not as a failed visualization.",
        fontsize=9,
        color=PALETTE["gray"],
    )
    fig.subplots_adjust(top=0.90, bottom=0.055, left=0.075, right=0.982, hspace=0.62, wspace=0.45)
    paths = [figures / "visual_evidence_pack.png", figures / "visual_evidence_pack.svg"]
    for path in paths:
        fig.savefig(path, dpi=240)
    plt.close(fig)
    print("visual_evidence_pack_status=PASS")
    for path in paths:
        print(path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
