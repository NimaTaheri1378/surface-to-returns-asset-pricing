from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from surface_returns.paths import assert_approved_root, ensure_project_dirs


PALETTE = {
    "ink": "#1f2933",
    "muted": "#5b6573",
    "line": "#d8dee6",
    "panel": "#f6f8fb",
    "green": "#2f7d32",
    "teal": "#3d9292",
    "blue": "#356da3",
    "orange": "#c77d20",
    "red": "#c45a42",
    "purple": "#8f4f9f",
    "gold": "#d6a53a",
    "gray": "#6b7280",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def fmt_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_pct(value: object, digits: int = 1) -> str:
    number = safe_float(value)
    return "n/a" if not np.isfinite(number) else f"{100.0 * number:.{digits}f}%"


def pretty_label(value: object) -> str:
    text = str(value).replace("_", " ").replace("/", " / ")
    return " ".join(part.capitalize() for part in text.split())


def wealth_index(frame: pd.DataFrame, return_col: str) -> pd.Series:
    returns = pd.to_numeric(frame[return_col], errors="coerce").fillna(0.0)
    return (1.0 + returns).cumprod()


def save_figure(fig, root: Path, stem: str) -> list[str]:
    figures = root / "outputs" / "figures" / "full"
    figures.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in [".png", ".svg"]:
        path = figures / f"{stem}{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path.relative_to(root)))
    plt.close(fig)
    return paths


def style_ax(ax, title: str | None = None, grid_axis: str = "y") -> None:
    if title:
        ax.set_title(title, loc="left", fontsize=13, weight="bold", color=PALETTE["ink"])
    ax.grid(True, axis=grid_axis, alpha=0.22)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=PALETTE["ink"], labelsize=9)


def add_note(ax, lines: list[str], title: str = "Evidence Summary") -> None:
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.04),
            0.96,
            0.90,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor="#fff7f4",
            edgecolor="none",
        )
    )
    ax.text(0.075, 0.84, title, transform=ax.transAxes, fontsize=13, weight="bold", color=PALETTE["ink"])
    y = 0.68
    for line in lines:
        wrapped = textwrap.fill(f"- {line}", width=60, subsequent_indent="  ")
        ax.text(
            0.085,
            y,
            wrapped,
            transform=ax.transAxes,
            fontsize=8.8,
            color=PALETTE["ink"],
            va="top",
            linespacing=1.12,
        )
        y -= 0.075 * (wrapped.count("\n") + 1) + 0.035


def draw_flow(ax, labels: list[str]) -> None:
    ax.axis("off")
    x0, y0, w, h = 0.025, 0.18, 0.148, 0.62
    gap = 0.013
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["purple"], PALETTE["orange"], PALETTE["gold"], PALETTE["green"]]
    for idx, label in enumerate(labels):
        x = x0 + idx * (w + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, y0),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.02",
                transform=ax.transAxes,
                facecolor="white",
                edgecolor=PALETTE["line"],
                linewidth=1.0,
            )
        )
        ax.add_patch(Rectangle((x, y0), 0.010, h, transform=ax.transAxes, color=colors[idx], lw=0))
        center_y = y0 + h / 2
        ax.text(x + 0.024, center_y, str(idx + 1), transform=ax.transAxes, fontsize=15, weight="bold", color=colors[idx], va="center")
        ax.text(x + 0.058, center_y, label, transform=ax.transAxes, fontsize=8.9, color=PALETTE["ink"], va="center", linespacing=1.08)
        if idx < len(labels) - 1:
            arrow_x = x + w + gap * 0.30
            ax.add_patch(
                Polygon(
                    [[arrow_x, y0 + h / 2], [arrow_x + gap * 0.40, y0 + h / 2 + 0.045], [arrow_x + gap * 0.40, y0 + h / 2 - 0.045]],
                    closed=True,
                    transform=ax.transAxes,
                    facecolor=PALETTE["line"],
                    edgecolor=PALETTE["line"],
                )
            )


def metric_strip(ax, metrics: list[tuple[str, str, str, str]]) -> None:
    ax.axis("off")
    n = len(metrics)
    gap = 0.012
    w = (1 - gap * (n - 1)) / n
    for idx, (label, value, detail, color) in enumerate(metrics):
        x = idx * (w + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.05),
                w,
                0.86,
                boxstyle="round,pad=0.010,rounding_size=0.018",
                transform=ax.transAxes,
                facecolor=PALETTE["panel"],
                edgecolor=PALETTE["line"],
            )
        )
        ax.add_patch(Rectangle((x, 0.88), w, 0.04, transform=ax.transAxes, color=color, lw=0))
        ax.text(x + 0.025, 0.72, label.upper(), transform=ax.transAxes, fontsize=7.8, color=PALETTE["muted"], weight="bold")
        ax.text(x + 0.025, 0.43, value, transform=ax.transAxes, fontsize=16, color=PALETTE["ink"], weight="bold")
        ax.text(x + 0.025, 0.20, detail, transform=ax.transAxes, fontsize=8.2, color=PALETTE["muted"])


def figure_1(root: Path, ctx: dict) -> list[str]:
    fig = plt.figure(figsize=(13.5, 7.2), facecolor="white")
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.70, 0.58, 1.15], width_ratios=[1.12, 0.88], hspace=0.14, wspace=0.30)
    fig.suptitle("Figure 1. Data and Modeling Stack", x=0.04, y=0.98, ha="left", fontsize=20, weight="bold", color=PALETTE["ink"])
    fig.text(0.04, 0.925, "WRDS extraction, no-arbitrage surface construction, characteristics, costs, inference, and interpretation at the usable 1996-2024 scale.", fontsize=10.5, color=PALETTE["muted"])

    metric_strip(
        fig.add_subplot(gs[0, :]),
        [
            ("Sample", "1996-2024", f"{fmt_int(ctx['extraction'].get('completed'))} shards", PALETTE["blue"]),
            ("SSVI", fmt_pct(ctx["ssvi"].get("pass_share")), f"{fmt_int(ctx['ssvi'].get('surfaces'))} surfaces", PALETTE["green"]),
            ("Characteristics", fmt_int(len(ctx["chars"].get("characteristics", []))), f"{fmt_int(ctx['chars'].get('panel_rows'))} rows", PALETTE["purple"]),
            ("SDF", f"{safe_float(ctx['sdf'].get('pricing_error', {}).get('rms')):.3f}", f"{fmt_int(ctx['sdf'].get('oos_months'))} OOS months", PALETTE["teal"]),
        ],
    )
    draw_flow(fig.add_subplot(gs[1, :]), ["WRDS\nlinkage", "SVI/SSVI\nsurfaces", "Firm\ncharacteristics", "GPU/SDF\nmodels", "TAQ-cost\nportfolios", "Inference and\ninterpretation"])

    ax = fig.add_subplot(gs[2, 0])
    labels = ["Panel rows", "OOS predictions", "SDF assets"]
    values = [
        safe_float(ctx["chars"].get("panel_rows"), 0),
        safe_float(ctx["gpu"].get("observations"), 0),
        safe_float(ctx["sdf"].get("oos_assets"), 0),
    ]
    colors = [PALETTE["blue"], PALETTE["purple"], PALETTE["teal"]]
    ax.barh(labels, values, color=colors)
    for idx, value in enumerate(values):
        ax.text(value, idx, f"  {fmt_int(value)}", va="center", fontsize=9, weight="bold")
    ax.set_xlabel("Rows")
    style_ax(ax, "Empirical Scale", "x")

    add_note(
        fig.add_subplot(gs[2, 1]),
        [
            "The option-surface layer passes no-arbitrage gates at scale.",
            "The SDF supplies interpretable pricing-kernel diagnostics.",
            "Cost-aware portfolios place return tests in economic-cost context.",
        ],
        "Empirical Interpretation",
    )
    return save_figure(fig, root, "paper_figure_1_pipeline")


def figure_2(root: Path, ctx: dict) -> list[str]:
    ssvi = ctx["ssvi_monthly"].copy()
    ssvi["date"] = pd.to_datetime(ssvi["date"])
    fig = plt.figure(figsize=(13.5, 7.6), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.26)
    fig.suptitle("Figure 2. SSVI Surface Quality and No-Arbitrage Diagnostics", x=0.04, y=0.98, ha="left", fontsize=19, weight="bold", color=PALETTE["ink"])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ssvi["date"], ssvi["pass_share"], color=PALETTE["green"], linewidth=2.0)
    ax.set_ylim(0.95, 1.005)
    ax.set_ylabel("Pass share")
    style_ax(ax, "Calendar and Butterfly Gate Pass Share")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ssvi["date"], ssvi["mean_rmse_total_variance"], color=PALETTE["purple"], linewidth=1.5)
    ax.set_ylabel("RMSE")
    style_ax(ax, "Fit Error Through Time")

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(ssvi["max_theta_projection_abs"].dropna(), bins=30, color=PALETTE["teal"], alpha=0.90)
    ax.set_xlabel("Absolute monotone-theta projection")
    ax.set_ylabel("Months")
    style_ax(ax, "Projection Needed to Enforce Monotone Theta")

    add_note(
        fig.add_subplot(gs[1, 1]),
        [
            f"{fmt_int(ctx['ssvi'].get('pass_surfaces'))} of {fmt_int(ctx['ssvi'].get('surfaces'))} date-security surfaces pass.",
            "Diagnostics enforce positivity, monotone theta, monotone theta-phi, calendar-grid monotonicity, and Gatheral-Jacquier bounds.",
            f"Maximum theta projection in monthly report: {ssvi['max_theta_projection_abs'].max():.2e}.",
        ],
        "Surface Evidence",
    )
    return save_figure(fig, root, "paper_figure_2_surface_quality")


def figure_3(root: Path, ctx: dict) -> list[str]:
    sdf = ctx["sdf_monthly"].copy()
    sdf["date"] = pd.to_datetime(sdf["date"])
    ig = ctx["integrated_gradients"].head(10).sort_values("mean_abs_attribution")
    shap = ctx["shap"].head(10).sort_values("mean_abs_shap")
    fig = plt.figure(figsize=(13.5, 7.8), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.34)
    fig.suptitle("Figure 3. Conditional Autoencoder SDF and Interpretation", x=0.04, y=0.98, ha="left", fontsize=19, weight="bold", color=PALETTE["ink"])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(sdf["date"], sdf["rms_pricing_error"], color=PALETTE["red"], label="Pricing RMS", linewidth=1.7)
    ax.plot(sdf["date"], sdf["rms_reconstruction_error"], color=PALETTE["teal"], label="Reconstruction RMS", linewidth=1.7)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylabel("Monthly RMS")
    style_ax(ax, "SDF Pricing and Reconstruction Errors")

    ax = fig.add_subplot(gs[0, 1])
    groups = ctx["sdf"].get("architecture", {})
    labels = ["Surface", "Tabular", "State"]
    values = [groups.get(name, 0) for name in ["surface_branch", "tabular_branch", "state_branch"]]
    ax.bar(labels, values, color=[PALETTE["blue"], PALETTE["purple"], PALETTE["teal"]])
    ax.set_ylabel("Inputs")
    style_ax(ax, "Three-Branch Feature Architecture")

    ax = fig.add_subplot(gs[1, 0])
    ax.barh(ig["feature"], ig["mean_abs_attribution"], color=PALETTE["orange"])
    ax.set_xlabel("Mean absolute attribution")
    style_ax(ax, "Integrated Gradients", "x")

    ax = fig.add_subplot(gs[1, 1])
    ax.barh(shap["feature"], shap["mean_abs_shap"], color=PALETTE["purple"])
    ax.set_xlabel("Mean absolute SHAP")
    style_ax(ax, "TreeSHAP Feature Importance", "x")
    return save_figure(fig, root, "paper_figure_3_sdf_interpretation")


def figure_4(root: Path, ctx: dict) -> list[str]:
    proposal = ctx["proposal"].copy()
    proposal["date"] = pd.to_datetime(proposal["date"])
    summary = ctx["proposal_summary"].copy()
    alphas = ctx["ff5_alphas"].copy()
    alphas["alpha_annualized"] = pd.to_numeric(alphas["alpha_monthly"], errors="coerce") * 12.0
    fig = plt.figure(figsize=(13.5, 7.8), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.30)
    fig.suptitle("Figure 4. Cost-Aware Return and Asset-Pricing Evidence", x=0.04, y=0.98, ha="left", fontsize=19, weight="bold", color=PALETTE["ink"])

    ax = fig.add_subplot(gs[0, 0])
    for label, col, color in [
        ("Gross beta-neutral", "beta_neutral_gross_return", PALETTE["blue"]),
        ("Net fixed cost", "net_return", PALETTE["orange"]),
        ("Net TAQ-calibrated", "taq_net_return", PALETTE["red"]),
    ]:
        ax.plot(proposal["date"], wealth_index(proposal, col), label=label, linewidth=1.8, color=color)
    ax.axhline(1.0, color=PALETTE["ink"], linewidth=0.8, alpha=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_ylabel("Growth of $1")
    style_ax(ax, "Buffered Long-Short Wealth")

    ax = fig.add_subplot(gs[0, 1])
    plot_summary = summary[summary["return_type"].isin(["gross", "beta_neutral_gross", "net", "taq_net"])].copy()
    summary_labels = {
        "gross": "Gross",
        "beta_neutral_gross": "Beta-neutral",
        "net": "Net",
        "taq_net": "TAQ net",
    }
    colors = [PALETTE["green"] if value > 0 else PALETTE["red"] for value in plot_summary["annualized_sharpe"]]
    ax.barh(plot_summary["return_type"].map(summary_labels), plot_summary["annualized_sharpe"], color=colors)
    ax.axvline(0, color=PALETTE["ink"], linewidth=0.8)
    ax.set_xlabel("Annualized Sharpe")
    style_ax(ax, "Portfolio Summary", "x")

    ax = fig.add_subplot(gs[1, 0])
    target = alphas[alphas["portfolio"].isin(["proposal_beta_neutral_ls", "proposal_net_ls", "proposal_taq_net_ls"])].copy()
    target = target.sort_values("alpha_annualized")
    alpha_labels = {
        "proposal_beta_neutral_ls": "Beta-neutral LS",
        "proposal_net_ls": "Net LS",
        "proposal_taq_net_ls": "TAQ-net LS",
    }
    ax.barh(target["portfolio"].map(alpha_labels), target["alpha_annualized"], color=PALETTE["red"])
    for idx, row in target.reset_index(drop=True).iterrows():
        ax.text(row["alpha_annualized"], idx, f"  t={row['alpha_t_newey_west']:.2f}", va="center", fontsize=8)
    ax.axvline(0, color=PALETTE["ink"], linewidth=0.8)
    ax.set_xlabel("Annualized alpha")
    style_ax(ax, "FF5+UMD Alpha", "x")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(proposal["date"], proposal["turnover"], color=PALETTE["gray"], label="Turnover", linewidth=1.4)
    ax.set_ylabel("Turnover")
    ax2 = ax.twinx()
    ax2.plot(proposal["date"], proposal["taq_total_cost"] * 10000.0, color=PALETTE["red"], label="TAQ cost", linewidth=1.4)
    ax2.plot(proposal["date"], proposal["fixed_cost"] * 10000.0, color=PALETTE["orange"], label="Fixed cost", linewidth=1.2)
    ax2.set_ylabel("Cost, bps")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper right")
    style_ax(ax, "Trading Frictions")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.tick_params(colors=PALETTE["ink"], labelsize=9)
    return save_figure(fig, root, "paper_figure_4_return_evidence")


def figure_5(root: Path, ctx: dict) -> list[str]:
    coeffs = ctx["regsho"].copy()
    external = ctx["external"].copy()
    shap = ctx["shap"].head(10).sort_values("mean_abs_shap")
    fig = plt.figure(figsize=(13.5, 7.8), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.34)
    fig.suptitle("Figure 5. Mechanisms, External Controls, and Interpretation", x=0.04, y=0.98, ha="left", fontsize=19, weight="bold", color=PALETTE["ink"])

    ax = fig.add_subplot(gs[0, 0])
    terms = ["surface_signal", "signal_x_pilot", "signal_x_post", "signal_x_pilot_x_post"]
    labels = {"surface_signal": "Surface", "signal_x_pilot": "Signal x pilot", "signal_x_post": "Signal x post", "signal_x_pilot_x_post": "Triple"}
    reg = coeffs[coeffs["term"].isin(terms)].copy()
    reg["label"] = reg["term"].map(labels)
    reg = reg.sort_values("coefficient")
    ax.barh(reg["label"], reg["coefficient"], color=[PALETTE["red"] if value < 0 else PALETTE["green"] for value in reg["coefficient"]])
    for idx, row in reg.reset_index(drop=True).iterrows():
        ax.text(row["coefficient"], idx, f"  t={row['t_cluster_date']:.2f}", va="center", fontsize=8)
    ax.axvline(0, color=PALETTE["ink"], linewidth=0.8)
    ax.set_xlabel("Coefficient")
    style_ax(ax, "Reg SHO Pilot DID", "x")

    ax = fig.add_subplot(gs[0, 1])
    coverage = external.head(10).sort_values("nonmissing")
    ax.barh(coverage["column"], coverage["nonmissing"], color=PALETTE["teal"])
    ax.set_xlabel("Nonmissing months")
    style_ax(ax, "External Control Coverage", "x")

    ax = fig.add_subplot(gs[1, 0])
    ax.barh(shap["feature"], shap["mean_abs_shap"], color=PALETTE["purple"])
    ax.set_xlabel("Mean absolute SHAP")
    style_ax(ax, "Return-Model Interpretation", "x")

    add_note(
        fig.add_subplot(gs[1, 1]),
        [
            "Official Reg SHO pilot evidence is included as a short-sale mechanism diagnostic.",
            "FRED, BLS, BEA, EIA, and SEC EDGAR controls are included with a one-month availability lag.",
            "TreeSHAP links the return model to rates, volatility, financing, oil, and factor-state variables.",
            "Cost-aware returns accompany pricing-kernel diagnostics.",
        ],
        "Empirical Interpretation",
    )
    return save_figure(fig, root, "paper_figure_5_mechanisms_interpretation")


def load_context(root: Path) -> dict:
    reports = root / "outputs" / "reports"
    manifests = root / "manifests"
    return {
        "extraction": read_json(manifests / "run_full" / "summary.json"),
        "ssvi": read_json(manifests / "ssvi_fit" / "summary.json"),
        "chars": read_json(manifests / "characteristic_library_manifest.json"),
        "sdf": read_json(manifests / "conditional_autoencoder_sdf_manifest.json"),
        "gpu": read_json(manifests / "gpu_return_model_manifest.json"),
        "ssvi_monthly": pd.read_csv(reports / "surfaces" / "ssvi_monthly_summary.csv"),
        "sdf_monthly": pd.read_csv(reports / "sdf" / "conditional_autoencoder_sdf_monthly.csv"),
        "integrated_gradients": pd.read_csv(reports / "interpretation" / "top_integrated_gradients.csv"),
        "shap": pd.read_csv(reports / "interpretation" / "tree_shap_feature_importance.csv"),
        "proposal": pd.read_csv(reports / "backtests" / "proposal_buffered_portfolio.csv"),
        "proposal_summary": pd.read_csv(reports / "backtests" / "proposal_buffered_summary.csv"),
        "ff5_alphas": pd.read_csv(reports / "inference" / "ff5_proposal_alphas.csv"),
        "regsho": pd.read_csv(reports / "regsho" / "regsho_pilot_did_coefficients.csv"),
        "external": pd.read_csv(reports / "external" / "monthly_external_api_coverage.csv"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-root-guard", action="store_true")
    args = parser.parse_args()

    root = Path.cwd() if args.no_root_guard else assert_approved_root(Path.cwd())
    ensure_project_dirs(root)
    ctx = load_context(root)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
        }
    )
    artifact_paths = []
    for builder in [figure_1, figure_2, figure_3, figure_4, figure_5]:
        artifact_paths.extend(builder(root, ctx))

    manifest = {
        "status": "PASS",
        "figures": artifact_paths,
        "figure_count": 5,
        "source": "non_raw_reports_and_manifests",
    }
    manifest_path = root / "manifests" / "paper_figure_package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("paper_figure_package_status=PASS")
    for path in artifact_paths:
        print(path)
    print(manifest_path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
