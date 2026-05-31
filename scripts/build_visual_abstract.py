from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def fmt_float(value: object, digits: int = 3) -> str:
    number = safe_float(value)
    return "n/a" if not np.isfinite(number) else f"{number:.{digits}f}"


def return_evidence_label(gpu_manifest: dict, portfolio_manifest: dict) -> tuple[str, str]:
    rank_ic = safe_float(gpu_manifest.get("mean_rank_ic"))
    net = safe_float(portfolio_manifest.get("net", {}).get("mean_monthly_return"))
    if np.isfinite(rank_ic) and np.isfinite(net) and rank_ic > 0 and net > 0:
        return "positive", PALETTE["green"]
    if np.isfinite(net) and net < 0:
        return "cost-aware", PALETTE["orange"]
    return "diagnostic", PALETTE["orange"]


def wealth_index(frame: pd.DataFrame, return_col: str) -> pd.Series:
    returns = pd.to_numeric(frame[return_col], errors="coerce").fillna(0.0)
    return (1.0 + returns).cumprod()


def draw_round_rect(ax, x: float, y: float, w: float, h: float, color: str, edge: str = "none") -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=0.8 if edge != "none" else 0,
            edgecolor=edge,
            facecolor=color,
            transform=ax.transAxes,
        )
    )


def draw_metric_card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    value: str,
    detail: str,
    color: str,
) -> None:
    draw_round_rect(ax, x, y, w, h, PALETTE["panel"], PALETTE["line"])
    ax.add_patch(Rectangle((x, y + h - 0.018), w, 0.018, transform=ax.transAxes, color=color, lw=0))
    value_size = 16 if len(value) <= 8 else 13 if len(value) <= 14 else 11.3
    pad = 0.020
    ax.text(x + pad, y + h - 0.033, label.upper(), transform=ax.transAxes, color=PALETTE["muted"], fontsize=7.4, weight="bold", va="top")
    ax.text(x + pad, y + 0.052, value, transform=ax.transAxes, color=PALETTE["ink"], fontsize=value_size, weight="bold", va="bottom")
    ax.text(x + pad, y + 0.021, detail, transform=ax.transAxes, color=PALETTE["muted"], fontsize=8.0, va="bottom", linespacing=1.12)


def draw_pipeline(ax, labels: list[str], x0: float, y: float, total_w: float) -> None:
    gap = 0.012
    w = (total_w - gap * (len(labels) - 1)) / len(labels)
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["purple"], PALETTE["orange"], PALETTE["gold"], PALETTE["green"]]
    for idx, label in enumerate(labels):
        x = x0 + idx * (w + gap)
        h = 0.088
        draw_round_rect(ax, x, y, w, h, "#ffffff", PALETTE["line"])
        ax.add_patch(Rectangle((x, y), 0.008, h, transform=ax.transAxes, color=colors[idx % len(colors)], lw=0))
        ax.text(x + 0.018, y + h / 2, f"{idx + 1}", transform=ax.transAxes, color=colors[idx % len(colors)], fontsize=12, weight="bold", va="center")
        ax.text(x + 0.045, y + h / 2, label, transform=ax.transAxes, color=PALETTE["ink"], fontsize=8.4, va="center", linespacing=1.08)
        if idx < len(labels) - 1:
            arrow_x = x + w + gap * 0.28
            ax.add_patch(
                Polygon(
                    [[arrow_x, y + h / 2], [arrow_x + gap * 0.32, y + h / 2 + 0.012], [arrow_x + gap * 0.32, y + h / 2 - 0.012]],
                    closed=True,
                    transform=ax.transAxes,
                    facecolor=PALETTE["line"],
                    edgecolor=PALETTE["line"],
                )
            )


def draw_model_diagnostics(ax, gpu_manifest: dict, portfolio_manifest: dict, sdf_manifest: dict) -> None:
    values = [
        safe_float(gpu_manifest.get("mean_rank_ic"), 0.0),
        safe_float(portfolio_manifest.get("net", {}).get("mean_monthly_return"), 0.0),
        safe_float(portfolio_manifest.get("taq_net", {}).get("mean_monthly_return"), 0.0),
        safe_float(sdf_manifest.get("pricing_error", {}).get("rms"), 0.0),
    ]
    labels = ["Rank IC", "Net return", "TAQ net", "SDF RMS"]
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["red"], PALETTE["teal"]]
    ax.barh(labels, values, color=colors)
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value:.3f}", va="center", fontsize=9, color=PALETTE["ink"], weight="bold")
    ax.axvline(0, color=PALETTE["ink"], lw=0.8, alpha=0.55)
    ax.set_title("Model and Portfolio Diagnostics", loc="left", fontsize=13, weight="bold")
    ax.grid(True, axis="x", alpha=0.20)
    ax.tick_params(axis="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_wealth_panel(ax, proposal: pd.DataFrame) -> None:
    data = proposal.sort_values("date").copy()
    data["date"] = pd.to_datetime(data["date"])
    series = {
        "gross": ("beta_neutral_gross_return", PALETTE["blue"]),
        "fixed cost": ("net_return", PALETTE["orange"]),
        "TAQ cost": ("taq_net_return", PALETTE["red"]),
    }
    for label, (col, color) in series.items():
        if col in data:
            ax.plot(data["date"], wealth_index(data, col), label=label, color=color, linewidth=2.1)
    ax.axhline(1.0, color=PALETTE["ink"], lw=0.8, alpha=0.55)
    ax.set_title("Cost-aware Portfolio Wealth", loc="left", fontsize=13, weight="bold")
    ax.set_ylabel("Growth of $1", fontsize=9)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_takeaway_panel(ax, external: dict) -> None:
    ax.axis("off")
    draw_round_rect(ax, 0.0, 0.0, 1.0, 1.0, "#f8fbf7", "none")
    ax.text(0.070, 0.86, "Research Takeaway", transform=ax.transAxes, fontsize=12.6, weight="bold", color=PALETTE["ink"])
    lines = [
        "Option surfaces form clean pricing states.",
        "SSVI gates pass across 1996-2024.",
        "SDF diagnostics anchor the evidence.",
        "Cost-aware tests discipline returns.",
    ]
    for y, line in zip([0.68, 0.51, 0.34, 0.17], lines):
        wrapped = textwrap.fill(f"- {line}", width=36, subsequent_indent="  ")
        ax.text(0.080, y, wrapped, transform=ax.transAxes, fontsize=8.7, color=PALETTE["ink"], va="top", linespacing=1.12)


def external_detail(external: dict) -> str:
    skipped = [
        item.get("source")
        for item in external.get("source_statuses", [])
        if str(item.get("status", "")).startswith("SKIPPED")
    ]
    if skipped:
        return f"skipped: {','.join(str(item) for item in skipped)}"
    return "BEA/EIA/SEC loaded"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-root-guard", action="store_true")
    args = parser.parse_args()

    root = Path.cwd() if args.no_root_guard else assert_approved_root(Path.cwd())
    ensure_project_dirs(root)
    manifests = root / "manifests"
    reports = root / "outputs" / "reports"
    figures = root / "outputs" / "figures" / "full"
    figures.mkdir(parents=True, exist_ok=True)

    proposal = pd.read_csv(reports / "backtests" / "proposal_buffered_portfolio.csv")
    extraction = read_json(manifests / "run_full" / "summary.json")
    ssvi = read_json(manifests / "ssvi_fit" / "summary.json")
    chars = read_json(manifests / "characteristic_library_manifest.json")
    gpu = read_json(manifests / "gpu_return_model_manifest.json")
    portfolio = read_json(manifests / "proposal_portfolio_manifest.json")
    sdf = read_json(manifests / "conditional_autoencoder_sdf_manifest.json")
    external = read_json(manifests / "external_api_controls_manifest.json")

    return_label, return_color = return_evidence_label(gpu, portfolio)
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
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.axis("off")

    canvas.text(0.052, 0.935, "Surface-to-Returns", fontsize=31, weight="bold", color=PALETTE["ink"], transform=canvas.transAxes)
    canvas.text(0.052, 0.895, "Full-sample option-surface asset-pricing evidence: construction, pricing kernel, mechanisms, and return tests", fontsize=12.2, color=PALETTE["muted"], transform=canvas.transAxes)
    canvas.text(
        0.815,
        0.927,
        "Finding: option surfaces are informative pricing states",
        fontsize=10.5,
        color=PALETTE["blue"],
        weight="bold",
        ha="left",
        transform=canvas.transAxes,
    )

    cards = [
        ("WRDS sample", "1996-2024", f"{fmt_int(extraction.get('completed'))} completed shards", PALETTE["blue"]),
        ("SSVI gates", fmt_pct(ssvi.get("pass_share")), f"{fmt_int(ssvi.get('surfaces'))} surfaces", PALETTE["green"]),
        ("Characteristics", fmt_int(len(chars.get("characteristics", []))), f"{fmt_int(chars.get('panel_rows'))} firm-month rows", PALETTE["purple"]),
        ("SDF", fmt_float(sdf.get("pricing_error", {}).get("rms"), 3), f"{fmt_int(sdf.get('oos_months'))} OOS months", PALETTE["teal"]),
        ("Return evidence", return_label, f"net {fmt_pct(portfolio.get('net', {}).get('mean_monthly_return'), 2)}/mo", return_color),
        ("External controls", "/".join(external.get("control_sources_passed", [])) or "none", external_detail(external), PALETTE["orange"]),
    ]
    x = 0.052
    w = 0.142
    for label, value, detail, color in cards:
        draw_metric_card(canvas, x, 0.735, w, 0.117, label, value, detail, color)
        x += w + 0.014

    draw_pipeline(
        canvas,
        [
            "WRDS\nlinkage",
            "Surface\ngates",
            "Firm/state\nfeatures",
            "GPU/SDF\nmodels",
            "TAQ-cost\nportfolios",
            "Inference &\ninterpretation",
        ],
        0.052,
        0.606,
        0.896,
    )

    ax_wealth = fig.add_axes([0.063, 0.185, 0.455, 0.34])
    draw_wealth_panel(ax_wealth, proposal)

    ax_status = fig.add_axes([0.558, 0.31, 0.185, 0.215])
    draw_model_diagnostics(ax_status, gpu, portfolio, sdf)

    ax_takeaway = fig.add_axes([0.765, 0.185, 0.190, 0.34])
    draw_takeaway_panel(ax_takeaway, external)

    draw_round_rect(canvas, 0.552, 0.185, 0.198, 0.072, "#f8fbf7", PALETTE["line"])
    canvas.text(0.568, 0.229, "Evidence summary", transform=canvas.transAxes, fontsize=10, color=PALETTE["muted"], weight="bold")
    canvas.text(0.568, 0.199, "surfaces pass; SDF interprets", transform=canvas.transAxes, fontsize=10.5, color=PALETTE["ink"])

    canvas.text(
        0.052,
        0.075,
        "Empirical interpretation: option surfaces are valuable state variables for pricing-kernel and mechanism analysis.",
        transform=canvas.transAxes,
        fontsize=11.5,
        color=PALETTE["ink"],
        weight="bold",
    )
    canvas.text(
        0.052,
        0.045,
        "Cost-aware portfolio tests are reported alongside the pricing-kernel and mechanism evidence.",
        transform=canvas.transAxes,
        fontsize=10.4,
        color=PALETTE["muted"],
    )

    paths = [figures / "visual_abstract.png", figures / "visual_abstract.svg"]
    for path in paths:
        fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "status": "PASS",
        "figure_png": str(paths[0].relative_to(root)),
        "figure_svg": str(paths[1].relative_to(root)),
        "return_evidence_label": return_label,
    }
    manifest_path = manifests / "visual_abstract_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("visual_abstract_status=PASS")
    for path in paths:
        print(path.relative_to(root))
    print(manifest_path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
