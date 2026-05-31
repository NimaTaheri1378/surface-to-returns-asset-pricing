from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from surface_returns.paths import assert_approved_root, ensure_project_dirs


@dataclass(frozen=True)
class PdfFigure:
    title: str
    image_name: str
    caption: str


FIGURES = [
    PdfFigure(
        "Visual Abstract",
        "visual_abstract.png",
        "First-glance summary of the full-sample option-surface evidence package.",
    ),
    PdfFigure(
        "Figure 1. Data and Modeling Stack",
        "paper_figure_1_pipeline.png",
        "WRDS extraction, linkage, surface construction, characteristics, costs, inference, and interpretation.",
    ),
    PdfFigure(
        "Figure 2. Surface Quality",
        "paper_figure_2_surface_quality.png",
        "SSVI pass share, fit error, monotone-theta projection, and no-arbitrage constraints.",
    ),
    PdfFigure(
        "Figure 3. SDF and Interpretation",
        "paper_figure_3_sdf_interpretation.png",
        "Three-branch conditional autoencoder SDF, integrated gradients, and TreeSHAP evidence.",
    ),
    PdfFigure(
        "Figure 4. Return Evidence",
        "paper_figure_4_return_evidence.png",
        "Buffered long-short wealth, Sharpe ratios, factor alphas, and trading frictions.",
    ),
    PdfFigure(
        "Figure 5. Mechanisms and Interpretation",
        "paper_figure_5_mechanisms_interpretation.png",
        "Reg SHO mechanism evidence, external controls, SHAP interpretation, and empirical discipline.",
    ),
    PdfFigure(
        "Proposal Evidence Pack",
        "visual_evidence_pack.png",
        "Detailed one-page view of diagnostics, model evidence, mechanisms, and interpretation.",
    ),
]


PALETTE = {
    "ink": "#1f2933",
    "muted": "#5b6573",
    "line": "#d8dee6",
    "panel": "#f6f8fb",
    "blue": "#356da3",
    "green": "#2f7d32",
    "red": "#c45a42",
    "purple": "#8f4f9f",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def fmt_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_pct(value: object, digits: int = 1) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def available_figures(figures_dir: Path, specs: list[PdfFigure] = FIGURES) -> list[PdfFigure]:
    return [spec for spec in specs if (figures_dir / spec.image_name).exists()]


def new_page() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax


def draw_title_page(root: Path) -> plt.Figure:
    manifests = root / "manifests"
    extraction = read_json(manifests / "run_full" / "summary.json")
    ssvi = read_json(manifests / "ssvi_fit" / "summary.json")
    chars = read_json(manifests / "characteristic_library_manifest.json")
    sdf = read_json(manifests / "conditional_autoencoder_sdf_manifest.json")
    portfolio = read_json(manifests / "proposal_portfolio_manifest.json")
    fig, ax = new_page()
    ax.text(0.055, 0.79, "Surface-to-Returns", fontsize=62, weight="bold", color=PALETTE["ink"], transform=ax.transAxes)
    ax.text(0.058, 0.705, "Visual Figure Package", fontsize=28, weight="bold", color=PALETTE["blue"], transform=ax.transAxes)
    ax.text(
        0.058,
        0.625,
        "Figures generated from curated reports, manifests, and model diagnostics.",
        fontsize=18,
        color=PALETTE["muted"],
        transform=ax.transAxes,
    )
    metric_rows = [
        ("Sample", "1996-2024", f"{fmt_int(extraction.get('completed'))} completed shards"),
        ("SSVI gates", fmt_pct(ssvi.get("pass_share")), f"{fmt_int(ssvi.get('surfaces'))} surfaces"),
        ("Characteristics", fmt_int(len(chars.get("characteristics", []))), f"{fmt_int(chars.get('panel_rows'))} rows"),
        ("SDF", f"{float(sdf.get('pricing_error', {}).get('rms', 0.0)):.3f}", f"{fmt_int(sdf.get('oos_months'))} OOS months"),
        ("Return test", fmt_pct(portfolio.get("net", {}).get("mean_monthly_return"), 2), f"TAQ-net {fmt_pct(portfolio.get('taq_net', {}).get('mean_monthly_return'), 2)}/mo"),
    ]
    x0, y0 = 0.058, 0.34
    for idx, (label, value, detail) in enumerate(metric_rows):
        x = x0 + idx * 0.18
        ax.add_patch(
            plt.Rectangle(
                (x, y0),
                0.155,
                0.16,
                transform=ax.transAxes,
                facecolor=PALETTE["panel"],
                edgecolor=PALETTE["line"],
                linewidth=1,
            )
        )
        ax.text(x + 0.014, y0 + 0.116, label.upper(), transform=ax.transAxes, fontsize=9, color=PALETTE["muted"], weight="bold")
        ax.text(x + 0.014, y0 + 0.065, value, transform=ax.transAxes, fontsize=22, color=PALETTE["ink"], weight="bold")
        ax.text(x + 0.014, y0 + 0.028, detail, transform=ax.transAxes, fontsize=10.5, color=PALETTE["muted"])
    ax.text(
        0.058,
        0.145,
        "Interpretation: option surfaces are a useful state representation; cost-aware return tests discipline the alpha evidence.",
        fontsize=14,
        color=PALETTE["blue"],
        transform=ax.transAxes,
        weight="bold",
    )
    return fig


def draw_image_page(root: Path, spec: PdfFigure) -> plt.Figure:
    image_path = root / "outputs" / "figures" / "full" / spec.image_name
    image = plt.imread(image_path)
    fig, ax = new_page()
    ax.text(0.047, 0.94, spec.title, fontsize=24, weight="bold", color=PALETTE["ink"], transform=ax.transAxes)
    ax.text(0.048, 0.905, spec.caption, fontsize=12.5, color=PALETTE["muted"], transform=ax.transAxes)
    image_ax = fig.add_axes([0.045, 0.075, 0.91, 0.79])
    image_ax.imshow(image)
    image_ax.axis("off")
    return fig


def draw_closing_page(root: Path) -> plt.Figure:
    fig, ax = new_page()
    ax.text(0.055, 0.82, "Research Interpretation", fontsize=42, weight="bold", color=PALETTE["ink"], transform=ax.transAxes)
    ax.text(
        0.056,
        0.755,
        "The package connects option-surface states, pricing-kernel diagnostics, mechanism tests, and cost-aware returns.",
        fontsize=17,
        color=PALETTE["muted"],
        transform=ax.transAxes,
    )
    left = [
        "Scaled WRDS-backed extraction and linkage through the usable 1996-2024 sample.",
        "SVI and SSVI no-arbitrage diagnostics pass at full usable scale.",
        "Characteristics, state controls, TAQ costs, IBES, Reg SHO, SDF, SHAP, and inference artifacts exist.",
        "Figures, report, deck, and PDF are generated from curated outputs and manifests.",
    ]
    right = [
        "Option surfaces work best here as state variables for pricing-kernel and mechanism analysis.",
        "TreeSHAP and integrated gradients connect the models to volatility, rates, financing, and factor-state variables.",
        "Cost-aware long-short tests are reported as diagnostics beside the pricing-kernel evidence.",
    ]
    for x, title, items, color in [
        (0.06, "Core Evidence", left, PALETTE["green"]),
        (0.53, "Empirical Read", right, PALETTE["blue"]),
    ]:
        ax.add_patch(
            plt.Rectangle(
                (x, 0.19),
                0.39,
                0.45,
                transform=ax.transAxes,
                facecolor=PALETTE["panel"] if color == PALETTE["green"] else "#fff7f4",
                edgecolor=PALETTE["line"],
                linewidth=1,
            )
        )
        ax.text(x + 0.025, 0.58, title, fontsize=19, weight="bold", color=color, transform=ax.transAxes)
        y = 0.51
        for item in items:
            ax.text(x + 0.033, y, f"- {item}", fontsize=12.2, color=PALETTE["ink"], transform=ax.transAxes, va="top", wrap=True)
            y -= 0.075
    return fig


def build_pdf(root: Path, output_path: Path) -> list[str]:
    figures_dir = root / "outputs" / "figures" / "full"
    specs = available_figures(figures_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for fig in [draw_title_page(root), *[draw_image_page(root, spec) for spec in specs], draw_closing_page(root)]:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return [spec.image_name for spec in specs]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-root-guard", action="store_true")
    args = parser.parse_args()

    root = Path.cwd() if args.no_root_guard else assert_approved_root(Path.cwd())
    ensure_project_dirs(root)
    output_path = root / "outputs" / "reports" / "visual_figure_package.pdf"
    figures = build_pdf(root, output_path)
    manifest = {
        "status": "PASS",
        "pdf": str(output_path.relative_to(root)),
        "figures": figures,
        "pages": len(figures) + 2,
        "source": "non_raw_reports_and_figures",
    }
    manifest_path = root / "manifests" / "visual_pdf_package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("visual_pdf_package_status=PASS")
    print(output_path.relative_to(root))
    print(manifest_path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
