from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from surface_returns.paths import assert_approved_root, ensure_project_dirs


@dataclass(frozen=True)
class MetricCard:
    label: str
    value: str
    detail: str
    tone: str = "neutral"


@dataclass(frozen=True)
class FigureSpec:
    title: str
    file_name: str
    caption: str
    group: str


FIGURES = [
    FigureSpec(
        "Visual Abstract",
        "visual_abstract.png",
        "Presentation-ready 16:9 summary of the option-surface evidence stack and empirical interpretation.",
        "overview",
    ),
    FigureSpec(
        "Paper Figure 1: Pipeline",
        "paper_figure_1_pipeline.png",
        "Numbered figure summarizing the WRDS-to-model pipeline, sample coverage, and empirical scale.",
        "paper",
    ),
    FigureSpec(
        "Paper Figure 2: Surface Quality",
        "paper_figure_2_surface_quality.png",
        "Numbered figure showing SSVI no-arbitrage pass share, fit error, theta projection, and surface constraints.",
        "paper",
    ),
    FigureSpec(
        "Paper Figure 3: SDF Interpretation",
        "paper_figure_3_sdf_interpretation.png",
        "Numbered figure tying the flagship conditional autoencoder SDF to integrated gradients and TreeSHAP evidence.",
        "paper",
    ),
    FigureSpec(
        "Paper Figure 4: Return Evidence",
        "paper_figure_4_return_evidence.png",
        "Numbered figure presenting cost-aware portfolio wealth, Sharpe ratios, FF5+UMD alphas, and trading frictions.",
        "paper",
    ),
    FigureSpec(
        "Paper Figure 5: Mechanisms and Interpretation",
        "paper_figure_5_mechanisms_interpretation.png",
        "Numbered figure for Reg SHO evidence, external controls, SHAP interpretation, and empirical discipline.",
        "paper",
    ),
    FigureSpec(
        "Evidence Pack",
        "visual_evidence_pack.png",
        "One-page view of no-arbitrage diagnostics, SDF evidence, mechanisms, interpretation, and return tests.",
        "overview",
    ),
    FigureSpec(
        "Full Evidence Dashboard",
        "evidence_dashboard.png",
        "Portfolio wealth, factor alphas, SDF errors, external controls, and interpretation panels.",
        "overview",
    ),
    FigureSpec(
        "SSVI No-arbitrage Diagnostics",
        "ssvi_no_arbitrage_diagnostics.png",
        "Full-sample SSVI pass share, calendar diagnostics, and fit-quality evidence.",
        "surface",
    ),
    FigureSpec(
        "Surface Feature Coverage",
        "surface_feature_coverage.png",
        "Month-end option-surface feature coverage across the usable WRDS-backed sample.",
        "surface",
    ),
    FigureSpec(
        "Surface Autoencoder Reconstruction",
        "surface_autoencoder_reconstruction.png",
        "CUDA autoencoder reconstruction diagnostics for fixed maturity-delta surface grids.",
        "model",
    ),
    FigureSpec(
        "GPU Neural OOS Evidence",
        "gpu_neural_oos.png",
        "Walk-forward GPU return-model diagnostics for the expanded-feature predictive stack.",
        "model",
    ),
    FigureSpec(
        "Buffered Portfolio",
        "proposal_buffered_portfolio.png",
        "Sector-balanced, beta-hedged, buffered portfolio construction with fixed and TAQ-calibrated costs.",
        "portfolio",
    ),
    FigureSpec(
        "Asset-pricing Inference",
        "asset_pricing_inference.png",
        "FF3/FF5+UMD alpha, GRS, HJ-style, Newey-West, and bootstrap evidence.",
        "portfolio",
    ),
    FigureSpec(
        "Conditional Autoencoder SDF",
        "conditional_autoencoder_sdf.png",
        "Three-branch conditional autoencoder SDF pricing and reconstruction diagnostics.",
        "sdf",
    ),
    FigureSpec(
        "Conditional SDF Interpretation",
        "conditional_sdf_interpretation.png",
        "Integrated gradients and latent-factor interpretation for the flagship SDF.",
        "sdf",
    ),
    FigureSpec(
        "TreeSHAP Interpretation",
        "tree_shap_interpretation.png",
        "TreeSHAP feature importance for the refreshed full-characteristic predictive stack.",
        "interpretation",
    ),
    FigureSpec(
        "Reg SHO Pilot DID",
        "regsho_pilot_did.png",
        "Official SEC Category A pilot diff-in-diff mechanism test, shown as a short-sale mechanism diagnostic.",
        "mechanism",
    ),
]


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


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def relpath_for_html(source: Path, target: Path) -> str:
    return Path(os.path.relpath(target, source.parent)).as_posix()


def card_html(card: MetricCard) -> str:
    return (
        f'<article class="metric metric-{html.escape(card.tone)}">'
        f"<span>{html.escape(card.label)}</span>"
        f"<strong>{html.escape(card.value)}</strong>"
        f"<p>{html.escape(card.detail)}</p>"
        "</article>"
    )


def figure_html(report_path: Path, figures_dir: Path, spec: FigureSpec) -> str:
    image = figures_dir / spec.file_name
    if not image.exists():
        return ""
    rel = html.escape(relpath_for_html(report_path, image))
    svg = image.with_suffix(".svg")
    svg_link = (
        f'<a href="{html.escape(relpath_for_html(report_path, svg))}">SVG</a>'
        if svg.exists()
        else ""
    )
    return (
        f'<figure class="figure-card" data-group="{html.escape(spec.group)}">'
        f'<a href="{rel}"><img src="{rel}" alt="{html.escape(spec.title)}"></a>'
        "<figcaption>"
        f"<strong>{html.escape(spec.title)}</strong>"
        f"<span>{html.escape(spec.caption)}</span>"
        f'<div class="figure-links"><a href="{rel}">PNG</a>{svg_link}</div>'
        "</figcaption>"
        "</figure>"
    )


def build_cards(manifests: dict[str, dict]) -> list[MetricCard]:
    extraction = manifests["extraction"]
    ssvi = manifests["ssvi"]
    chars = manifests["chars"]
    gpu = manifests["gpu"]
    portfolio = manifests["portfolio"]
    inference = manifests["inference"]
    sdf = manifests["sdf"]
    regsho = manifests["regsho"]
    external = manifests["external"]
    reg_t = (
        regsho.get("regression", {})
        .get("target_coefficient", {})
        .get("t_cluster_date")
    )
    external_sources = ",".join(external.get("control_sources_passed", [])) or "none"
    skipped_external = [
        item.get("source")
        for item in external.get("source_statuses", [])
        if str(item.get("status", "")).startswith("SKIPPED")
    ]
    external_detail = (
        f"Skipped: {', '.join(str(item) for item in skipped_external)}"
        if skipped_external
        else "BEA/EIA/SEC runtime checks loaded with conservative timing"
    )
    return [
        MetricCard(
            "Usable Sample",
            "1996-2024",
            f"{fmt_int(extraction.get('completed'))} completed shards; {fmt_int(extraction.get('skipped'))} skipped",
            "good",
        ),
        MetricCard(
            "SSVI Gates",
            fmt_pct(ssvi.get("pass_share")),
            f"{fmt_int(ssvi.get('surfaces'))} date-security surfaces",
            "good",
        ),
        MetricCard(
            "Characteristics",
            fmt_int(len(chars.get("characteristics", []))),
            f"{fmt_int(chars.get('panel_rows'))} firm-month rows",
            "good",
        ),
        MetricCard(
            "GPU Signal",
            fmt_float(gpu.get("mean_rank_ic")),
            f"top-bottom {fmt_float(gpu.get('mean_top_bottom_return'), 4)} per month",
            "caution",
        ),
        MetricCard(
            "Portfolio Net",
            fmt_pct(portfolio.get("net", {}).get("mean_monthly_return"), 2),
            f"TAQ-net {fmt_pct(portfolio.get('taq_net', {}).get('mean_monthly_return'), 2)} per month",
            "caution",
        ),
        MetricCard(
            "FF5+UMD Alpha",
            fmt_float(inference.get("proposal_net_ff5_alpha_t"), 2),
            "Newey-West t-stat for proposal net long-short alpha",
            "caution",
        ),
        MetricCard(
            "SDF",
            fmt_float(sdf.get("pricing_error", {}).get("rms"), 3),
            f"{fmt_int(sdf.get('oos_months'))} OOS months; three-branch conditional autoencoder",
            "good",
        ),
        MetricCard(
            "Reg SHO Mechanism",
            fmt_float(reg_t, 2),
            "Target triple-interaction t-stat; mechanism diagnostic",
            "mixed",
        ),
        MetricCard(
            "External Controls",
            external_sources,
            external_detail,
            "mixed" if skipped_external else "good",
        ),
    ]


def bullets_html(items: Iterable[str]) -> str:
    return "".join(f"<li>{html.escape(item)}</li>" for item in items)


def render_report(root: Path, report_path: Path) -> str:
    manifests_dir = root / "manifests"
    reports_dir = root / "outputs" / "reports"
    figures_dir = root / "outputs" / "figures" / "full"
    manifests = {
        "extraction": read_json(manifests_dir / "run_full" / "summary.json"),
        "ssvi": read_json(manifests_dir / "ssvi_fit" / "summary.json"),
        "chars": read_json(manifests_dir / "characteristic_library_manifest.json"),
        "gpu": read_json(manifests_dir / "gpu_return_model_manifest.json"),
        "portfolio": read_json(manifests_dir / "proposal_portfolio_manifest.json"),
        "inference": read_json(manifests_dir / "asset_pricing_inference_manifest.json"),
        "sdf": read_json(manifests_dir / "conditional_autoencoder_sdf_manifest.json"),
        "regsho": read_json(manifests_dir / "regsho_pilot_did_manifest.json"),
        "external": read_json(manifests_dir / "external_api_controls_manifest.json"),
    }
    cards = build_cards(manifests)
    figures = "\n".join(figure_html(report_path, figures_dir, spec) for spec in FIGURES)
    deck_path = root / "outputs" / "reports" / "visual_slide_deck.html"
    deck_link = (
        f'<a class="deck-link" href="{html.escape(relpath_for_html(report_path, deck_path))}">Open visual slide deck</a>'
        if deck_path.exists()
        else ""
    )
    pdf_path = root / "outputs" / "reports" / "visual_figure_package.pdf"
    pdf_link = (
        f'<a class="deck-link" href="{html.escape(relpath_for_html(report_path, pdf_path))}">Open PDF package</a>'
        if pdf_path.exists()
        else ""
    )
    artifact_index_path = root / "outputs" / "reports" / "visual_artifact_index.html"
    artifact_index_link = (
        f'<a class="deck-link" href="{html.escape(relpath_for_html(report_path, artifact_index_path))}">Open artifact index</a>'
        if artifact_index_path.exists()
        else ""
    )
    takeaways = [
        "SSVI surfaces pass the no-arbitrage gates across the usable full sample.",
        "The conditional SDF and interpretation layers are the main finished empirical product.",
        "Cost-aware long-short tests discipline the return interpretation.",
    ]

    css = """
:root {
  color-scheme: light;
  --ink: #1f2933;
  --muted: #5b6573;
  --line: #d8dee6;
  --panel: #f6f8fb;
  --good: #2f7d32;
  --mixed: #6f8f51;
  --caution: #c45a42;
  --blue: #356da3;
  --teal: #3d9292;
  --gold: #c77d20;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Aptos", "Segoe UI", Arial, sans-serif;
  color: var(--ink);
  background: #ffffff;
}
header {
  padding: 42px min(6vw, 72px) 26px;
  border-bottom: 1px solid var(--line);
}
.eyebrow {
  color: var(--blue);
  font-weight: 700;
  font-size: 12px;
  letter-spacing: 0;
  text-transform: uppercase;
}
h1 {
  margin: 8px 0 8px;
  font-size: clamp(34px, 4.5vw, 64px);
  line-height: 0.98;
  letter-spacing: 0;
}
.subtitle {
  margin: 0;
  max-width: 980px;
  color: var(--muted);
  font-size: 18px;
  line-height: 1.45;
}
main { padding: 28px min(6vw, 72px) 54px; }
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 12px;
  margin-bottom: 30px;
}
.metric {
  min-height: 132px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  padding: 14px 16px 14px;
  border-top: 5px solid var(--blue);
}
.metric-good { border-top-color: var(--good); }
.metric-mixed { border-top-color: var(--mixed); }
.metric-caution { border-top-color: var(--caution); }
.metric span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
.metric strong {
  display: block;
  margin-top: 8px;
  font-size: 28px;
  line-height: 1.05;
}
.metric p {
  margin: 8px 0 0;
  color: var(--muted);
  line-height: 1.35;
}
.section-head {
  margin: 34px 0 16px;
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: end;
}
h2 { margin: 0; font-size: 25px; letter-spacing: 0; }
.section-head p {
  margin: 0;
  color: var(--muted);
  max-width: 660px;
  line-height: 1.4;
}
.deck-link {
  display: inline-block;
  margin-top: 16px;
  border: 1px solid var(--blue);
  border-radius: 8px;
  padding: 10px 14px;
  font-weight: 800;
  background: #f4f8fc;
}
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 18px;
}
.figure-card {
  margin: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  background: #fff;
}
.figure-card img {
  width: 100%;
  aspect-ratio: 16 / 10;
  object-fit: contain;
  display: block;
  background: #fff;
  border-bottom: 1px solid var(--line);
}
.figure-card figcaption {
  display: grid;
  gap: 7px;
  padding: 13px 14px 14px;
}
.figure-card figcaption span {
  color: var(--muted);
  line-height: 1.35;
}
.figure-links {
  display: flex;
  gap: 10px;
  font-size: 13px;
  font-weight: 700;
}
a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }
.interpretation {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(280px, 0.9fr);
  gap: 18px;
  align-items: stretch;
}
.callout {
  border-left: 5px solid var(--caution);
  background: #fff7f4;
  padding: 18px 20px;
  border-radius: 8px;
}
.callout p { margin: 0; line-height: 1.5; }
.list-panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px 18px;
  background: var(--panel);
}
.list-panel ul {
  margin: 10px 0 0;
  padding-left: 20px;
  line-height: 1.55;
}
footer {
  padding: 18px min(6vw, 72px) 34px;
  color: var(--muted);
  border-top: 1px solid var(--line);
}
@media (max-width: 760px) {
  header { padding-top: 28px; }
  main { padding-top: 22px; }
  .interpretation { grid-template-columns: 1fr; }
  .gallery { grid-template-columns: 1fr; }
  .metric strong { font-size: 24px; }
}
"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Surface-to-Returns Visual Report</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <div class="eyebrow">Research evidence</div>
    <h1>Surface-to-Returns Visual Report</h1>
    <p class="subtitle">A visual summary of the option-surface asset-pricing package: no-arbitrage construction, full characteristic/state integration, GPU and SDF models, cost-aware portfolio evidence, and interpretation.</p>
  </header>
  <main>
    <section class="metrics">
      {''.join(card_html(card) for card in cards)}
    </section>

    <section class="interpretation">
      <div class="callout">
        <p><strong>Current interpretation.</strong> The strongest empirical product is the option-surface state representation: SVI/SSVI diagnostics pass at scale and feed an interpretable conditional SDF. The expanded-feature return and portfolio tests are included as cost-aware diagnostics beside the pricing-kernel evidence.</p>
      </div>
      <div class="list-panel">
        <strong>Research takeaways</strong>
        <ul>{bullets_html(takeaways)}</ul>
        {deck_link}
        {pdf_link}
        {artifact_index_link}
      </div>
    </section>

    <div class="section-head">
      <h2>Figure Gallery</h2>
      <p>Each panel links to the generated PNG and, when available, its SVG companion. All figures are built from curated manifests, reports, and model diagnostics.</p>
    </div>
    <section class="gallery">
      {figures}
    </section>
  </main>
  <footer>
    Generated from curated artifacts under outputs and manifests. Raw WRDS data, logs, credentials, and private API values are excluded.
  </footer>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-root-guard", action="store_true")
    args = parser.parse_args()

    root = Path.cwd() if args.no_root_guard else assert_approved_root(Path.cwd())
    ensure_project_dirs(root)
    report_path = root / "outputs" / "reports" / "visual_report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    html_text = render_report(root, report_path)
    report_path.write_text(html_text, encoding="utf-8")

    manifest = {
        "status": "PASS",
        "report": str(report_path.relative_to(root)),
        "figures_available": [
            spec.file_name
            for spec in FIGURES
            if (root / "outputs" / "figures" / "full" / spec.file_name).exists()
        ],
    }
    manifest_path = root / "manifests" / "visual_report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("visual_report_status=PASS")
    print(report_path.relative_to(root))
    print(manifest_path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
