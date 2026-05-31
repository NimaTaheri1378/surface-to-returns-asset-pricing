from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from surface_returns.paths import assert_approved_root, ensure_project_dirs


@dataclass(frozen=True)
class SlideSpec:
    title: str
    subtitle: str
    image_name: str | None = None
    accent: str = "blue"


PALETTE = {
    "blue": "#356da3",
    "green": "#2f7d32",
    "teal": "#3d9292",
    "orange": "#c77d20",
    "red": "#c45a42",
    "purple": "#8f4f9f",
    "ink": "#1f2933",
    "muted": "#5b6573",
    "line": "#d8dee6",
    "panel": "#f6f8fb",
}


SLIDES = [
    SlideSpec(
        "Visual Abstract",
        "First-glance summary of the scaled implementation, no-arbitrage success, and honest return boundary.",
        "visual_abstract.png",
        "blue",
    ),
    SlideSpec(
        "Figure 1. Data and Modeling Stack",
        "WRDS extraction, linkage, surface construction, characteristics, costs, inference, and interpretation.",
        "paper_figure_1_pipeline.png",
        "blue",
    ),
    SlideSpec(
        "Figure 2. Surface Quality",
        "SSVI pass share, fit error, monotone-theta projection, and no-arbitrage constraints.",
        "paper_figure_2_surface_quality.png",
        "green",
    ),
    SlideSpec(
        "Figure 3. SDF and Interpretation",
        "Three-branch conditional autoencoder SDF, integrated gradients, and TreeSHAP evidence.",
        "paper_figure_3_sdf_interpretation.png",
        "purple",
    ),
    SlideSpec(
        "Figure 4. Return Evidence",
        "Buffered long-short wealth, Sharpe ratios, factor alphas, and trading frictions.",
        "paper_figure_4_return_evidence.png",
        "red",
    ),
    SlideSpec(
        "Figure 5. Mechanisms and Readiness",
        "Reg SHO mechanism evidence, external controls, non-pass readiness items, and interpretation discipline.",
        "paper_figure_5_mechanisms_readiness.png",
        "orange",
    ),
    SlideSpec(
        "Evidence Pack",
        "Detailed one-page view of implementation status, diagnostics, model evidence, and boundaries.",
        "visual_evidence_pack.png",
        "teal",
    ),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def relpath_for_html(source: Path, target: Path) -> str:
    return Path(os.path.relpath(target, source.parent)).as_posix()


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


def status_counts(readiness: pd.DataFrame) -> dict[str, int]:
    if readiness.empty or "status" not in readiness:
        return {}
    return readiness["status"].value_counts().to_dict()


def metric_tile(label: str, value: str, detail: str, accent: str) -> str:
    color = PALETTE.get(accent, PALETTE["blue"])
    return (
        f'<article class="metric" style="--accent:{html.escape(color)}">'
        f"<span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        f"<p>{html.escape(detail)}</p>"
        "</article>"
    )


def image_slide(deck_path: Path, figures_dir: Path, spec: SlideSpec, number: int) -> str:
    if spec.image_name is None:
        return ""
    image = figures_dir / spec.image_name
    if not image.exists():
        return ""
    rel = html.escape(relpath_for_html(deck_path, image))
    accent = html.escape(PALETTE.get(spec.accent, PALETTE["blue"]))
    return f"""
<section class="slide image-slide" style="--accent:{accent}">
  <div class="slide-head">
    <span class="kicker">Surface-to-Returns visual deck</span>
    <span class="page">{number:02d}</span>
  </div>
  <div class="slide-title">
    <h2>{html.escape(spec.title)}</h2>
    <p>{html.escape(spec.subtitle)}</p>
  </div>
  <figure>
    <img src="{rel}" alt="{html.escape(spec.title)}">
  </figure>
</section>
"""


def title_slide(cards: list[str], blocked_note: str) -> str:
    return f"""
<section class="slide title-slide">
  <div class="title-copy">
    <span class="kicker">Proposal evidence deck</span>
    <h1>Surface-to-Returns</h1>
    <p class="subtitle">A public-safe visual package for the verified option-surface asset-pricing implementation.</p>
    <p class="boundary">{html.escape(blocked_note)}</p>
  </div>
  <div class="metric-grid">
    {''.join(cards)}
  </div>
</section>
"""


def closing_slide(nonpass: list[str]) -> str:
    items = "".join(f"<li>{html.escape(item)}</li>" for item in nonpass)
    return f"""
<section class="slide closing-slide">
  <span class="kicker">Evidence discipline</span>
  <h2>What The Visual Package Proves</h2>
  <div class="two-col">
    <div class="proof">
      <h3>Implemented and verified</h3>
      <ul>
        <li>Scaled WRDS extraction through the usable 1996-2024 optionable sample.</li>
        <li>SVI and SSVI no-arbitrage diagnostics pass at full usable scale.</li>
        <li>CRSP-Compustat characteristics, state controls, TAQ costs, IBES, Reg SHO, SDF, SHAP, and inference artifacts exist.</li>
        <li>Visual report, abstract, evidence pack, and numbered figure package are generated from non-raw outputs.</li>
      </ul>
    </div>
    <div class="limits">
      <h3>Not overclaimed</h3>
      <ul>{items}</ul>
    </div>
  </div>
</section>
"""


def render_deck(root: Path, deck_path: Path) -> tuple[str, list[str]]:
    manifests = root / "manifests"
    reports = root / "outputs" / "reports"
    figures = root / "outputs" / "figures" / "full"
    readiness = read_csv(reports / "readiness" / "proposal_readiness_audit.csv")
    extraction = read_json(manifests / "run_full" / "summary.json")
    ssvi = read_json(manifests / "ssvi_fit" / "summary.json")
    chars = read_json(manifests / "characteristic_library_manifest.json")
    sdf = read_json(manifests / "conditional_autoencoder_sdf_manifest.json")
    counts = status_counts(readiness)
    nonpass = [
        f"{row.requirement}: {row.status.replace('_', ' ')}"
        for row in readiness[readiness["status"].ne("PASS")].itertuples(index=False)
    ]
    if not nonpass:
        nonpass = ["No non-pass readiness items in the current audit."]

    cards = [
        metric_tile("Sample", "1996-2024", f"{fmt_int(extraction.get('completed'))} completed shards", "blue"),
        metric_tile("SSVI Gates", fmt_pct(ssvi.get("pass_share")), f"{fmt_int(ssvi.get('surfaces'))} surfaces", "green"),
        metric_tile("Characteristics", fmt_int(len(chars.get("characteristics", []))), f"{fmt_int(chars.get('panel_rows'))} firm-month rows", "purple"),
        metric_tile("SDF", f"{float(sdf.get('pricing_error', {}).get('rms', 0.0)):.3f}", f"{fmt_int(sdf.get('oos_months'))} OOS months", "teal"),
        metric_tile("Readiness", f"{counts.get('PASS', 0)} pass", f"{counts.get('NEGATIVE_RESULT', 0)} negative, {counts.get('BLOCKED', 0)} blocked", "orange"),
    ]
    blocked_note = "Current boundary: return evidence is weak/negative; external APIs are loaded only from safe runtime env vars."
    slides = [title_slide(cards, blocked_note)]
    available = []
    for idx, spec in enumerate(SLIDES, start=2):
        html_slide = image_slide(deck_path, figures, spec, idx)
        if html_slide:
            slides.append(html_slide)
            available.append(spec.image_name or "")
    slides.append(closing_slide(nonpass))

    css = """
:root {
  color-scheme: light;
  --ink: #1f2933;
  --muted: #5b6573;
  --line: #d8dee6;
  --panel: #f6f8fb;
  --accent: #356da3;
}
* { box-sizing: border-box; }
html { scroll-snap-type: y mandatory; }
html, body { overflow-x: hidden; }
body {
  margin: 0;
  background: #ffffff;
  color: var(--ink);
  font-family: "Aptos", "Segoe UI", Arial, sans-serif;
}
.slide {
  width: 100%;
  min-height: 100vh;
  scroll-snap-align: start;
  padding: 4.5vh 5.2vw;
  display: grid;
  align-content: start;
  gap: 2.4vh;
  page-break-after: always;
}
.kicker {
  color: var(--accent);
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0;
  text-transform: uppercase;
}
.slide-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.page {
  color: var(--muted);
  font-weight: 700;
}
h1, h2, h3, p { margin: 0; letter-spacing: 0; }
h1 {
  margin-top: 1.2vh;
  font-size: clamp(58px, 8vw, 118px);
  line-height: 0.92;
}
h2 {
  font-size: clamp(30px, 4vw, 58px);
  line-height: 1.02;
}
h3 {
  font-size: 24px;
  margin-bottom: 14px;
}
.subtitle {
  margin-top: 2.4vh;
  max-width: 900px;
  font-size: 24px;
  line-height: 1.38;
  color: var(--muted);
}
.boundary {
  margin-top: 3vh;
  max-width: 820px;
  padding: 16px 20px;
  border-left: 6px solid #c45a42;
  background: #fff7f4;
  border-radius: 8px;
  font-size: 18px;
  line-height: 1.4;
}
.title-slide {
  grid-template-columns: minmax(0, 1.1fr) minmax(420px, 0.9fr);
  align-items: center;
}
.metric-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 14px;
}
.metric {
  min-height: 112px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  padding: 15px 18px;
  border-top: 7px solid var(--accent);
}
.metric span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}
.metric strong {
  display: block;
  margin-top: 8px;
  font-size: 31px;
}
.metric p {
  margin-top: 6px;
  color: var(--muted);
  font-size: 16px;
}
.slide-title {
  display: grid;
  grid-template-columns: minmax(0, 0.72fr) minmax(360px, 0.28fr);
  gap: 28px;
  align-items: end;
}
.slide-title p {
  color: var(--muted);
  font-size: 18px;
  line-height: 1.35;
}
figure {
  margin: 0;
  width: 100%;
  height: 73vh;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  display: grid;
  place-items: center;
  overflow: hidden;
}
figure img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.closing-slide {
  align-content: center;
}
.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-top: 3vh;
}
.proof, .limits {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 28px 32px;
  background: var(--panel);
}
.limits {
  background: #fff7f4;
  border-color: #f2d5cc;
}
li {
  margin-bottom: 15px;
  font-size: 20px;
  line-height: 1.38;
}
@media print {
  @page { size: 16in 9in; margin: 0; }
  html { scroll-snap-type: none; }
  .slide { width: 16in; min-height: 9in; padding: 0.42in 0.55in; }
  figure { height: 6.25in; }
  h1 { font-size: 76pt; }
  h2 { font-size: 34pt; }
}
@media (max-width: 880px) {
  .title-slide, .slide-title, .two-col { grid-template-columns: 1fr; }
  .slide { padding: 26px 22px; }
  figure { height: 58vh; }
}
"""
    body = "\n".join(slides)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Surface-to-Returns Visual Deck</title>
  <style>{css}</style>
</head>
<body>
  {body}
</body>
</html>
""", available


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-root-guard", action="store_true")
    args = parser.parse_args()

    root = Path.cwd() if args.no_root_guard else assert_approved_root(Path.cwd())
    ensure_project_dirs(root)
    deck_path = root / "outputs" / "reports" / "visual_slide_deck.html"
    deck_path.parent.mkdir(parents=True, exist_ok=True)
    html_text, available = render_deck(root, deck_path)
    deck_path.write_text(html_text, encoding="utf-8")

    manifest = {
        "status": "PASS",
        "deck": str(deck_path.relative_to(root)),
        "slides": len(available) + 2,
        "figures_available": available,
    }
    manifest_path = root / "manifests" / "visual_slide_deck_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("visual_slide_deck_status=PASS")
    print(deck_path.relative_to(root))
    print(manifest_path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
