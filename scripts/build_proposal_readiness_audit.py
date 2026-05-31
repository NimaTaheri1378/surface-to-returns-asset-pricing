from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, ensure_project_dirs
from surface_returns.readiness import evaluate_readiness, readiness_summary


STATUS_COLORS = {
    "PASS": "#2f7d32",
    "PASS_WEAK_RESULT": "#7aa95c",
    "NEGATIVE_RESULT": "#c45a42",
    "PARTIAL": "#d6a23a",
    "BLOCKED": "#8f4f9f",
    "MISSING": "#8a8a8a",
}


def write_readiness_figure(frame: pd.DataFrame, output_prefix: Path) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot = frame.sort_values(["score", "requirement"], ascending=[True, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13.5, 9.5))
    colors = [STATUS_COLORS.get(status, "#8a8a8a") for status in plot["status"]]
    ax.barh(plot["requirement"], plot["score"], color=colors, height=0.72)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Readiness score")
    ax.set_title("Proposal Readiness Matrix")
    ax.grid(True, axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    for idx, row in plot.iterrows():
        label = str(row["status"]).replace("_", " ")
        score = float(row["score"])
        if score >= 0.98:
            ax.text(0.985, idx, label, va="center", ha="right", fontsize=8.5, color="white")
        else:
            ax.text(min(score + 0.02, 1.08), idx, label, va="center", ha="left", fontsize=8.5)
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=9, color=color, label=status.replace("_", " "))
        for status, color in STATUS_COLORS.items()
        if status in set(plot["status"])
    ]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="lower right")
    fig.text(
        0.01,
        0.012,
        "Negative-result items are implemented and verified but do not support a positive return-prediction claim.",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=240)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-root-guard", action="store_true", help="Allow tests to run in temporary directories.")
    args = parser.parse_args()

    root = Path.cwd() if args.no_root_guard else assert_approved_root(Path.cwd())
    dirs = ensure_project_dirs(root)
    report_dir = root / "outputs" / "reports" / "readiness"
    figure_dir = root / "outputs" / "figures" / "full"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    items = evaluate_readiness(root)
    frame = pd.DataFrame([row.to_dict() for row in items])
    csv_path = report_dir / "proposal_readiness_audit.csv"
    frame.to_csv(csv_path, index=False)
    figures = write_readiness_figure(frame, figure_dir / "proposal_readiness_matrix")
    summary = readiness_summary(items)
    manifest = {
        "status": summary["overall_status"],
        **summary,
        "artifacts": {
            "csv": str(csv_path.relative_to(root)),
            "figures": [str(path.relative_to(root)) for path in figures],
        },
    }
    write_json_atomic(dirs["manifests"] / "proposal_readiness_audit_manifest.json", manifest)
    print(f"proposal_readiness_status={manifest['status']}")
    print(f"items={manifest['items']} mean_score={manifest['mean_score']:.3f}")
    print("counts=" + ",".join(f"{key}:{value}" for key, value in sorted(manifest["counts"].items())))
    return 0 if manifest["status"] in {"PASS", "PASS_WITH_BOUNDARIES", "INCOMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
