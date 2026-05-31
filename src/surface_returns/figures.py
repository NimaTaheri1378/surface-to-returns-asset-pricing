from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_smoke_surface_figure(options: pd.DataFrame, output_prefix: Path) -> list[Path]:
    if options.empty or "impl_volatility" not in options or "delta" not in options:
        return []
    frame = options.copy()
    frame["impl_volatility"] = pd.to_numeric(frame["impl_volatility"], errors="coerce")
    frame["delta"] = pd.to_numeric(frame["delta"], errors="coerce")
    frame = frame.dropna(subset=["impl_volatility", "delta"])
    if frame.empty:
        return []
    if "secid" in frame:
        secid = frame["secid"].value_counts().index[0]
        frame = frame[frame["secid"].eq(secid)]
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if "cp_flag" in frame:
        for flag, group in frame.groupby(frame["cp_flag"].astype(str).str.upper().str[0]):
            ax.scatter(group["delta"], group["impl_volatility"], s=16, alpha=0.65, label=flag)
        ax.legend(title="Option")
    else:
        ax.scatter(frame["delta"], frame["impl_volatility"], s=16, alpha=0.65)
    ax.set_title("Smoke Month Option Surface Slice")
    ax.set_xlabel("Delta")
    ax.set_ylabel("Implied volatility")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    return paths
