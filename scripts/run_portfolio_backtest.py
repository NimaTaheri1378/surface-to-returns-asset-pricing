from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surface_returns.backtest import (
    long_short_decile_weights,
    long_short_portfolio_returns,
    max_drawdown,
    performance_summary,
)
from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs


def load_predictions(root: Path, path_arg: str) -> pd.DataFrame:
    path = root / path_arg
    if not path.exists():
        raise FileNotFoundError(f"Missing GPU predictions: {path}")
    frame = pd.read_parquet(path)
    required = {"date", "permno", "next_ret", "pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Prediction file is missing required columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_panel_cost_columns(root: Path, path_arg: str) -> pd.DataFrame:
    path = root / path_arg
    if not path.exists():
        return pd.DataFrame()
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"])
    keep = [col for col in ["date", "permno", "median_spread_pct", "market_equity"] if col in panel]
    if {"date", "permno"}.difference(keep):
        return pd.DataFrame()
    panel = panel[keep].copy()
    if "median_spread_pct" in panel:
        panel["half_spread"] = pd.to_numeric(panel["median_spread_pct"], errors="coerce").clip(0, 1.0) / 2.0
    return panel.groupby(["date", "permno"], as_index=False).median(numeric_only=True)


def write_backtest_figure(monthly: pd.DataFrame, output_prefix: Path) -> list[Path]:
    if monthly.empty:
        return []
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    monthly = monthly.sort_values("date").copy()
    monthly["gross_wealth"] = (1.0 + monthly["gross_return"].fillna(0.0)).cumprod()
    monthly["net_wealth"] = (1.0 + monthly["net_return"].fillna(0.0)).cumprod()
    monthly["net_drawdown"] = monthly["net_wealth"] / monthly["net_wealth"].cummax() - 1.0

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.0), sharex=True)
    axes[0].plot(monthly["date"], monthly["gross_wealth"], label="Gross", color="#4c78a8", linewidth=1.7)
    axes[0].plot(monthly["date"], monthly["net_wealth"], label="Net", color="#f58518", linewidth=1.7)
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)

    axes[1].fill_between(
        monthly["date"],
        monthly["net_drawdown"],
        0,
        color="#e45756",
        alpha=0.35,
        linewidth=0,
    )
    axes[1].set_ylabel("Net drawdown")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(monthly["date"], monthly["turnover"], color="#54a24b", linewidth=1.3, label="Turnover")
    axes[2].set_ylabel("Turnover")
    axes[2].set_xlabel("Rebalance month")
    axes[2].grid(True, alpha=0.25)
    positive_turnover = monthly.loc[monthly["turnover"].gt(0), ["turnover", "total_cost"]]
    if not positive_turnover.empty:
        cost_per_turnover_bps = float((positive_turnover["total_cost"] / positive_turnover["turnover"]).median() * 10000.0)
        if np.isfinite(cost_per_turnover_bps) and cost_per_turnover_bps > 0:
            sec_axis = axes[2].secondary_yaxis(
                "right",
                functions=(lambda x: x * cost_per_turnover_bps, lambda y: y / cost_per_turnover_bps),
            )
            sec_axis.set_ylabel("Cost, bps")
    axes[2].legend(frameon=False, loc="upper right")

    fig.suptitle("GPU Surface Model Decile Portfolio")
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="outputs/reports/gpu_model/gpu_neural_oos_predictions.parquet",
        help="Path relative to project root containing GPU OOS predictions.",
    )
    parser.add_argument(
        "--panel",
        default="data/processed/panel/surface_features_panel.parquet",
        help="Path relative to project root containing feature panel cost proxies.",
    )
    parser.add_argument("--score-col", default="pred")
    parser.add_argument("--quantiles", type=int, default=10)
    parser.add_argument("--min-assets", type=int, default=20)
    parser.add_argument("--one-way-cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--use-option-spread-cost",
        action="store_true",
        help=(
            "Also subtract option quote half-spreads from trade weights. Off by default because "
            "option spreads are a surface-quality diagnostic, not the equity portfolio's trading spread."
        ),
    )
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)

    predictions = load_predictions(root, args.predictions)
    panel_costs = load_panel_cost_columns(root, args.panel)
    frame = predictions.copy()
    panel_used = False
    if not panel_costs.empty:
        frame = frame.merge(panel_costs, on=["date", "permno"], how="left")
        panel_used = True

    weights = long_short_decile_weights(
        frame,
        score_col=args.score_col,
        quantiles=args.quantiles,
        min_assets=args.min_assets,
    )
    if weights.empty:
        raise RuntimeError("No eligible long-short weights were formed.")

    return_cols = ["date", "permno", "next_ret"]
    half_spread_col = "half_spread" if args.use_option_spread_cost and "half_spread" in frame.columns else None
    if half_spread_col:
        return_cols.append(half_spread_col)
    returns = frame[return_cols].drop_duplicates(["date", "permno"])
    monthly = long_short_portfolio_returns(
        weights,
        returns,
        return_col="next_ret",
        half_spread_col=half_spread_col,
        one_way_cost_bps=args.one_way_cost_bps,
    )

    report_dir = root / "outputs" / "reports" / "backtests"
    report_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = report_dir / "gpu_neural_decile_backtest.csv"
    summary_path = report_dir / "gpu_neural_decile_summary.csv"
    monthly.to_csv(monthly_path, index=False)

    gross_summary = performance_summary(monthly, "gross_return")
    net_summary = performance_summary(monthly, "net_return")
    summary = pd.DataFrame(
        [
            {"return_type": "gross", **gross_summary},
            {"return_type": "net", **net_summary},
        ]
    )
    summary.to_csv(summary_path, index=False)
    figures = write_backtest_figure(monthly, root / "outputs" / "figures" / "full" / "gpu_decile_backtest")

    manifest = {
        "status": "PASS",
        "model_source": "gpu_neural_oos_predictions",
        "predictions_rows": int(len(predictions)),
        "weights_rows": int(len(weights)),
        "months": int(monthly["date"].nunique()) if not monthly.empty else 0,
        "date_min": str(monthly["date"].min().date()) if not monthly.empty else None,
        "date_max": str(monthly["date"].max().date()) if not monthly.empty else None,
        "quantiles": args.quantiles,
        "min_assets": args.min_assets,
        "one_way_cost_bps": args.one_way_cost_bps,
        "panel_costs_used": panel_used,
        "spread_costs_used": bool(half_spread_col),
        "option_half_spread_available": bool("half_spread" in frame.columns),
        "median_option_half_spread": (
            float(pd.to_numeric(frame["half_spread"], errors="coerce").median())
            if "half_spread" in frame.columns
            else None
        ),
        "cost_note": (
            "Default net returns subtract fixed one-way equity trading costs only. "
            "Option quote spreads are retained as diagnostics unless --use-option-spread-cost is set."
        ),
        "average_turnover": float(monthly["turnover"].mean()) if not monthly.empty else None,
        "average_total_cost": float(monthly["total_cost"].mean()) if not monthly.empty else None,
        "gross": gross_summary,
        "net": net_summary,
        "net_max_drawdown_check": max_drawdown(monthly["net_return"]) if not monthly.empty else None,
        "artifacts": {
            "monthly_csv": str(monthly_path.relative_to(root)),
            "summary_csv": str(summary_path.relative_to(root)),
            "figures": [str(path.relative_to(root)) for path in figures],
        },
    }
    write_json_atomic(dirs["manifests"] / "portfolio_backtest_manifest.json", manifest)
    print("portfolio_backtest_status=PASS")
    print(
        "net_mean_monthly="
        f"{net_summary.get('mean_monthly_return')} net_sharpe={net_summary.get('annualized_sharpe')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
