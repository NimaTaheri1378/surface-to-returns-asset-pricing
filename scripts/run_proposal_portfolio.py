from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from surface_returns.backtest import performance_summary, weight_trades
from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs
from surface_returns.portfolio import (
    buffered_memberships,
    portfolio_returns_with_hedge,
    rolling_monthly_beta,
    sector_balanced_weights,
    sic_to_ff12,
)
from surface_returns.trading_costs import portfolio_taq_costs


def load_predictions(root: Path, rel_path: str) -> pd.DataFrame:
    path = root / rel_path
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_panel(root: Path) -> pd.DataFrame:
    candidates = [
        root / "data" / "processed" / "panel" / "surface_characteristic_state_ibes_regsho_external_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_taq_external_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_external_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_external_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_ibes_regsho_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_taq_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_panel.parquet",
    ]
    path = next((item for item in candidates if item.exists()), candidates[-1])
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_crsp_sector(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("data/processed/full/crsp_monthly/year=*/month=*.parquet"))
    if not paths:
        return pd.DataFrame()
    frame = pd.concat((pd.read_parquet(path, columns=["permno", "date", "siccd"]) for path in paths), ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    frame["sector"] = frame["siccd"].map(sic_to_ff12)
    return frame[["date", "permno", "siccd", "sector"]].drop_duplicates(["date", "permno"])


def load_holding_factors(root: Path) -> pd.DataFrame:
    state = pd.read_parquet(root / "data" / "processed" / "states" / "monthly_state_controls.parquet")
    state["date"] = pd.to_datetime(state["month"]) - pd.DateOffset(months=1)
    state["date"] = state["date"].dt.to_period("M").dt.to_timestamp()
    return state[["date", "mktrf"]].dropna()


def write_figure(monthly: pd.DataFrame, output_prefix: Path) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot = monthly.copy().sort_values("date")
    plot["gross_wealth"] = (1.0 + plot["gross_return"].fillna(0)).cumprod()
    plot["beta_neutral_wealth"] = (1.0 + plot["beta_neutral_gross_return"].fillna(0)).cumprod()
    plot["net_wealth"] = (1.0 + plot["net_return"].fillna(0)).cumprod()
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.0), sharex=True)
    axes[0].plot(plot["date"], plot["gross_wealth"], label="Sector-balanced gross", color="#4c78a8")
    axes[0].plot(plot["date"], plot["beta_neutral_wealth"], label="Beta-hedged gross", color="#f58518")
    axes[0].plot(plot["date"], plot["net_wealth"], label="Net", color="#54a24b")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(plot["date"], plot["beta_exposure"], color="#e45756")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Equity beta exposure")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(plot["date"], plot["turnover"], label="Turnover", color="#72b7b2")
    if "taq_total_cost" in plot:
        axes[2].plot(plot["date"], plot["taq_total_cost"], label="TAQ cost", color="#b279a2", alpha=0.8)
        axes[2].legend(frameon=False)
    axes[2].set_ylabel("Turnover")
    axes[2].set_xlabel("Signal month")
    axes[2].grid(True, alpha=0.25)
    fig.suptitle("Buffered Sector-Balanced Beta-Hedged Portfolio")
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/reports/gpu_model/gpu_neural_oos_predictions.parquet")
    parser.add_argument("--entry-pct", type=float, default=0.10)
    parser.add_argument("--exit-pct", type=float, default=0.20)
    parser.add_argument("--one-way-cost-bps", type=float, default=10.0)
    args = parser.parse_args()
    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)

    preds = load_predictions(root, args.predictions)
    panel = load_panel(root)
    sector = load_crsp_sector(root)
    factors = load_holding_factors(root)
    if "beta_252d" in panel:
        beta_col = "beta_signal"
        beta = panel[["date", "permno", "beta_252d"]].drop_duplicates(["date", "permno"]).rename(
            columns={"beta_252d": beta_col}
        )
        beta["date"] = pd.to_datetime(beta["date"]).dt.to_period("M").dt.to_timestamp()
    else:
        beta_col = "beta_signal"
        beta = rolling_monthly_beta(panel).rename(columns={"beta_60m": beta_col})
    frame = preds.merge(panel[["date", "permno", "next_ret", "ret", "mktrf"]].drop_duplicates(["date", "permno"]), on=["date", "permno"], how="left", suffixes=("", "_panel"))
    if "next_ret_panel" in frame:
        frame["next_ret"] = frame["next_ret"].where(frame["next_ret"].notna(), frame["next_ret_panel"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    frame = frame.merge(sector, on=["date", "permno"], how="left")
    frame["sector"] = frame["sector"].fillna("Other")
    frame = frame.merge(beta, on=["date", "permno"], how="left")
    members = buffered_memberships(frame, entry_pct=args.entry_pct, exit_pct=args.exit_pct)
    weights = sector_balanced_weights(members)
    monthly = portfolio_returns_with_hedge(
        weights,
        frame[["date", "permno", "next_ret", beta_col]].drop_duplicates(["date", "permno"]),
        factors,
        one_way_cost_bps=args.one_way_cost_bps,
        beta_col=beta_col,
    )
    taq_cols = [col for col in panel.columns if col.startswith("taq_")]
    if taq_cols:
        cost_panel = panel[["date", "permno", *taq_cols]].drop_duplicates(["date", "permno"])
        cost_panel["date"] = pd.to_datetime(cost_panel["date"]).dt.to_period("M").dt.to_timestamp()
        taq_monthly = portfolio_taq_costs(weight_trades(weights), cost_panel)
        monthly = monthly.merge(taq_monthly, on="date", how="left")
        monthly["taq_total_cost"] = monthly["taq_total_cost"].fillna(monthly["fixed_cost"])
        monthly["taq_net_return"] = monthly["beta_neutral_gross_return"] - monthly["taq_total_cost"]

    report_dir = root / "outputs" / "reports" / "backtests"
    report_dir.mkdir(parents=True, exist_ok=True)
    weights_path = report_dir / "proposal_buffered_weights.parquet"
    monthly_path = report_dir / "proposal_buffered_portfolio.csv"
    summary_path = report_dir / "proposal_buffered_summary.csv"
    weights.to_parquet(weights_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    summary = pd.DataFrame(
        [
            {"return_type": "gross", **performance_summary(monthly, "gross_return")},
            {"return_type": "beta_neutral_gross", **performance_summary(monthly, "beta_neutral_gross_return")},
            {"return_type": "net", **performance_summary(monthly, "net_return")},
            *(
                [{"return_type": "taq_net", **performance_summary(monthly, "taq_net_return")}]
                if "taq_net_return" in monthly
                else []
            ),
        ]
    )
    summary.to_csv(summary_path, index=False)
    figures = write_figure(monthly, root / "outputs" / "figures" / "full" / "proposal_buffered_portfolio")
    manifest = {
        "status": "PASS",
        "predictions_rows": int(len(preds)),
        "members_rows": int(len(members)),
        "weights_rows": int(len(weights)),
        "months": int(monthly["date"].nunique()) if not monthly.empty else 0,
        "entry_pct": args.entry_pct,
        "exit_pct": args.exit_pct,
        "one_way_cost_bps": args.one_way_cost_bps,
        "beta_source": "daily_beta_252d" if "beta_252d" in panel else "monthly_beta_60m",
        "taq_costs_used": "taq_net_return" in monthly,
        "average_turnover": float(monthly["turnover"].mean()) if not monthly.empty else None,
        "average_abs_beta_exposure": float(monthly["beta_exposure"].abs().mean()) if "beta_exposure" in monthly else None,
        "net": performance_summary(monthly, "net_return"),
        "taq_net": performance_summary(monthly, "taq_net_return") if "taq_net_return" in monthly else None,
        "artifacts": {
            "weights": str(weights_path.relative_to(root)),
            "monthly_csv": str(monthly_path.relative_to(root)),
            "summary_csv": str(summary_path.relative_to(root)),
            "figures": [str(path.relative_to(root)) for path in figures],
        },
    }
    write_json_atomic(dirs["manifests"] / "proposal_portfolio_manifest.json", manifest)
    print("proposal_portfolio_status=PASS")
    print(f"net_mean={manifest['net']['mean_monthly_return']} turnover={manifest['average_turnover']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
