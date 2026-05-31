from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surface_returns.inference import (
    decile_returns_from_predictions,
    factor_alpha_table,
    grs_test,
    moving_block_bootstrap_summary,
)
from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs


def load_state_table(root: Path) -> pd.DataFrame:
    candidates = [
        root / "data" / "processed" / "states" / "monthly_state_controls_external.parquet",
        root / "data" / "processed" / "states" / "monthly_state_controls.parquet",
    ]
    path = next((item for item in candidates if item.exists()), candidates[-1])
    if not path.exists():
        raise FileNotFoundError(f"Missing state controls: {path}")
    state = pd.read_parquet(path)
    state["date"] = pd.to_datetime(state["month"]) - pd.DateOffset(months=1)
    state["date"] = state["date"].dt.to_period("M").dt.to_timestamp()
    return state.drop(columns=["month"])


def load_predictions(root: Path, rel_path: str) -> pd.DataFrame:
    path = root / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {path}")
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_backtest(root: Path) -> pd.DataFrame:
    path = root / "outputs" / "reports" / "backtests" / "gpu_neural_decile_backtest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing backtest returns: {path}")
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    frame = frame.rename(columns={"gross_return": "gross_ls", "net_return": "net_ls"})
    return frame


def load_proposal_portfolio(root: Path) -> pd.DataFrame:
    path = root / "outputs" / "reports" / "backtests" / "proposal_buffered_portfolio.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    return frame.rename(
        columns={
            "gross_return": "proposal_gross_ls",
            "beta_neutral_gross_return": "proposal_beta_neutral_ls",
            "net_return": "proposal_net_ls",
            "taq_net_return": "proposal_taq_net_ls",
        }
    )


def write_inference_figure(alpha_tables: dict[str, pd.DataFrame], bootstrap: dict[str, object], output_prefix: Path) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0))
    rows = []
    for model, table in alpha_tables.items():
        if table.empty:
            continue
        subset = table[
            table["portfolio"].isin(
                [
                    "gross_ls",
                    "net_ls",
                    "decile_ls",
                    "proposal_gross_ls",
                    "proposal_beta_neutral_ls",
                    "proposal_net_ls",
                    "proposal_taq_net_ls",
                ]
            )
        ].copy()
        subset["model"] = model
        rows.append(subset)
    if rows:
        plot = pd.concat(rows, ignore_index=True)
        plot["label"] = plot["model"] + " / " + plot["portfolio"]
        axes[0].barh(plot["label"], plot["alpha_annualized"], color="#4c78a8")
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_xlabel("Annualized alpha")
        axes[0].grid(True, axis="x", alpha=0.25)
    else:
        axes[0].text(0.5, 0.5, "No alpha estimates", ha="center", va="center")
    if bootstrap.get("status") == "PASS":
        mean = float(bootstrap["mean_monthly"])
        low = float(bootstrap["mean_monthly_ci_low"])
        high = float(bootstrap["mean_monthly_ci_high"])
        axes[1].errorbar(["Net LS"], [mean], yerr=[[mean - low], [high - mean]], fmt="o", color="#f58518")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_ylabel("Monthly return")
        axes[1].set_title("Moving-Block Bootstrap 95% CI")
        axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Factor-Adjusted Asset-Pricing Inference")
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/reports/gpu_model/gpu_neural_oos_predictions.parquet")
    parser.add_argument("--nw-lags", type=int, default=6)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-block", type=int, default=6)
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)

    predictions = load_predictions(root, args.predictions)
    deciles = decile_returns_from_predictions(predictions)
    backtest = load_backtest(root)
    proposal = load_proposal_portfolio(root)
    states = load_state_table(root)
    factor_cols_3 = [col for col in ["mktrf", "smb", "hml"] if col in states]
    factor_cols_5 = [col for col in ["mktrf", "smb", "hml", "rmw", "cma", "umd"] if col in states]

    decile_return_cols = [f"decile_{idx}" for idx in range(1, 11)] + ["decile_ls"]
    ff3_deciles = factor_alpha_table(deciles, states, decile_return_cols, factor_cols_3, lags=args.nw_lags)
    ff5_deciles = factor_alpha_table(deciles, states, decile_return_cols, factor_cols_5, lags=args.nw_lags)
    ff3_backtest = factor_alpha_table(backtest, states, ["gross_ls", "net_ls"], factor_cols_3, lags=args.nw_lags)
    ff5_backtest = factor_alpha_table(backtest, states, ["gross_ls", "net_ls"], factor_cols_5, lags=args.nw_lags)
    proposal_cols = [
        col
        for col in ["proposal_gross_ls", "proposal_beta_neutral_ls", "proposal_net_ls", "proposal_taq_net_ls"]
        if col in proposal
    ]
    ff3_proposal = (
        factor_alpha_table(proposal, states, proposal_cols, factor_cols_3, lags=args.nw_lags)
        if not proposal.empty
        else pd.DataFrame()
    )
    ff5_proposal = (
        factor_alpha_table(proposal, states, proposal_cols, factor_cols_5, lags=args.nw_lags)
        if not proposal.empty
        else pd.DataFrame()
    )
    grs_ff3 = grs_test(deciles, states, [f"decile_{idx}" for idx in range(1, 11)], factor_cols_3)
    grs_ff5 = grs_test(deciles, states, [f"decile_{idx}" for idx in range(1, 11)], factor_cols_5)
    bootstrap = moving_block_bootstrap_summary(
        backtest["net_ls"],
        block_length=args.bootstrap_block,
        n_boot=args.bootstrap_reps,
    )

    report_dir = root / "outputs" / "reports" / "inference"
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "decile_returns": report_dir / "gpu_prediction_decile_returns.csv",
        "ff3_decile_alphas": report_dir / "ff3_decile_alphas.csv",
        "ff5_decile_alphas": report_dir / "ff5_decile_alphas.csv",
        "ff3_backtest_alphas": report_dir / "ff3_backtest_alphas.csv",
        "ff5_backtest_alphas": report_dir / "ff5_backtest_alphas.csv",
        "ff3_proposal_alphas": report_dir / "ff3_proposal_alphas.csv",
        "ff5_proposal_alphas": report_dir / "ff5_proposal_alphas.csv",
    }
    deciles.to_csv(paths["decile_returns"], index=False)
    ff3_deciles.to_csv(paths["ff3_decile_alphas"], index=False)
    ff5_deciles.to_csv(paths["ff5_decile_alphas"], index=False)
    ff3_backtest.to_csv(paths["ff3_backtest_alphas"], index=False)
    ff5_backtest.to_csv(paths["ff5_backtest_alphas"], index=False)
    ff3_proposal.to_csv(paths["ff3_proposal_alphas"], index=False)
    ff5_proposal.to_csv(paths["ff5_proposal_alphas"], index=False)
    figures = write_inference_figure(
        {
            "FF3 base": ff3_backtest,
            "FF5 base": ff5_backtest,
            "FF3 buffered": ff3_proposal,
            "FF5 buffered": ff5_proposal,
        },
        bootstrap,
        root / "outputs" / "figures" / "full" / "asset_pricing_inference",
    )
    manifest = {
        "status": "PASS",
        "predictions_rows": int(len(predictions)),
        "decile_months": int(deciles["date"].nunique()) if not deciles.empty else 0,
        "backtest_months": int(backtest["date"].nunique()) if not backtest.empty else 0,
        "proposal_portfolio_months": int(proposal["date"].nunique()) if not proposal.empty else 0,
        "factor_models": {
            "ff3": factor_cols_3,
            "ff5_plus_umd": factor_cols_5,
        },
        "grs": {
            "ff3": grs_ff3,
            "ff5_plus_umd": grs_ff5,
        },
        "bootstrap_net_ls": bootstrap,
        "net_ff5_alpha_monthly": (
            float(ff5_backtest.loc[ff5_backtest["portfolio"].eq("net_ls"), "alpha_monthly"].iloc[0])
            if not ff5_backtest.empty and ff5_backtest["portfolio"].eq("net_ls").any()
            else None
        ),
        "net_ff5_alpha_t": (
            float(ff5_backtest.loc[ff5_backtest["portfolio"].eq("net_ls"), "alpha_t_newey_west"].iloc[0])
            if not ff5_backtest.empty and ff5_backtest["portfolio"].eq("net_ls").any()
            else None
        ),
        "proposal_net_ff5_alpha_monthly": (
            float(ff5_proposal.loc[ff5_proposal["portfolio"].eq("proposal_net_ls"), "alpha_monthly"].iloc[0])
            if not ff5_proposal.empty and ff5_proposal["portfolio"].eq("proposal_net_ls").any()
            else None
        ),
        "proposal_net_ff5_alpha_t": (
            float(ff5_proposal.loc[ff5_proposal["portfolio"].eq("proposal_net_ls"), "alpha_t_newey_west"].iloc[0])
            if not ff5_proposal.empty and ff5_proposal["portfolio"].eq("proposal_net_ls").any()
            else None
        ),
        "proposal_taq_net_ff5_alpha_monthly": (
            float(ff5_proposal.loc[ff5_proposal["portfolio"].eq("proposal_taq_net_ls"), "alpha_monthly"].iloc[0])
            if not ff5_proposal.empty and ff5_proposal["portfolio"].eq("proposal_taq_net_ls").any()
            else None
        ),
        "proposal_taq_net_ff5_alpha_t": (
            float(ff5_proposal.loc[ff5_proposal["portfolio"].eq("proposal_taq_net_ls"), "alpha_t_newey_west"].iloc[0])
            if not ff5_proposal.empty and ff5_proposal["portfolio"].eq("proposal_taq_net_ls").any()
            else None
        ),
        "artifacts": {
            key: str(path.relative_to(root)) for key, path in paths.items()
        }
        | {"figures": [str(path.relative_to(root)) for path in figures]},
    }
    write_json_atomic(dirs["manifests"] / "asset_pricing_inference_manifest.json", manifest)
    print("asset_pricing_inference_status=PASS")
    print(
        f"net_ff5_alpha={manifest['net_ff5_alpha_monthly']} t={manifest['net_ff5_alpha_t']} "
        f"proposal_net_ff5_alpha={manifest['proposal_net_ff5_alpha_monthly']} "
        f"proposal_t={manifest['proposal_net_ff5_alpha_t']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
