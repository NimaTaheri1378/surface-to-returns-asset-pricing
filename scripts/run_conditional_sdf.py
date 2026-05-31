from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs


def load_predictions(root: Path, rel_path: str) -> pd.DataFrame:
    path = root / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    frame = pd.read_parquet(path)
    required = {"date", "permno", "pred", "next_ret"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Prediction file is missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.dropna(subset=["date", "permno", "pred", "next_ret"])


def build_decile_panel(predictions: pd.DataFrame, quantiles: int = 10) -> pd.DataFrame:
    rows = []
    for date, group in predictions.groupby("date", sort=True):
        if len(group) < quantiles * 5:
            continue
        ranks = group["pred"].rank(method="first", pct=True)
        decile = np.ceil(ranks * quantiles).clip(1, quantiles).astype(int)
        returns = group.assign(decile=decile).groupby("decile")["next_ret"].mean()
        if len(returns) < quantiles:
            continue
        row = {
            "date": date,
            "signal_dispersion": float(group["pred"].std(ddof=0)),
            "mean_abs_signal": float(group["pred"].abs().mean()),
        }
        for idx in range(1, quantiles + 1):
            row[f"decile_{idx}"] = float(returns.loc[idx])
        row["hml"] = row[f"decile_{quantiles}"] - row["decile_1"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def walk_forward_dates(dates: list[pd.Timestamp], min_train_months: int, test_months: int):
    start = min_train_months
    while start < len(dates):
        end = min(start + test_months, len(dates))
        yield dates[:start], dates[start:end]
        start = end


def factor_matrix(frame: pd.DataFrame, state_means: pd.Series, state_stds: pd.Series) -> np.ndarray:
    states = frame[["signal_dispersion", "mean_abs_signal"]].copy()
    states = (states - state_means) / state_stds.replace(0, 1.0)
    hml = frame["hml"].to_numpy(dtype=np.float64)
    return np.column_stack(
        [
            hml,
            hml * states["signal_dispersion"].to_numpy(dtype=np.float64),
            hml * states["mean_abs_signal"].to_numpy(dtype=np.float64),
        ]
    )


def solve_sdf_theta(asset_returns: np.ndarray, factors: np.ndarray, ridge: float, device: str):
    import torch

    torch_device = torch.device(device)
    returns = torch.tensor(asset_returns, dtype=torch.float64, device=torch_device)
    factor_tensor = torch.tensor(factors, dtype=torch.float64, device=torch_device)
    y = returns.mean(dim=0)
    moments = torch.einsum("tk,tn->nk", factor_tensor, returns) / returns.shape[0]
    lhs = moments.T @ moments + ridge * torch.eye(moments.shape[1], dtype=torch.float64, device=torch_device)
    rhs = moments.T @ y
    theta = torch.linalg.solve(lhs, rhs)
    fitted = moments @ theta
    errors = y - fitted
    return (
        theta.detach().cpu().numpy(),
        fitted.detach().cpu().numpy(),
        errors.detach().cpu().numpy(),
    )


def evaluate_fold(train: pd.DataFrame, test: pd.DataFrame, decile_cols: list[str], ridge: float, device: str):
    state_means = train[["signal_dispersion", "mean_abs_signal"]].mean()
    state_stds = train[["signal_dispersion", "mean_abs_signal"]].std(ddof=0).replace(0, 1.0)
    train_factors = factor_matrix(train, state_means, state_stds)
    test_factors = factor_matrix(test, state_means, state_stds)
    theta, _train_fitted, _train_errors = solve_sdf_theta(
        train[decile_cols].to_numpy(dtype=np.float64),
        train_factors,
        ridge,
        device,
    )
    test_returns = test[decile_cols].to_numpy(dtype=np.float64)
    test_mean = test_returns.mean(axis=0)
    test_moments = np.einsum("tk,tn->nk", test_factors, test_returns) / len(test)
    fitted = test_moments @ theta
    errors = test_mean - fitted
    sdf_payoff = test_factors @ theta
    return theta, fitted, errors, sdf_payoff


def write_sdf_figure(monthly: pd.DataFrame, errors: pd.DataFrame, output_prefix: Path) -> list[Path]:
    if monthly.empty or errors.empty:
        return []
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_monthly = monthly.sort_values("date").copy()
    plot_monthly["hml_wealth"] = (1.0 + plot_monthly["hml"].fillna(0.0)).cumprod()
    plot_monthly["sdf_payoff_12m"] = plot_monthly["conditional_sdf_payoff"].rolling(12, min_periods=3).mean()
    mean_errors = errors.groupby("decile", as_index=False)["pricing_error"].mean()

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5))
    axes[0].plot(plot_monthly["date"], plot_monthly["hml_wealth"], label="Signal HML", color="#4c78a8")
    axes[0].set_ylabel("HML growth of $1")
    axes[0].grid(True, alpha=0.25)
    payoff_axis = axes[0].twinx()
    payoff_axis.plot(
        plot_monthly["date"],
        plot_monthly["sdf_payoff_12m"],
        label="SDF payoff, 12m avg.",
        color="#f58518",
        linewidth=1.5,
    )
    payoff_axis.axhline(0, color="#f58518", linewidth=0.8, alpha=0.45)
    payoff_axis.set_ylabel("SDF payoff")
    handles = [axes[0].lines[0], payoff_axis.lines[0]]
    axes[0].legend(handles, [item.get_label() for item in handles], frameon=False)

    axes[1].bar(mean_errors["decile"].astype(str), mean_errors["pricing_error"], color="#54a24b")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Predicted-return decile")
    axes[1].set_ylabel("OOS pricing error")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Conditional SDF Diagnostics")
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/reports/gpu_model/gpu_neural_oos_predictions.parquet")
    parser.add_argument("--quantiles", type=int, default=10)
    parser.add_argument("--min-train-months", type=int, default=60)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--ridge", type=float, default=1e-6)
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    predictions = load_predictions(root, args.predictions)
    panel = build_decile_panel(predictions, args.quantiles)
    decile_cols = [f"decile_{idx}" for idx in range(1, args.quantiles + 1)]
    dates = sorted(panel["date"].dropna().unique())
    folds = list(walk_forward_dates(dates, args.min_train_months, args.test_months))
    if not folds:
        raise RuntimeError("Not enough OOS prediction months for conditional SDF evaluation.")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    monthly_rows = []
    error_rows = []
    theta_rows = []
    for fold_id, (train_dates, test_dates) in enumerate(folds, start=1):
        train = panel[panel["date"].isin(train_dates)]
        test = panel[panel["date"].isin(test_dates)]
        theta, fitted, errors, sdf_payoff = evaluate_fold(train, test, decile_cols, args.ridge, device)
        for idx, decile in enumerate(range(1, args.quantiles + 1)):
            error_rows.append(
                {
                    "fold": fold_id,
                    "decile": decile,
                    "mean_return": float(test[decile_cols[idx]].mean()),
                    "fitted_mean_return": float(fitted[idx]),
                    "pricing_error": float(errors[idx]),
                }
            )
        for idx, value in enumerate(theta, start=1):
            theta_rows.append({"fold": fold_id, "factor": f"theta_{idx}", "value": float(value)})
        for date, hml, payoff in zip(test["date"], test["hml"], sdf_payoff, strict=True):
            monthly_rows.append(
                {
                    "date": date,
                    "fold": fold_id,
                    "hml": float(hml),
                    "conditional_sdf_payoff": float(payoff),
                }
            )

    monthly = pd.DataFrame(monthly_rows)
    errors = pd.DataFrame(error_rows)
    thetas = pd.DataFrame(theta_rows)
    report_dir = root / "outputs" / "reports" / "sdf"
    report_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = report_dir / "conditional_sdf_monthly.csv"
    errors_path = report_dir / "conditional_sdf_pricing_errors.csv"
    theta_path = report_dir / "conditional_sdf_thetas.csv"
    monthly.to_csv(monthly_path, index=False)
    errors.to_csv(errors_path, index=False)
    thetas.to_csv(theta_path, index=False)
    figures = write_sdf_figure(monthly, errors, root / "outputs" / "figures" / "full" / "conditional_sdf")
    pricing_error = errors["pricing_error"]
    manifest = {
        "status": "PASS",
        "device": device,
        "predictions_rows": int(len(predictions)),
        "decile_months": int(len(panel)),
        "oos_months": int(monthly["date"].nunique()) if not monthly.empty else 0,
        "folds": len(folds),
        "factors": ["hml", "hml_x_signal_dispersion", "hml_x_mean_abs_signal"],
        "pricing_error_mean_abs": float(pricing_error.abs().mean()),
        "pricing_error_rms": float(np.sqrt(np.mean(np.square(pricing_error)))),
        "hml_mean_monthly": float(monthly["hml"].mean()) if not monthly.empty else None,
        "conditional_sdf_payoff_mean_monthly": (
            float(monthly["conditional_sdf_payoff"].mean()) if not monthly.empty else None
        ),
        "artifacts": {
            "monthly_csv": str(monthly_path.relative_to(root)),
            "pricing_errors_csv": str(errors_path.relative_to(root)),
            "thetas_csv": str(theta_path.relative_to(root)),
            "figures": [str(path.relative_to(root)) for path in figures],
        },
    }
    write_json_atomic(dirs["manifests"] / "conditional_sdf_manifest.json", manifest)
    print("conditional_sdf_status=PASS")
    print(
        "device="
        f"{device} pricing_error_rms={manifest['pricing_error_rms']} "
        f"oos_months={manifest['oos_months']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
