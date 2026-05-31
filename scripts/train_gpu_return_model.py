from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surface_returns.conditional_sdf import DEFAULT_FEATURE_CANDIDATES
from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs


def candidate_panel_paths(root: Path) -> list[Path]:
    return [
        root / "data" / "processed" / "panel" / "surface_characteristic_state_ibes_regsho_external_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_taq_external_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_external_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_external_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_ibes_regsho_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_taq_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_features_panel.parquet",
    ]


def select_features(frame: pd.DataFrame, min_nonmissing: float) -> list[str]:
    candidates = list(DEFAULT_FEATURE_CANDIDATES)
    candidates.extend(["log_n_contracts"])
    candidates.extend(sorted(col for col in frame.columns if col.startswith("surface_ae_")))
    selected = []
    for col in dict.fromkeys(candidates):
        if col not in frame:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().mean() >= min_nonmissing and values.nunique(dropna=True) > 2:
            selected.append(col)
    return selected


def standardize_by_month(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out.groupby("date")[col].transform(lambda s: s.fillna(s.median()))
        out[col] = out[col].fillna(out[col].median())
        def _scale(s: pd.Series) -> pd.Series:
            std = s.std(ddof=0)
            if not np.isfinite(std) or std == 0:
                return pd.Series(0.0, index=s.index)
            return (s - s.mean()) / std

        out[col] = out.groupby("date")[col].transform(_scale)
    return out


def load_training_frame(root: Path, panel_arg: str | None, min_nonmissing: float) -> tuple[pd.DataFrame, list[str], Path]:
    panel_path = root / panel_arg if panel_arg else next((path for path in candidate_panel_paths(root) if path.exists()), candidate_panel_paths(root)[-1])
    emb_path = root / "data" / "processed" / "panel" / "surface_autoencoder_embeddings.parquet"
    if not panel_path.exists() or not emb_path.exists():
        raise FileNotFoundError("Panel and autoencoder embeddings must exist before GPU model training.")
    panel = pd.read_parquet(panel_path)
    emb = pd.read_parquet(emb_path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.to_period("M").dt.to_timestamp()
    emb["date"] = pd.to_datetime(emb["date"])
    panel["_surface_month"] = panel["date"].dt.to_period("M").dt.to_timestamp()
    emb["_surface_month"] = emb["date"].dt.to_period("M").dt.to_timestamp()
    key_cols = ["_surface_month", "secid"]
    if "permno" in emb.columns and "permno" in panel.columns:
        key_cols = ["_surface_month", "secid", "permno"]
    emb_cols = key_cols + [col for col in emb.columns if col.startswith("surface_ae_")]
    frame = panel.merge(emb[emb_cols].drop_duplicates(key_cols), on=key_cols, how="inner")
    frame = frame.drop(columns=["_surface_month"], errors="ignore")
    if "market_equity" in frame:
        frame["log_market_equity"] = np.log(pd.to_numeric(frame["market_equity"], errors="coerce").clip(lower=1))
    if "n_contracts" in frame:
        frame["log_n_contracts"] = np.log1p(pd.to_numeric(frame["n_contracts"], errors="coerce").clip(lower=0))
    feature_cols = select_features(frame, min_nonmissing=min_nonmissing)
    keep_cols = ["date", "permno", "secid", "next_ret"] + feature_cols
    frame = frame[keep_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=["next_ret"])
    frame = standardize_by_month(frame, feature_cols)
    frame = frame.dropna(subset=feature_cols)
    return frame.sort_values(["date", "permno"]).reset_index(drop=True), feature_cols, panel_path


def make_folds(dates: list[pd.Timestamp], min_train_months: int, test_months: int) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    folds = []
    start = min_train_months
    while start < len(dates):
        end = min(start + test_months, len(dates))
        folds.append((dates[:start], dates[start:end]))
        start = end
    return folds


def train_fold(x_train, y_train, x_test, args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std == 0 or np.isnan(y_std):
        y_std = 1.0
    train_tensor = torch.tensor((x_train - x_mean) / x_std, dtype=torch.float32)
    y_tensor = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32).view(-1, 1)
    test_tensor = torch.tensor((x_test - x_mean) / x_std, dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_tensor, y_tensor), batch_size=args.batch_size, shuffle=True)
    model = nn.Sequential(
        nn.Linear(train_tensor.shape[1], args.hidden_dim),
        nn.GELU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_dim, args.hidden_dim // 2),
        nn.GELU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_dim // 2, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.HuberLoss(delta=1.0)
    losses = []
    for _epoch in range(args.epochs):
        batch_losses = []
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(batch_losses)))
    model.eval()
    with torch.no_grad():
        preds = model(test_tensor.to(device)).detach().cpu().numpy().reshape(-1) * y_std + y_mean
    return preds, losses, device.type


def write_prediction_figure(preds: pd.DataFrame, output_prefix: Path) -> list[Path]:
    if preds.empty:
        return []
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    monthly_rows = []
    for date, group in preds.groupby("date"):
        ranks = group["pred"].rank(pct=True)
        monthly_rows.append(
            {
                "date": date,
                "rank_ic": group["pred"].corr(group["next_ret"], method="spearman"),
                "spread": group.loc[ranks >= 0.9, "next_ret"].mean()
                - group.loc[ranks <= 0.1, "next_ret"].mean(),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(monthly["date"], monthly["rank_ic"], color="#4c78a8", linewidth=1.4)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Rank IC")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(monthly["date"], monthly["spread"], color="#f58518", linewidth=1.4)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Top-bottom return")
    axes[1].set_xlabel("Test month")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("GPU Neural Surface Model OOS Diagnostics")
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-train-months", type=int, default=120)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1378)
    parser.add_argument("--panel", default=None)
    parser.add_argument("--min-feature-nonmissing", type=float, default=0.65)
    args = parser.parse_args()
    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    np.random.seed(args.seed)
    frame, feature_cols, panel_path = load_training_frame(root, args.panel, args.min_feature_nonmissing)
    dates = sorted(frame["date"].dropna().unique())
    folds = make_folds(dates, args.min_train_months, args.test_months)
    if not folds:
        raise RuntimeError("Not enough months for walk-forward GPU model training.")
    all_preds = []
    fold_losses = []
    device = "unknown"
    for fold_id, (train_dates, test_dates) in enumerate(folds, start=1):
        train = frame[frame["date"].isin(train_dates)]
        test = frame[frame["date"].isin(test_dates)]
        x_train = train[feature_cols].to_numpy(dtype=np.float32)
        y_train = train["next_ret"].to_numpy(dtype=np.float32)
        x_test = test[feature_cols].to_numpy(dtype=np.float32)
        preds, losses, device = train_fold(x_train, y_train, x_test, args)
        out = test[["date", "permno", "secid", "next_ret"]].copy()
        out["pred"] = preds
        out["fold"] = fold_id
        all_preds.append(out)
        fold_losses.append({"fold": fold_id, "train_months": len(train_dates), "test_months": len(test_dates), "final_loss": losses[-1]})
        print(f"fold={fold_id} train_months={len(train_dates)} test_months={len(test_dates)} final_loss={losses[-1]:.6f}", flush=True)
    predictions = pd.concat(all_preds, ignore_index=True)
    pred_path = root / "outputs" / "reports" / "gpu_model" / "gpu_neural_oos_predictions.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(pred_path, index=False)
    monthly_ic_values = []
    long_short_values = []
    for _date, group in predictions.groupby("date"):
        ranks = group["pred"].rank(pct=True)
        monthly_ic_values.append(group["pred"].corr(group["next_ret"], method="spearman"))
        long_short_values.append(
            group.loc[ranks >= 0.9, "next_ret"].mean() - group.loc[ranks <= 0.1, "next_ret"].mean()
        )
    monthly_ic = pd.Series(monthly_ic_values).dropna()
    long_short = pd.Series(long_short_values).dropna()
    figures = write_prediction_figure(predictions, root / "outputs" / "figures" / "full" / "gpu_neural_oos")
    manifest = {
        "status": "PASS",
        "device": device,
        "observations": int(len(predictions)),
        "folds": len(folds),
        "panel": str(panel_path.relative_to(root)),
        "features": feature_cols,
        "mean_rank_ic": float(monthly_ic.mean()) if len(monthly_ic) else None,
        "rank_ic_t_naive": float(monthly_ic.mean() / (monthly_ic.std(ddof=1) / np.sqrt(len(monthly_ic)))) if len(monthly_ic) > 1 and monthly_ic.std(ddof=1) > 0 else None,
        "mean_top_bottom_return": float(long_short.mean()) if len(long_short) else None,
        "top_bottom_t_naive": float(long_short.mean() / (long_short.std(ddof=1) / np.sqrt(len(long_short)))) if len(long_short) > 1 and long_short.std(ddof=1) > 0 else None,
        "fold_losses": fold_losses,
        "artifacts": {
            "predictions": str(pred_path.relative_to(root)),
            "figures": [str(path.relative_to(root)) for path in figures],
        },
    }
    write_json_atomic(dirs["manifests"] / "gpu_return_model_manifest.json", manifest)
    print("gpu_return_model_status=PASS")
    print(f"device={device} mean_rank_ic={manifest['mean_rank_ic']} top_bottom={manifest['mean_top_bottom_return']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
