from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surface_returns.conditional_sdf import (
    classify_conditional_sdf_features,
    make_walk_forward_sdf_splits,
    month_arrays,
    prepare_conditional_sdf_frame,
    sdf_pricing_error_summary,
    select_conditional_sdf_features,
)
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


def load_panel_with_embeddings(root: Path, explicit_panel: str | None = None) -> tuple[pd.DataFrame, Path]:
    if explicit_panel:
        panel_path = root / explicit_panel
    else:
        panel_path = next((path for path in candidate_panel_paths(root) if path.exists()), candidate_panel_paths(root)[-1])
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing conditional SDF panel: {panel_path}")
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.to_period("M").dt.to_timestamp()
    emb_path = root / "data" / "processed" / "panel" / "surface_autoencoder_embeddings.parquet"
    if emb_path.exists():
        emb = pd.read_parquet(emb_path)
        emb["date"] = pd.to_datetime(emb["date"]).dt.to_period("M").dt.to_timestamp()
        keys = [col for col in ["date", "permno", "secid"] if col in panel.columns and col in emb.columns]
        emb_cols = keys + [col for col in emb.columns if col.startswith("surface_ae_")]
        panel = panel.merge(emb[emb_cols].drop_duplicates(keys), on=keys, how="left")
    if "log_market_equity" not in panel and "market_equity" in panel:
        panel["log_market_equity"] = np.log(pd.to_numeric(panel["market_equity"], errors="coerce").clip(lower=1.0))
    if "log_n_contracts" not in panel and "n_contracts" in panel:
        panel["log_n_contracts"] = np.log1p(pd.to_numeric(panel["n_contracts"], errors="coerce").clip(lower=0.0))
    return panel, panel_path


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray


def fit_feature_scaler(months) -> FeatureScaler:
    x = np.concatenate([item[1] for item in months if len(item[1])], axis=0)
    finite = np.isfinite(x)
    counts = finite.sum(axis=0)
    safe = np.where(finite, x, 0.0)
    mean = np.divide(safe.sum(axis=0), counts, out=np.zeros(x.shape[1], dtype=np.float32), where=counts > 0).astype(np.float32)
    centered = np.where(finite, x - mean, 0.0)
    scale = np.sqrt(
        np.divide(
            np.square(centered).sum(axis=0),
            counts,
            out=np.ones(x.shape[1], dtype=np.float32),
            where=counts > 0,
        )
    ).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return FeatureScaler(mean=mean, scale=scale)


def transform_months(months, scaler: FeatureScaler):
    out = []
    for date, x_np, r_np, ids in months:
        x = (x_np.astype(np.float32) - scaler.mean) / scaler.scale
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        out.append((date, x.astype(np.float32), r_np.astype(np.float32), ids))
    return out


def build_model(group_dims: dict[str, int], latent_dim: int, hidden_dim: int, branch_dim: int, dropout: float):
    import torch
    import torch.nn as nn

    def branch(input_dim: int) -> nn.Module | None:
        if input_dim <= 0:
            return None
        return nn.Sequential(
            nn.Linear(input_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, branch_dim),
            nn.GELU(),
        )

    class FlagshipConditionalAutoencoderSDF(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.surface_dim = int(group_dims["surface"])
            self.tabular_dim = int(group_dims["tabular"])
            self.state_dim = int(group_dims["state"])
            self.surface_net = branch(self.surface_dim)
            self.tabular_net = branch(self.tabular_dim)
            self.state_net = branch(self.state_dim)
            fused_dim = branch_dim * 3
            self.beta_net = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.factor_weight_net = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.lambda_net = nn.Sequential(
                nn.Linear(branch_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.direct_return_head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def _apply_branch(self, net, values, n_obs: int, device):
            if net is None:
                return torch.zeros((n_obs, branch_dim), dtype=values.dtype, device=device)
            return net(values)

        def forward(self, x):
            n_obs = x.shape[0]
            cursor = 0
            surface_x = x[:, cursor : cursor + self.surface_dim]
            cursor += self.surface_dim
            tabular_x = x[:, cursor : cursor + self.tabular_dim]
            cursor += self.tabular_dim
            state_x = x[:, cursor : cursor + self.state_dim]
            surface_h = self._apply_branch(self.surface_net, surface_x, n_obs, x.device)
            tabular_h = self._apply_branch(self.tabular_net, tabular_x, n_obs, x.device)
            state_h = self._apply_branch(self.state_net, state_x, n_obs, x.device)
            fused = torch.cat([surface_h, tabular_h, state_h], dim=1)
            beta = self.beta_net(fused)
            factor_scores = self.factor_weight_net(fused)
            aggregate_state = state_h.mean(dim=0, keepdim=True)
            risk_price = self.lambda_net(aggregate_state).reshape(-1)
            direct_return = self.direct_return_head(fused).reshape(-1)
            return beta, factor_scores, risk_price, direct_return

    return FlagshipConditionalAutoencoderSDF()


def managed_portfolio_weights(factor_scores, eps: float = 1e-6):
    import torch

    scores = factor_scores - factor_scores.mean(dim=0, keepdim=True)
    raw = torch.tanh(scores)
    raw = raw - raw.mean(dim=0, keepdim=True)
    denom = raw.abs().sum(dim=0, keepdim=True).clamp_min(eps)
    return raw / denom


def pricing_components(model, x, returns, args):
    import torch

    target = returns - returns.mean()
    beta, factor_scores, risk_price, direct_return = model(x)
    weights = managed_portfolio_weights(factor_scores)
    latent_factor = weights.T @ target
    structural_expected = beta @ risk_price
    direct_expected = args.direct_head_weight * direct_return
    expected = structural_expected + direct_expected
    reconstruction = beta @ latent_factor
    pricing_error = target - structural_expected
    forecast_error = target - expected
    sdf_payoff = 1.0 - torch.dot(risk_price, latent_factor)
    beta_moments = torch.mean(pricing_error.reshape(-1, 1) * beta, dim=0)
    weight_concentration = torch.mean(torch.sum(torch.square(weights), dim=0) * weights.shape[0])
    return {
        "target": target,
        "beta": beta,
        "weights": weights,
        "latent_factor": latent_factor,
        "risk_price": risk_price,
        "structural_expected": structural_expected,
        "direct_expected": direct_expected,
        "expected": expected,
        "reconstruction": reconstruction,
        "pricing_error": pricing_error,
        "forecast_error": forecast_error,
        "sdf_payoff": sdf_payoff,
        "beta_moments": beta_moments,
        "weight_concentration": weight_concentration,
    }


def objective_loss(model, x, returns, args):
    import torch
    import torch.nn.functional as functional

    comp = pricing_components(model, x, returns, args)
    recon_loss = functional.huber_loss(comp["reconstruction"], comp["target"], delta=args.huber_delta)
    pricing_loss = functional.huber_loss(comp["structural_expected"], comp["target"], delta=args.huber_delta)
    forecast_loss = functional.huber_loss(comp["expected"], comp["target"], delta=args.huber_delta)
    moment_loss = torch.mean(torch.square(comp["beta_moments"]))
    sdf_loss = torch.square(comp["sdf_payoff"] - 1.0)
    beta_penalty = torch.mean(torch.square(comp["beta"]))
    lambda_penalty = torch.mean(torch.square(comp["risk_price"]))
    weight_penalty = comp["weight_concentration"]
    loss = (
        recon_loss
        + args.pricing_weight * pricing_loss
        + args.forecast_weight * forecast_loss
        + args.moment_weight * moment_loss
        + args.sdf_weight * sdf_loss
        + args.beta_penalty * beta_penalty
        + args.lambda_penalty * lambda_penalty
        + args.weight_penalty * weight_penalty
    )
    return loss, comp


def mean_loss(model, months, args, device) -> float:
    import torch

    losses = []
    model.eval()
    with torch.no_grad():
        for _date, x_np, r_np, _ids in months:
            if len(r_np) < args.min_assets_per_month:
                continue
            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            r = torch.tensor(r_np, dtype=torch.float32, device=device)
            loss, _comp = objective_loss(model, x, r, args)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan


def train_fold(train_months, validation_months, group_dims: dict[str, int], args):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(group_dims, args.latent_dim, args.hidden_dim, args.branch_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    losses: list[float] = []
    validation_losses: list[float] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation = np.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_months))
        epoch_losses = []
        model.train()
        for idx in order:
            _date, x_np, r_np, _ids = train_months[int(idx)]
            if len(r_np) < args.min_assets_per_month:
                continue
            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            r = torch.tensor(r_np, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, _comp = objective_loss(model, x, r, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
        val_loss = mean_loss(model, validation_months, args, device) if validation_months else train_loss
        losses.append(train_loss)
        validation_losses.append(val_loss)
        if np.isfinite(val_loss) and val_loss < best_validation - args.early_stop_min_delta:
            best_validation = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if args.early_stop_patience and stale_epochs >= args.early_stop_patience:
            break
    model.load_state_dict(best_state)
    info = {
        "epochs_run": int(len(losses)),
        "best_epoch": int(best_epoch),
        "final_loss": float(losses[-1]) if losses else np.nan,
        "best_validation_loss": float(best_validation) if np.isfinite(best_validation) else np.nan,
    }
    return model, device.type, losses, validation_losses, info


def evaluate_months(model, months, args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import torch

    device = next(model.parameters()).device
    asset_rows = []
    month_rows = []
    factor_rows = []
    model.eval()
    for date, x_np, r_np, ids in months:
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        r = torch.tensor(r_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            comp = pricing_components(model, x, r, args)
        beta_np = comp["beta"].detach().cpu().numpy()
        weights_np = comp["weights"].detach().cpu().numpy()
        factor_np = comp["latent_factor"].detach().cpu().numpy()
        risk_np = comp["risk_price"].detach().cpu().numpy()
        target_np = comp["target"].detach().cpu().numpy()
        structural_np = comp["structural_expected"].detach().cpu().numpy()
        expected_np = comp["expected"].detach().cpu().numpy()
        direct_np = comp["direct_expected"].detach().cpu().numpy()
        recon_np = comp["reconstruction"].detach().cpu().numpy()
        pricing_np = comp["pricing_error"].detach().cpu().numpy()
        forecast_np = comp["forecast_error"].detach().cpu().numpy()
        for row_idx, permno in enumerate(ids):
            row = {
                "date": date,
                "permno": int(permno),
                "next_ret": float(r_np[row_idx]),
                "sdf_target_return": float(target_np[row_idx]),
                "sdf_expected_return": float(structural_np[row_idx]),
                "sdf_direct_expected_return": float(direct_np[row_idx]),
                "sdf_total_expected_return": float(expected_np[row_idx]),
                "sdf_reconstruction": float(recon_np[row_idx]),
                "sdf_pricing_error": float(pricing_np[row_idx]),
                "sdf_forecast_error": float(forecast_np[row_idx]),
            }
            for factor_idx in range(beta_np.shape[1]):
                row[f"sdf_beta_{factor_idx + 1:02d}"] = float(beta_np[row_idx, factor_idx])
                row[f"sdf_weight_{factor_idx + 1:02d}"] = float(weights_np[row_idx, factor_idx])
            asset_rows.append(row)
        effective_n = 1.0 / np.maximum(np.sum(np.square(weights_np), axis=0), 1e-12)
        month_rows.append(
            {
                "date": date,
                "n_assets": int(len(ids)),
                "sdf_payoff": float(comp["sdf_payoff"].detach().cpu()),
                "mean_realized_return": float(np.mean(r_np)),
                "mean_target_return": float(np.mean(target_np)),
                "mean_expected_return": float(np.mean(structural_np)),
                "rms_pricing_error": float(np.sqrt(np.mean(np.square(pricing_np)))),
                "rms_forecast_error": float(np.sqrt(np.mean(np.square(forecast_np)))),
                "rms_reconstruction_error": float(np.sqrt(np.mean(np.square(target_np - recon_np)))),
                "mean_factor_effective_n": float(np.mean(effective_n)),
                "mean_abs_factor_weight": float(np.mean(np.abs(weights_np))),
            }
        )
        factor_row = {"date": date}
        for factor_idx, value in enumerate(factor_np, start=1):
            factor_row[f"sdf_factor_{factor_idx:02d}"] = float(value)
        for factor_idx, value in enumerate(risk_np, start=1):
            factor_row[f"sdf_lambda_{factor_idx:02d}"] = float(value)
        factor_rows.append(factor_row)
    return pd.DataFrame(asset_rows), pd.DataFrame(month_rows), pd.DataFrame(factor_rows)


def integrated_gradients(model, months, feature_cols: list[str], steps: int, max_assets: int, direct_head_weight: float) -> pd.DataFrame:
    import torch

    device = next(model.parameters()).device
    rows = []
    model.eval()
    used = 0
    for date, x_np, _r_np, _ids in months:
        if used >= max_assets:
            break
        take = min(len(x_np), max_assets - used)
        if take <= 0:
            continue
        x = torch.tensor(x_np[:take], dtype=torch.float32, device=device)
        baseline = torch.zeros_like(x)
        total_grad = torch.zeros_like(x)
        for alpha in torch.linspace(0.0, 1.0, int(steps), device=device):
            scaled = baseline + alpha * (x - baseline)
            scaled.requires_grad_(True)
            beta, _scores, risk_price, direct = model(scaled)
            score = (beta @ risk_price + direct_head_weight * direct).sum()
            grad = torch.autograd.grad(score, scaled)[0]
            total_grad = total_grad + grad.detach()
        attribution = (x - baseline) * total_grad / float(steps)
        mean_attr = attribution.detach().cpu().numpy().mean(axis=0)
        for feature, value in zip(feature_cols, mean_attr, strict=True):
            rows.append({"date": date, "feature": feature, "integrated_gradient": float(value)})
        used += take
    return pd.DataFrame(rows)


def write_figures(monthly: pd.DataFrame, attributions: pd.DataFrame, output_prefix: Path) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
    plot = monthly.sort_values("date").copy()
    if not plot.empty:
        axes[0].plot(plot["date"], plot["rms_pricing_error"], color="#4c78a8", label="Pricing error")
        axes[0].plot(plot["date"], plot["rms_reconstruction_error"], color="#f58518", label="Reconstruction error")
        axes[0].set_ylabel("Monthly RMS")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(frameon=False)
    if not attributions.empty:
        top = (
            attributions.groupby("feature")["integrated_gradient"]
            .mean()
            .abs()
            .sort_values(ascending=False)
            .head(18)
            .sort_values()
        )
        axes[1].barh(top.index, top.values, color="#54a24b")
        axes[1].set_xlabel("Mean absolute integrated gradient")
        axes[1].grid(True, axis="x", alpha=0.25)
    fig.suptitle("Conditional Autoencoder SDF Diagnostics")
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default=None)
    parser.add_argument("--min-train-months", type=int, default=120)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--min-assets-per-month", type=int, default=80)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--branch-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--validation-months", type=int, default=24)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=7.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-delta", type=float, default=0.05)
    parser.add_argument("--pricing-weight", type=float, default=0.35)
    parser.add_argument("--forecast-weight", type=float, default=0.15)
    parser.add_argument("--moment-weight", type=float, default=0.15)
    parser.add_argument("--sdf-weight", type=float, default=0.02)
    parser.add_argument("--direct-head-weight", type=float, default=0.25)
    parser.add_argument("--beta-penalty", type=float, default=1e-3)
    parser.add_argument("--lambda-penalty", type=float, default=1e-3)
    parser.add_argument("--weight-penalty", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--ig-max-assets", type=int, default=4000)
    parser.add_argument("--max-months", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1378)
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    np.random.seed(args.seed)

    import torch

    torch.manual_seed(args.seed)
    panel, panel_path = load_panel_with_embeddings(root, args.panel)
    selected_features = select_conditional_sdf_features(panel)
    feature_groups = classify_conditional_sdf_features(selected_features)
    feature_cols = feature_groups.ordered
    if not feature_cols:
        raise RuntimeError("No eligible conditional SDF features found.")
    frame = prepare_conditional_sdf_frame(
        panel,
        feature_cols,
        min_assets_per_month=args.min_assets_per_month,
        state_cols=feature_groups.state,
    )
    dates = sorted(frame["date"].dropna().unique())
    if args.max_months:
        dates = dates[: args.max_months]
        frame = frame[frame["date"].isin(dates)].copy()
    splits = make_walk_forward_sdf_splits(dates, args.min_train_months, args.test_months)
    if not splits:
        raise RuntimeError("Not enough months for flagship conditional autoencoder SDF training.")
    all_months = month_arrays(frame, feature_cols)
    month_dict = {date: (date, x, r, ids) for date, x, r, ids in all_months}
    group_dims = {
        "surface": len(feature_groups.surface),
        "tabular": len(feature_groups.tabular),
        "state": len(feature_groups.state),
    }

    asset_outputs = []
    monthly_outputs = []
    factor_outputs = []
    fold_rows = []
    attribution_outputs = []
    last_model = None
    device = "unknown"
    for split in splits:
        train_months = [month_dict[date] for date in split.train_dates if date in month_dict]
        test_months = [month_dict[date] for date in split.test_dates if date in month_dict]
        if not train_months or not test_months:
            continue
        validation_count = min(max(0, args.validation_months), max(0, len(train_months) - 12))
        if validation_count:
            train_core = train_months[:-validation_count]
            validation_months = train_months[-validation_count:]
        else:
            train_core = train_months
            validation_months = []
        scaler = fit_feature_scaler(train_core)
        train_core = transform_months(train_core, scaler)
        validation_months = transform_months(validation_months, scaler)
        test_months = transform_months(test_months, scaler)
        model, device, losses, validation_losses, train_info = train_fold(train_core, validation_months, group_dims, args)
        assets, monthly, factors = evaluate_months(model, test_months, args)
        assets["fold"] = split.fold
        monthly["fold"] = split.fold
        factors["fold"] = split.fold
        asset_outputs.append(assets)
        monthly_outputs.append(monthly)
        factor_outputs.append(factors)
        fold_rows.append(
            {
                "fold": split.fold,
                "train_months": len(train_core),
                "validation_months": len(validation_months),
                "test_months": len(test_months),
                "final_loss": train_info["final_loss"],
                "best_validation_loss": train_info["best_validation_loss"],
                "best_epoch": train_info["best_epoch"],
                "epochs_run": train_info["epochs_run"],
            }
        )
        if split.fold == splits[-1].fold:
            attribution_outputs.append(
                integrated_gradients(
                    model,
                    test_months,
                    feature_cols,
                    args.ig_steps,
                    args.ig_max_assets,
                    args.direct_head_weight,
                )
            )
        last_model = model
        print(
            f"fold={split.fold} train_months={len(train_core)} validation_months={len(validation_months)} "
            f"test_months={len(test_months)} final_loss={fold_rows[-1]['final_loss']:.6f} "
            f"best_val={fold_rows[-1]['best_validation_loss']:.6f}",
            flush=True,
        )

    if not asset_outputs:
        raise RuntimeError("Conditional autoencoder SDF produced no OOS folds.")
    asset_panel = pd.concat(asset_outputs, ignore_index=True)
    monthly_panel = pd.concat(monthly_outputs, ignore_index=True)
    factor_panel = pd.concat(factor_outputs, ignore_index=True)
    attributions = pd.concat(attribution_outputs, ignore_index=True) if attribution_outputs else pd.DataFrame()

    report_dir = root / "outputs" / "reports" / "sdf"
    model_dir = root / "outputs" / "models"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    asset_path = report_dir / "conditional_autoencoder_sdf_assets.parquet"
    monthly_path = report_dir / "conditional_autoencoder_sdf_monthly.csv"
    factor_path = report_dir / "conditional_autoencoder_sdf_factors.csv"
    attr_path = report_dir / "conditional_autoencoder_sdf_integrated_gradients.csv"
    model_path = model_dir / "conditional_autoencoder_sdf.pt"
    asset_panel.to_parquet(asset_path, index=False)
    monthly_panel.to_csv(monthly_path, index=False)
    factor_panel.to_csv(factor_path, index=False)
    attributions.to_csv(attr_path, index=False)
    if last_model is not None:
        torch.save(
            {
                "state_dict": last_model.state_dict(),
                "feature_cols": feature_cols,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "branch_dim": args.branch_dim,
                "dropout": args.dropout,
                "feature_groups": {
                    "surface": feature_groups.surface,
                    "tabular": feature_groups.tabular,
                    "state": feature_groups.state,
                },
                "last_fold_scaler_mean": scaler.mean.tolist(),
                "last_fold_scaler_scale": scaler.scale.tolist(),
            },
            model_path,
        )
    figures = write_figures(monthly_panel, attributions, root / "outputs" / "figures" / "full" / "conditional_autoencoder_sdf")
    pricing = sdf_pricing_error_summary(asset_panel["sdf_pricing_error"])
    recon = sdf_pricing_error_summary(asset_panel["next_ret"] - asset_panel["sdf_reconstruction"])
    manifest = {
        "status": "PASS",
        "device": device,
        "panel": str(panel_path.relative_to(root)),
        "features": feature_cols,
        "feature_groups": {
            "surface": feature_groups.surface,
            "tabular": feature_groups.tabular,
            "state": feature_groups.state,
        },
        "architecture": {
            "name": "three_branch_characteristic_managed_conditional_autoencoder_sdf",
            "surface_branch": group_dims["surface"],
            "tabular_branch": group_dims["tabular"],
            "state_branch": group_dims["state"],
            "managed_factor_portfolios": True,
            "separate_state_to_risk_price_network": True,
            "direct_return_head_weight": float(args.direct_head_weight),
            "validation_months": int(args.validation_months),
            "cross_sectional_centered_target": True,
        },
        "latent_dim": int(args.latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "branch_dim": int(args.branch_dim),
        "folds": fold_rows,
        "oos_assets": int(len(asset_panel)),
        "oos_months": int(monthly_panel["date"].nunique()) if not monthly_panel.empty else 0,
        "pricing_error": pricing,
        "reconstruction_error": recon,
        "mean_sdf_payoff": float(monthly_panel["sdf_payoff"].mean()) if not monthly_panel.empty else None,
        "artifacts": {
            "asset_panel": str(asset_path.relative_to(root)),
            "monthly_csv": str(monthly_path.relative_to(root)),
            "latent_factors_csv": str(factor_path.relative_to(root)),
            "integrated_gradients_csv": str(attr_path.relative_to(root)),
            "model": str(model_path.relative_to(root)),
            "figures": [str(path.relative_to(root)) for path in figures],
        },
    }
    write_json_atomic(dirs["manifests"] / "conditional_autoencoder_sdf_manifest.json", manifest)
    print("conditional_autoencoder_sdf_status=PASS")
    print(f"device={device} oos_months={manifest['oos_months']} pricing_rms={pricing['rms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
