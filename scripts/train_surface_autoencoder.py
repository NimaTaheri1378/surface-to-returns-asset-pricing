from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs


def load_grid(root: Path, max_rows: int = 0) -> pd.DataFrame:
    paths = sorted(root.glob("data/processed/full/surface_grid/year=*/month=*.parquet"))
    if not paths:
        raise FileNotFoundError("No full surface-grid shards found.")
    frames = []
    rows = 0
    for path in paths:
        frame = pd.read_parquet(path)
        frames.append(frame)
        rows += len(frame)
        if max_rows and rows >= max_rows:
            break
    grid = pd.concat(frames, ignore_index=True)
    if max_rows:
        grid = grid.head(max_rows)
    return grid


def pivot_grid(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = grid.copy()
    grid["date"] = pd.to_datetime(grid["date"])
    grid["cell"] = (
        "dte"
        + grid["target_dte"].astype(int).astype(str)
        + "_delta"
        + (grid["target_abs_delta"] * 100).round().astype(int).astype(str)
    )
    index_cols = ["date", "secid"]
    if "permno" in grid.columns:
        index_cols.append("permno")
    wide = (
        grid.pivot_table(index=index_cols, columns="cell", values="impl_volatility", aggfunc="mean")
        .sort_index()
        .reset_index()
    )
    meta = wide[index_cols].copy()
    x = wide.drop(columns=index_cols)
    x = x.dropna(axis=0, thresh=max(5, int(x.shape[1] * 0.6)))
    meta = meta.loc[x.index].reset_index(drop=True)
    x = x.reset_index(drop=True)
    x = x.apply(lambda col: col.fillna(col.median()), axis=0)
    return meta, x


def train_autoencoder(x: pd.DataFrame, latent_dim: int, epochs: int, batch_size: int, seed: int):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    values = x.to_numpy(dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    scaled = (values - mean) / std
    tensor = torch.tensor(scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = tensor.shape[1]
    hidden = max(32, min(128, input_dim * 4))
    model = nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.GELU(),
        nn.Dropout(0.05),
        nn.Linear(hidden, latent_dim),
        nn.GELU(),
        nn.Linear(latent_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, input_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    losses = []
    for _epoch in range(epochs):
        epoch_losses = []
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    encoder = nn.Sequential(*list(model.children())[:4]).to(device)
    with torch.no_grad():
        z = encoder(tensor.to(device)).detach().cpu().numpy()
        recon = model(tensor.to(device)).detach().cpu().numpy()
    return model, z, recon * std + mean, losses, device.type


def write_reconstruction_figure(x: pd.DataFrame, recon: np.ndarray, losses: list[float], output_prefix: Path) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    actual = x.iloc[0].to_numpy(dtype=float)
    fitted = recon[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(losses, color="#4c78a8", linewidth=1.8)
    axes[0].set_title("Autoencoder Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(actual, label="Actual", color="#1f77b4", linewidth=1.8)
    axes[1].plot(fitted, label="Reconstruction", color="#ff7f0e", linewidth=1.5)
    axes[1].set_title("First Surface Grid Reconstruction")
    axes[1].set_xlabel("Grid cell")
    axes[1].set_ylabel("Implied volatility")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1378)
    parser.add_argument("--max-grid-rows", type=int, default=0)
    args = parser.parse_args()
    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    grid = load_grid(root, max_rows=args.max_grid_rows)
    meta, x = pivot_grid(grid)
    model, z, recon, losses, device = train_autoencoder(x, args.latent_dim, args.epochs, args.batch_size, args.seed)
    embeddings = meta.copy()
    for idx in range(z.shape[1]):
        embeddings[f"surface_ae_{idx + 1:02d}"] = z[:, idx]
    emb_path = root / "data" / "processed" / "panel" / "surface_autoencoder_embeddings.parquet"
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings.to_parquet(emb_path, index=False)
    model_path = root / "outputs" / "models" / "surface_autoencoder.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save({"model_state_dict": model.state_dict(), "columns": list(x.columns), "losses": losses}, model_path)
    figures = write_reconstruction_figure(x, recon, losses, root / "outputs" / "figures" / "full" / "surface_autoencoder_reconstruction")
    manifest = {
        "status": "PASS",
        "device": device,
        "input_rows": int(len(x)),
        "input_dim": int(x.shape[1]),
        "latent_dim": int(args.latent_dim),
        "epochs": int(args.epochs),
        "final_loss": float(losses[-1]) if losses else None,
        "artifacts": {
            "embeddings": str(emb_path.relative_to(root)),
            "model": str(model_path.relative_to(root)),
            "figures": [str(path.relative_to(root)) for path in figures],
        },
    }
    write_json_atomic(dirs["manifests"] / "surface_autoencoder_manifest.json", manifest)
    print("surface_autoencoder_status=PASS")
    print(f"input_rows={manifest['input_rows']} final_loss={manifest['final_loss']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
