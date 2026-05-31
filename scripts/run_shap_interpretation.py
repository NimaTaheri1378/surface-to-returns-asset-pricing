from __future__ import annotations

import argparse
import importlib.metadata
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


def sample_panel(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    rng = np.random.default_rng(seed)
    dates = sorted(frame["date"].dropna().unique())
    per_month = max(1, max_rows // max(len(dates), 1))
    sampled = []
    for date, group in frame.groupby("date", sort=True):
        take = min(len(group), per_month)
        if take:
            sampled.append(group.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    out = pd.concat(sampled) if sampled else frame.head(0).copy()
    if len(out) < max_rows and len(out) < len(frame):
        extra = frame.drop(index=out.index, errors="ignore")
        take = min(max_rows - len(out), len(extra))
        if take > 0:
            out = pd.concat(
                [out, extra.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))],
            )
    return out.sample(frac=1.0, random_state=seed).head(max_rows).reset_index(drop=True)


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


def load_frame(root: Path, explicit_panel: str | None, min_nonmissing: float) -> tuple[pd.DataFrame, list[str], Path]:
    panel_path = root / explicit_panel if explicit_panel else next(
        (path for path in candidate_panel_paths(root) if path.exists()),
        candidate_panel_paths(root)[-1],
    )
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing SHAP panel: {panel_path}")
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.to_period("M").dt.to_timestamp()
    emb_path = root / "data" / "processed" / "panel" / "surface_autoencoder_embeddings.parquet"
    if emb_path.exists():
        emb = pd.read_parquet(emb_path)
        emb["date"] = pd.to_datetime(emb["date"]).dt.to_period("M").dt.to_timestamp()
        keys = [col for col in ["date", "secid", "permno"] if col in panel.columns and col in emb.columns]
        emb_cols = keys + [col for col in emb.columns if col.startswith("surface_ae_")]
        panel = panel.merge(emb[emb_cols].drop_duplicates(keys), on=keys, how="left")
    if "market_equity" in panel:
        panel["log_market_equity"] = np.log(pd.to_numeric(panel["market_equity"], errors="coerce").clip(lower=1))
    if "n_contracts" in panel:
        panel["log_n_contracts"] = np.log1p(pd.to_numeric(panel["n_contracts"], errors="coerce").clip(lower=0))
    feature_cols = select_features(panel, min_nonmissing=min_nonmissing)
    if not feature_cols:
        raise RuntimeError("No eligible SHAP features found.")
    keep = ["date", "permno", "next_ret", *feature_cols]
    frame = panel[keep].replace([np.inf, -np.inf], np.nan).dropna(subset=["next_ret"])
    frame["next_ret"] = pd.to_numeric(frame["next_ret"], errors="coerce").astype(float)
    frame = standardize_by_month(frame, feature_cols)
    frame[feature_cols] = frame[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    frame = frame.dropna(subset=["next_ret", *feature_cols])
    return frame.sort_values(["date", "permno"]).reset_index(drop=True), feature_cols, panel_path


def write_shap_figure(summary: pd.DataFrame, shap_frame: pd.DataFrame, output_prefix: Path) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    plot = summary.head(20).sort_values("mean_abs_shap")
    axes[0].barh(plot["feature"], plot["mean_abs_shap"], color="#4c78a8")
    axes[0].set_title("SHAP Importance")
    axes[0].set_xlabel("Mean absolute SHAP value")
    axes[0].grid(True, axis="x", alpha=0.25)

    if not summary.empty:
        top_feature = str(summary.iloc[0]["feature"])
        sample = shap_frame[[top_feature, f"shap_{top_feature}"]].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=1378)
        axes[1].scatter(sample[top_feature], sample[f"shap_{top_feature}"], s=8, alpha=0.25, color="#f58518")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title(f"Dependence: {top_feature}")
        axes[1].set_xlabel("Standardized feature")
        axes[1].set_ylabel("SHAP value")
        axes[1].grid(True, alpha=0.25)
    fig.suptitle("TreeSHAP Return Model Interpretation")
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default=None)
    parser.add_argument("--min-feature-nonmissing", type=float, default=0.65)
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--max-train-rows", type=int, default=250_000)
    parser.add_argument("--max-explain-rows", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=1378)
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)

    from lightgbm import LGBMRegressor
    import shap

    frame, feature_cols, panel_path = load_frame(root, args.panel, args.min_feature_nonmissing)
    train_end = pd.Timestamp(args.train_end).to_period("M").to_timestamp()
    train = frame[frame["date"] <= train_end]
    explain = frame[frame["date"] > train_end]
    if train.empty or explain.empty:
        train = frame.iloc[: int(len(frame) * 0.7)]
        explain = frame.iloc[int(len(frame) * 0.7) :]
    train = sample_panel(train, args.max_train_rows, args.seed)
    explain = sample_panel(explain, args.max_explain_rows, args.seed + 1)
    model = LGBMRegressor(
        objective="regression",
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.10,
        random_state=args.seed,
        n_jobs=8,
        verbose=-1,
    )
    model.fit(train[feature_cols], train["next_ret"])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(explain[feature_cols])
    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 0]

    summary_rows = []
    for idx, feature in enumerate(feature_cols):
        col_values = values[:, idx]
        summary_rows.append(
            {
                "feature": feature,
                "mean_shap": float(np.nanmean(col_values)),
                "mean_abs_shap": float(np.nanmean(np.abs(col_values))),
                "shap_std": float(np.nanstd(col_values)),
            }
        )
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_frame = explain[["date", "permno", *feature_cols]].reset_index(drop=True)
    shap_cols = pd.DataFrame(values, columns=[f"shap_{col}" for col in feature_cols])
    shap_frame = pd.concat([shap_frame, shap_cols], axis=1)

    report_dir = root / "outputs" / "reports" / "interpretation"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "tree_shap_feature_importance.csv"
    values_path = report_dir / "tree_shap_values_sample.parquet"
    summary.to_csv(summary_path, index=False)
    shap_frame.to_parquet(values_path, index=False)
    figures = write_shap_figure(summary, shap_frame, root / "outputs" / "figures" / "full" / "tree_shap_interpretation")

    manifest = {
        "status": "PASS",
        "panel": str(panel_path.relative_to(root)),
        "features": feature_cols,
        "train_rows": int(len(train)),
        "explain_rows": int(len(explain)),
        "train_end": str(train_end.date()),
        "shap_version": importlib.metadata.version("shap"),
        "lightgbm_version": importlib.metadata.version("lightgbm"),
        "top_features": summary.head(20).to_dict(orient="records"),
        "artifacts": {
            "summary_csv": str(summary_path.relative_to(root)),
            "values_sample": str(values_path.relative_to(root)),
            "figures": [str(path.relative_to(root)) for path in figures],
        },
    }
    write_json_atomic(dirs["manifests"] / "shap_interpretation_manifest.json", manifest)
    print("shap_interpretation_status=PASS")
    print(f"features={len(feature_cols)} train_rows={len(train)} explain_rows={len(explain)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
