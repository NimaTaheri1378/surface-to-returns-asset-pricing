from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def load_panel(root: Path, explicit_panel: str | None = None) -> tuple[pd.DataFrame, Path]:
    path = root / explicit_panel if explicit_panel else next((item for item in candidate_panel_paths(root) if item.exists()), candidate_panel_paths(root)[-1])
    if not path.exists():
        raise FileNotFoundError(f"Missing feature panel: {path}")
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel, path


def select_features(frame: pd.DataFrame, min_nonmissing: float) -> list[str]:
    candidates = list(DEFAULT_FEATURE_CANDIDATES)
    candidates.extend(["log_n_contracts"])
    selected = []
    for col in dict.fromkeys(candidates):
        if col not in frame:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().mean() >= min_nonmissing and values.nunique(dropna=True) > 2:
            selected.append(col)
    return selected


def prepare_model_frame(panel: pd.DataFrame, min_nonmissing: float) -> pd.DataFrame:
    frame = panel.copy()
    if "market_equity" in frame:
        frame["log_market_equity"] = np.log(pd.to_numeric(frame["market_equity"], errors="coerce").clip(lower=1))
    if "n_contracts" in frame:
        frame["log_n_contracts"] = np.log1p(pd.to_numeric(frame["n_contracts"], errors="coerce").clip(lower=0))
    feature_cols = select_features(frame, min_nonmissing=min_nonmissing)
    frame = frame[["date", "permno", "next_ret", *feature_cols]].copy()
    frame[feature_cols] = frame[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    frame["next_ret"] = pd.to_numeric(frame["next_ret"], errors="coerce").astype(float)
    for col in feature_cols:
        frame[col] = frame.groupby("date")[col].transform(lambda s: s.fillna(s.median()))
        frame[col] = frame[col].fillna(frame[col].median())
        def _scale(s: pd.Series) -> pd.Series:
            std = s.std(ddof=0)
            if not np.isfinite(std) or std == 0:
                return pd.Series(0.0, index=s.index)
            return (s - s.mean()) / std

        frame[col] = frame.groupby("date")[col].transform(_scale)
    return frame.dropna(subset=["next_ret"] + feature_cols), feature_cols


def fama_macbeth(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("date"):
        if len(group) <= len(feature_cols) + 5:
            continue
        y = group["next_ret"].astype(float)
        x = sm.add_constant(group[feature_cols].astype(float), has_constant="add")
        try:
            result = sm.OLS(y, x, missing="drop").fit()
        except Exception:
            continue
        row = {"date": date}
        row.update(result.params.to_dict())
        rows.append(row)
    coefs = pd.DataFrame(rows)
    if coefs.empty:
        return coefs
    out = []
    for col in [c for c in coefs.columns if c != "date"]:
        series = coefs[col].dropna()
        out.append(
            {
                "term": col,
                "mean_coef": float(series.mean()),
                "t_stat_naive": float(series.mean() / (series.std(ddof=1) / np.sqrt(len(series)))) if len(series) > 1 and series.std(ddof=1) > 0 else np.nan,
                "months": int(len(series)),
            }
        )
    return pd.DataFrame(out)


def elastic_net_oos(frame: pd.DataFrame, feature_cols: list[str]) -> dict[str, float | int]:
    monthly_dates = sorted(frame["date"].dropna().unique())
    if len(monthly_dates) < 24:
        return {"status": "SKIPPED_TOO_FEW_MONTHS", "months": len(monthly_dates)}
    date_index = pd.Series(range(len(monthly_dates)), index=monthly_dates)
    fold_id = frame["date"].map(date_index)
    x = frame[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y = frame["next_ret"].to_numpy(dtype=float)
    splits = TimeSeriesSplit(n_splits=min(5, max(2, len(monthly_dates) // 24)))
    preds = np.full_like(y, fill_value=np.nan, dtype=float)
    for train_month_idx, test_month_idx in splits.split(monthly_dates):
        train_dates = set(np.array(monthly_dates, dtype=object)[train_month_idx])
        test_dates = set(np.array(monthly_dates, dtype=object)[test_month_idx])
        train_mask = frame["date"].isin(train_dates).to_numpy()
        test_mask = frame["date"].isin(test_dates).to_numpy()
        model = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=5000))
        model.fit(x[train_mask], y[train_mask])
        preds[test_mask] = model.predict(x[test_mask])
    mask = ~np.isnan(preds)
    return {
        "status": "PASS",
        "observations": int(mask.sum()),
        "oos_r2": float(r2_score(y[mask], preds[mask])) if mask.any() else np.nan,
        "rank_ic": float(pd.Series(preds[mask]).corr(pd.Series(y[mask]), method="spearman")) if mask.any() else np.nan,
    }


def walk_forward_folds(monthly_dates: list[pd.Timestamp], min_train_months: int = 120, test_months: int = 12):
    start = min_train_months
    while start < len(monthly_dates):
        end = min(start + test_months, len(monthly_dates))
        yield monthly_dates[:start], monthly_dates[start:end]
        start = end


def lightgbm_oos(frame: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> dict[str, object]:
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:
        return {"status": "SKIPPED_LIGHTGBM_UNAVAILABLE", "reason": type(exc).__name__}

    monthly_dates = sorted(frame["date"].dropna().unique())
    folds = list(walk_forward_folds(monthly_dates))
    if not folds:
        return {"status": "SKIPPED_TOO_FEW_MONTHS", "months": len(monthly_dates)}

    all_preds = []
    for fold_id, (train_dates, test_dates) in enumerate(folds, start=1):
        train = frame[frame["date"].isin(train_dates)]
        test = frame[frame["date"].isin(test_dates)]
        if train.empty or test.empty:
            continue
        model = LGBMRegressor(
            objective="regression",
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.10,
            random_state=1378 + fold_id,
            n_jobs=4,
            verbose=-1,
        )
        model.fit(train[feature_cols], train["next_ret"])
        pred = model.predict(test[feature_cols])
        out = test[["date", "permno", "next_ret"]].copy()
        out["pred"] = pred
        out["fold"] = fold_id
        all_preds.append(out)
    if not all_preds:
        return {"status": "FAILED_NO_PREDICTIONS"}

    predictions = pd.concat(all_preds, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "lightgbm_oos_predictions.parquet"
    predictions.to_parquet(pred_path, index=False)
    rank_ic = []
    spread = []
    for _date, group in predictions.groupby("date"):
        if group["pred"].nunique() <= 1:
            continue
        ranks = group["pred"].rank(pct=True)
        rank_ic.append(group["pred"].corr(group["next_ret"], method="spearman"))
        spread.append(group.loc[ranks >= 0.9, "next_ret"].mean() - group.loc[ranks <= 0.1, "next_ret"].mean())
    return {
        "status": "PASS",
        "observations": int(len(predictions)),
        "folds": int(predictions["fold"].nunique()),
        "oos_r2": float(r2_score(predictions["next_ret"], predictions["pred"])),
        "mean_rank_ic": float(pd.Series(rank_ic).dropna().mean()) if rank_ic else np.nan,
        "mean_top_bottom_return": float(pd.Series(spread).dropna().mean()) if spread else np.nan,
        "predictions": str(pred_path),
    }


def write_fm_figure(fm: pd.DataFrame, output_path: Path) -> list[Path]:
    if fm.empty:
        return []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = fm[~fm["term"].eq("const")].copy()
    if plot_frame.empty:
        plot_frame = fm.copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = plot_frame.sort_values("mean_coef")
    ax.barh(ordered["term"], ordered["mean_coef"], color="#4c78a8")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Average monthly slope")
    ax.set_title("Full-Sample Fama-MacBeth Slopes")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    paths = [output_path.with_suffix(".png"), output_path.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--panel", default=None)
    parser.add_argument("--min-feature-nonmissing", type=float, default=0.65)
    args = parser.parse_args()
    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    panel, panel_path = load_panel(root, args.panel)
    frame, feature_cols = prepare_model_frame(panel, args.min_feature_nonmissing)
    fm = fama_macbeth(frame, feature_cols)
    results_dir = root / "outputs" / "reports" / "baselines"
    results_dir.mkdir(parents=True, exist_ok=True)
    fm_path = results_dir / "fama_macbeth_surface_slopes.csv"
    fm.to_csv(fm_path, index=False)
    enet = elastic_net_oos(frame, feature_cols)
    lgbm = lightgbm_oos(frame, feature_cols, results_dir)
    figures = write_fm_figure(fm, root / "outputs" / "figures" / "full" / "fama_macbeth_slopes")
    manifest = {
        "status": "PASS" if not fm.empty else "PARTIAL",
        "observations": int(len(frame)),
        "months": int(frame["date"].nunique()) if not frame.empty else 0,
        "panel": str(panel_path.relative_to(root)),
        "features": feature_cols,
        "elastic_net": enet,
        "lightgbm": {
            **{key: value for key, value in lgbm.items() if key != "predictions"},
            "predictions": str(Path(lgbm["predictions"]).relative_to(root)) if "predictions" in lgbm else None,
        },
        "artifacts": {
            "fama_macbeth_csv": str(fm_path.relative_to(root)),
            "figures": [str(path.relative_to(root)) for path in figures],
        },
    }
    write_json_atomic(dirs["manifests"] / "baseline_manifest.json", manifest)
    print(f"baseline_status={manifest['status']}")
    return 0 if manifest["status"] in {"PASS", "PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
