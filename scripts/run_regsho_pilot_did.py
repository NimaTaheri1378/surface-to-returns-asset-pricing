from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs
from surface_returns.regsho import (
    event_study_surface_spreads,
    fit_regsho_did,
    load_category_a_pilot_list,
    prepare_regsho_did_frame,
)


def candidate_panel_paths(root: Path) -> list[Path]:
    return [
        root / "data" / "processed" / "panel" / "surface_characteristic_state_ibes_regsho_external_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_ibes_regsho_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_taq_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_daily_risk_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_state_panel.parquet",
        root / "data" / "processed" / "panel" / "surface_characteristic_panel.parquet",
    ]


def control_columns(panel: pd.DataFrame, window_start: str, window_end: str, min_coverage: float = 0.25) -> list[str]:
    candidates = [
        "log_market_equity",
        "book_to_market_comp",
        "momentum_12_2",
        "short_reversal",
        "beta_252d",
        "idio_vol_252d",
        "median_spread_pct",
        "turnover",
        "ibes_analyst_coverage",
        "ibes_forecast_dispersion",
        "regsho_short_share",
    ]
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.to_period("M").dt.to_timestamp()
    data = data[data["date"].between(pd.Timestamp(window_start), pd.Timestamp(window_end))]
    selected = []
    for col in candidates:
        if col not in data.columns:
            continue
        values = pd.to_numeric(data[col], errors="coerce")
        if values.notna().mean() >= min_coverage and values.nunique(dropna=True) > 2:
            selected.append(col)
    return selected


def coefficient_table(result) -> pd.DataFrame:
    rows = []
    for name, value in result.params.items():
        rows.append(
            {
                "term": name,
                "coefficient": value,
                "se_hc1": result.se_hc1.get(name),
                "t_hc1": result.tstat(name, "hc1"),
                "se_cluster_date": result.se_cluster_date.get(name),
                "t_cluster_date": result.tstat(name, "cluster_date"),
                "se_cluster_entity": result.se_cluster_entity.get(name),
                "t_cluster_entity": result.tstat(name, "cluster_entity"),
            }
        )
    return pd.DataFrame(rows)


def write_regsho_figure(coefs: pd.DataFrame, event: pd.DataFrame, output_prefix: Path) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    focus = coefs[coefs["term"].isin(["surface_signal", "signal_x_pilot", "signal_x_post", "signal_x_pilot_x_post"])]
    if not focus.empty:
        plot = focus.set_index("term").loc[
            ["surface_signal", "signal_x_pilot", "signal_x_post", "signal_x_pilot_x_post"]
        ].reset_index()
        axes[0].barh(plot["term"], plot["coefficient"], xerr=1.96 * plot["se_cluster_date"], color="#4c78a8")
        axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Reg SHO Pilot DID Coefficients")
    axes[0].set_xlabel("Coefficient with 95% date-cluster interval")
    axes[0].grid(True, axis="x", alpha=0.25)

    if not event.empty:
        pivot = event.pivot_table(index="event_month", columns="regsho_pilot", values="high_minus_low", aggfunc="mean")
        if 0 in pivot:
            axes[1].plot(pivot.index, pivot[0], label="Control", color="#f58518", linewidth=1.8)
        if 1 in pivot:
            axes[1].plot(pivot.index, pivot[1], label="Pilot", color="#54a24b", linewidth=1.8)
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].axvline(11, color="black", linewidth=0.8, linestyle="--")
        axes[1].legend(frameon=False)
    axes[1].set_title("Surface-Signal High Minus Low Returns")
    axes[1].set_xlabel("Months from May 2005 pilot start")
    axes[1].set_ylabel("Next-month return spread")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Reg SHO Pilot Mechanism Test")
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default=None)
    parser.add_argument("--signal-col", default="put_call_iv_spread")
    parser.add_argument("--return-col", default="next_ret")
    parser.add_argument("--window-start", default="2004-01-01")
    parser.add_argument("--window-end", default="2006-12-31")
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)

    panel_path = root / args.panel if args.panel else next((p for p in candidate_panel_paths(root) if p.exists()), candidate_panel_paths(root)[-1])
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    panel = pd.read_parquet(panel_path)
    pilot, source_url = load_category_a_pilot_list()
    controls = control_columns(panel, args.window_start, args.window_end)
    did = prepare_regsho_did_frame(
        panel,
        pilot,
        signal_col=args.signal_col,
        return_col=args.return_col,
        window_start=args.window_start,
        window_end=args.window_end,
        controls=controls,
    )
    result = fit_regsho_did(did, return_col=args.return_col, controls=controls)
    coefs = coefficient_table(result)
    event = event_study_surface_spreads(did, return_col=args.return_col)

    data_dir = root / "data" / "external"
    report_dir = root / "outputs" / "reports" / "regsho"
    figure_dir = root / "outputs" / "figures" / "full"
    for path in [data_dir, report_dir, figure_dir]:
        path.mkdir(parents=True, exist_ok=True)
    pilot_path = data_dir / "regsho_category_a_pilot_securities.csv"
    did_path = report_dir / "regsho_pilot_did_sample.parquet"
    coef_path = report_dir / "regsho_pilot_did_coefficients.csv"
    event_path = report_dir / "regsho_surface_event_spreads.csv"
    pilot.to_csv(pilot_path, index=False)
    did.to_parquet(did_path, index=False)
    coefs.to_csv(coef_path, index=False)
    event.to_csv(event_path, index=False)
    figures = write_regsho_figure(coefs, event, figure_dir / "regsho_pilot_did")

    triple = coefs[coefs["term"].eq("signal_x_pilot_x_post")]
    triple_row = triple.iloc[0].to_dict() if not triple.empty else {}
    manifest = {
        "status": "PASS",
        "panel": str(panel_path.relative_to(root)),
        "source_url": source_url,
        "pilot_category_a_count": int(len(pilot)),
        "signal_col": args.signal_col,
        "return_col": args.return_col,
        "window_start": args.window_start,
        "window_end": args.window_end,
        "controls": controls,
        "did_rows": int(len(did)),
        "pilot_rows": int(did["regsho_pilot"].sum()),
        "pilot_permnos": int(did.loc[did["regsho_pilot"].eq(1), "permno"].nunique()),
        "control_permnos": int(did.loc[did["regsho_pilot"].eq(0), "permno"].nunique()),
        "months": int(did["date"].nunique()),
        "regression": {
            "fixed_effects": ["permno", "date"],
            "nobs": result.nobs,
            "r2_within": result.r2,
            "target_coefficient": triple_row,
        },
        "artifacts": {
            "pilot_list": str(pilot_path.relative_to(root)),
            "did_sample": str(did_path.relative_to(root)),
            "coefficients": str(coef_path.relative_to(root)),
            "event_spreads": str(event_path.relative_to(root)),
            "figures": [str(path.relative_to(root)) for path in figures],
        },
    }
    write_json_atomic(dirs["manifests"] / "regsho_pilot_did_manifest.json", manifest)
    print("regsho_pilot_did_status=PASS")
    print(
        f"rows={manifest['did_rows']} pilot_permnos={manifest['pilot_permnos']} "
        f"triple_coef={triple_row.get('coefficient')} t_date={triple_row.get('t_cluster_date')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
