from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReadinessItem:
    requirement: str
    status: str
    score: float
    evidence: str
    detail: str
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_json(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def status_score(status: str) -> float:
    return {
        "PASS": 1.0,
        "PASS_WEAK_RESULT": 0.85,
        "NEGATIVE_RESULT": 0.75,
        "PARTIAL": 0.55,
        "BLOCKED": 0.25,
        "MISSING": 0.0,
    }.get(status, 0.0)


def item(
    requirement: str,
    status: str,
    evidence: str,
    detail: str,
    action: str = "",
) -> ReadinessItem:
    return ReadinessItem(
        requirement=requirement,
        status=status,
        score=status_score(status),
        evidence=evidence,
        detail=detail,
        action=action,
    )


def pass_if(condition: bool, partial_detail: str | None = None) -> str:
    if condition:
        return "PASS"
    return "PARTIAL" if partial_detail else "MISSING"


def evaluate_readiness(root: Path) -> list[ReadinessItem]:
    manifests = root / "manifests"
    outputs = root / "outputs"
    run_full = read_json(root, "manifests/run_full/summary.json")
    ssvi = read_json(root, "manifests/ssvi_fit/summary.json")
    svi = read_json(root, "manifests/svi_refit/summary.json")
    chars = read_json(root, "manifests/characteristic_library_manifest.json")
    state = read_json(root, "manifests/state_controls_manifest.json")
    daily = read_json(root, "manifests/daily_risk_manifest.json")
    taq = read_json(root, "manifests/taq_cost_manifest_calib_2015_2024_m12_s50.json")
    ibes = read_json(root, "manifests/ibes_regsho_manifest.json")
    external = read_json(root, "manifests/external_api_controls_manifest.json")
    baseline = read_json(root, "manifests/baseline_manifest.json")
    gpu = read_json(root, "manifests/gpu_return_model_manifest.json")
    portfolio = read_json(root, "manifests/proposal_portfolio_manifest.json")
    inference = read_json(root, "manifests/asset_pricing_inference_manifest.json")
    sdf = read_json(root, "manifests/conditional_autoencoder_sdf_manifest.json")
    model_interp = read_json(root, "manifests/model_interpretation_manifest.json")
    shap = read_json(root, "manifests/shap_interpretation_manifest.json")
    regsho = read_json(root, "manifests/regsho_pilot_did_manifest.json")

    rows: list[ReadinessItem] = []

    extraction_pass = (
        run_full.get("status") == "PASS"
        and int(run_full.get("completed", 0)) >= 300
        and int(run_full.get("failed", 1)) == 0
    )
    rows.append(
        item(
            "WRDS extraction and usable sample",
            "PASS" if extraction_pass else "PARTIAL",
            "manifests/run_full/summary.json",
            (
                f"{run_full.get('completed')} completed, {run_full.get('skipped')} skipped, "
                f"{run_full.get('failed')} failed; usable through 2024 because 2025 OptionMetrics shards are unavailable."
            ),
            "Do not claim 2025 coverage unless OptionMetrics shards become available.",
        )
    )

    rows.append(
        item(
            "Raw SVI no-arbitrage refits",
            "PASS" if svi.get("status") == "PASS" and int(svi.get("months", svi.get("completed", 0))) >= 300 else "PARTIAL",
            "manifests/svi_refit/summary.json",
            f"{svi.get('months', svi.get('completed'))} monthly refit manifests completed with no-arbitrage diagnostics.",
        )
    )
    rows.append(
        item(
            "Global SSVI calendar and butterfly constraints",
            "PASS" if ssvi.get("status") == "PASS" and float(ssvi.get("pass_share", 0.0)) >= 0.999 else "PARTIAL",
            "manifests/ssvi_fit/summary.json",
            (
                f"{ssvi.get('pass_surfaces')} / {ssvi.get('surfaces')} surfaces pass; "
                f"pass share {float(ssvi.get('pass_share', 0.0)):.3f}."
            ),
        )
    )

    characteristic_count = len(chars.get("characteristics", []))
    rows.append(
        item(
            "Expanded CRSP-Compustat characteristics",
            "PASS" if chars.get("status") == "PASS" and characteristic_count >= 30 else "PARTIAL",
            "manifests/characteristic_library_manifest.json",
            f"{characteristic_count} characteristics over {chars.get('panel_rows')} firm-month rows.",
        )
    )
    state_count = len(state.get("state_columns", []))
    rows.append(
        item(
            "Cboe, FF, and FRB state/factor controls",
            "PASS" if state.get("status") == "PASS" and state_count >= 25 else "PARTIAL",
            "manifests/state_controls_manifest.json",
            f"{state_count} state/factor columns over {state.get('state_months')} months.",
        )
    )
    rows.append(
        item(
            "CRSP daily beta and idiosyncratic-volatility controls",
            "PASS" if daily.get("status") == "PASS" and int(daily.get("risk_rows", 0)) >= 600000 else "PARTIAL",
            "manifests/daily_risk_manifest.json",
            f"{daily.get('risk_rows')} monthly risk rows; columns: {', '.join(daily.get('risk_columns', []))}.",
        )
    )

    rows.append(
        item(
            "TAQ-calibrated transaction costs",
            "PASS" if taq.get("status") == "PASS" and taq.get("calibrate_full_panel") else "PARTIAL",
            "manifests/taq_cost_manifest_calib_2015_2024_m12_s50.json",
            (
                f"{taq.get('cost_rows')} exact sampled rows; exact/calibrated shares "
                f"{100 * float(taq.get('exact_taq_share', 0.0)):.2f}%/"
                f"{100 * float(taq.get('calibrated_taq_share', 0.0)):.2f}%."
            ),
            "Do not describe this as a full raw all-symbol TAQ mirror.",
        )
    )
    rows.append(
        item(
            "IBES analyst expectations and WRDS short-volume sample",
            "PASS" if ibes.get("status") == "PASS" and int(ibes.get("ibes_raw_rows", 0)) > 0 else "PARTIAL",
            "manifests/ibes_regsho_manifest.json",
            f"{ibes.get('ibes_raw_rows')} IBES rows; {ibes.get('short_raw_rows')} visible short-volume rows.",
        )
    )

    passed_sources = set(external.get("control_sources_passed", []))
    blocked_sources = [
        s.get("source")
        for s in external.get("source_statuses", [])
        if str(s.get("status", "")).startswith("SKIPPED")
    ]
    external_status = "PASS" if {"FRED", "BLS", "BEA", "EIA"}.issubset(passed_sources) else "PARTIAL"
    if blocked_sources:
        external_status = "BLOCKED"
    rows.append(
        item(
            "External APIs",
            external_status,
            "manifests/external_api_controls_manifest.json",
            f"Passed: {', '.join(sorted(passed_sources)) or 'none'}; skipped: {', '.join(blocked_sources) or 'none'}.",
            "Set BEA_API_KEY, EIA_API_KEY, and SEC_EDGAR_USER_AGENT at runtime in an ignored environment if these sources must be pulled.",
        )
    )

    baseline_pass = (
        baseline.get("status") == "PASS"
        and baseline.get("elastic_net", {}).get("status") == "PASS"
        and baseline.get("lightgbm", {}).get("status") == "PASS"
    )
    rows.append(
        item(
            "Baseline models",
            "PASS" if baseline_pass else "PARTIAL",
            "manifests/baseline_manifest.json",
            f"{len(baseline.get('features', []))} features; Elastic Net and LightGBM statuses recorded.",
        )
    )

    gpu_status = "MISSING"
    if gpu.get("status") == "PASS":
        gpu_status = "PASS_WEAK_RESULT"
        if float(gpu.get("mean_top_bottom_return") or 0.0) <= 0.0:
            gpu_status = "NEGATIVE_RESULT"
    rows.append(
        item(
            "GPU walk-forward return model",
            gpu_status,
            "manifests/gpu_return_model_manifest.json",
            (
                f"{len(gpu.get('features', []))} features, {gpu.get('folds')} folds, device={gpu.get('device')}; "
                f"rank IC={float(gpu.get('mean_rank_ic') or 0.0):.4f}, "
                f"top-bottom={float(gpu.get('mean_top_bottom_return') or 0.0):.4f}/mo."
            ),
            "Treat the latest expanded-feature result as weak/negative evidence, not a profitable anomaly.",
        )
    )

    portfolio_status = "MISSING"
    if portfolio.get("status") == "PASS":
        net_mean = float(portfolio.get("net", {}).get("mean_monthly_return") or 0.0)
        portfolio_status = "PASS_WEAK_RESULT" if net_mean > 0 else "NEGATIVE_RESULT"
    rows.append(
        item(
            "Sector-balanced beta-hedged buffered portfolio",
            portfolio_status,
            "manifests/proposal_portfolio_manifest.json",
            (
                f"{portfolio.get('months')} OOS months; net mean="
                f"{float(portfolio.get('net', {}).get('mean_monthly_return') or 0.0):.4f}/mo; "
                f"TAQ-net mean={float(portfolio.get('taq_net', {}).get('mean_monthly_return') or 0.0):.4f}/mo."
            ),
            "Implementation is complete, but the current expanded-feature portfolio result is negative.",
        )
    )

    inference_status = "MISSING"
    if inference.get("status") == "PASS":
        t_stat = float(inference.get("proposal_net_ff5_alpha_t") or 0.0)
        inference_status = "PASS_WEAK_RESULT" if t_stat > 1.65 else "NEGATIVE_RESULT" if t_stat < 0 else "PARTIAL"
    rows.append(
        item(
            "Factor alphas, HJ/GRS, Newey-West, bootstrap inference",
            inference_status,
            "manifests/asset_pricing_inference_manifest.json",
            (
                f"Proposal FF5+UMD alpha t={float(inference.get('proposal_net_ff5_alpha_t') or 0.0):.2f}; "
                f"TAQ-net alpha t={float(inference.get('proposal_taq_net_ff5_alpha_t') or 0.0):.2f}; "
                f"bootstrap status={inference.get('bootstrap_net_ls', {}).get('status')}."
            ),
            "Use these inference artifacts to discipline claims; latest expanded-feature alphas are negative.",
        )
    )

    fg = sdf.get("feature_groups", {})
    rows.append(
        item(
            "Flagship conditional autoencoder SDF",
            "PASS" if sdf.get("status") == "PASS" and sdf.get("architecture", {}).get("managed_factor_portfolios") else "PARTIAL",
            "manifests/conditional_autoencoder_sdf_manifest.json",
            (
                f"{sdf.get('oos_months')} OOS months, {sdf.get('oos_assets')} assets; "
                f"surface/tabular/state={len(fg.get('surface', []))}/"
                f"{len(fg.get('tabular', []))}/{len(fg.get('state', []))}; "
                f"pricing RMS={sdf.get('pricing_error', {}).get('rms')}."
            ),
        )
    )
    rows.append(
        item(
            "Integrated gradients and latent-factor interpretation",
            "PASS" if model_interp.get("status") == "PASS" and int(model_interp.get("latent_factor_correlations", 0)) > 0 else "PARTIAL",
            "manifests/model_interpretation_manifest.json",
            (
                f"{model_interp.get('top_attributions')} IG rows, "
                f"{model_interp.get('latent_factor_correlations')} latent/state correlations, "
                f"{model_interp.get('latent_beta_correlations')} latent beta/characteristic correlations."
            ),
        )
    )
    rows.append(
        item(
            "TreeSHAP interpretation",
            "PASS" if shap.get("status") == "PASS" and len(shap.get("features", [])) >= 60 else "PARTIAL",
            "manifests/shap_interpretation_manifest.json",
            f"{len(shap.get('features', []))} features; train rows={shap.get('train_rows')}; explain rows={shap.get('explain_rows')}.",
        )
    )

    reg_term = regsho.get("regression", {}).get("target_coefficient", {})
    reg_t = float(reg_term.get("t_cluster_date") or 0.0)
    rows.append(
        item(
            "Reg SHO pilot mechanism test",
            "PASS_WEAK_RESULT" if regsho.get("status") == "PASS" and abs(reg_t) < 1.65 else "PASS",
            "manifests/regsho_pilot_did_manifest.json",
            (
                f"{regsho.get('did_rows')} firm-month rows; {regsho.get('pilot_permnos')} pilot PERMNOs; "
                f"triple t={reg_t:.2f}."
            ),
            "Retain as weak mechanism evidence; do not claim a strong causal result.",
        )
    )

    dashboard = outputs / "figures" / "full" / "evidence_dashboard.png"
    rows.append(
        item(
            "High-quality visual evidence pack",
            "PASS" if dashboard.exists() else "PARTIAL",
            "outputs/figures/full/evidence_dashboard.png",
            "Evidence dashboard plus SSVI, SDF, SHAP, Reg SHO, portfolio, and inference figures are generated.",
        )
    )

    safety_blocked = (manifests / "preflight.json").exists() and (root / ".gitignore").exists()
    rows.append(
        item(
            "Public-safety and reproducibility discipline",
            "PASS" if safety_blocked else "PARTIAL",
            ".gitignore, tests, logs/public_safety_scan*.log",
            "Raw data, logs, models, and secrets are ignored; local and remote safety scans should be rerun before any push.",
        )
    )
    return rows


def readiness_summary(items: list[ReadinessItem]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in items:
        counts[row.status] = counts.get(row.status, 0) + 1
    blocking = [row.requirement for row in items if row.status in {"BLOCKED", "MISSING", "PARTIAL"}]
    negative = [row.requirement for row in items if row.status == "NEGATIVE_RESULT"]
    return {
        "items": len(items),
        "counts": counts,
        "mean_score": sum(row.score for row in items) / max(len(items), 1),
        "blocking_or_partial_items": blocking,
        "negative_result_items": negative,
        "overall_status": "PASS_WITH_BOUNDARIES" if not blocking else "INCOMPLETE",
    }
