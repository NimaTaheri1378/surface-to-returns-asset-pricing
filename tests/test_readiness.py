from __future__ import annotations

import json
from pathlib import Path

from surface_returns.readiness import evaluate_readiness, readiness_summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_readiness_marks_external_env_gap_as_blocked(tmp_path):
    write_json(
        tmp_path / "manifests" / "external_api_controls_manifest.json",
        {
            "status": "PASS",
            "control_sources_passed": ["FRED", "BLS"],
            "source_statuses": [
                {"source": "FRED", "status": "PASS"},
                {"source": "BLS", "status": "PASS"},
                {"source": "BEA", "status": "SKIPPED_MISSING_ENV"},
                {"source": "EIA", "status": "SKIPPED_MISSING_ENV"},
                {"source": "SEC_EDGAR", "status": "SKIPPED_MISSING_ENV"},
            ],
        },
    )

    items = evaluate_readiness(tmp_path)
    external = next(row for row in items if row.requirement == "External APIs")

    assert external.status == "BLOCKED"
    assert "BEA" in external.detail


def test_readiness_keeps_negative_portfolio_as_verified_negative_result(tmp_path):
    write_json(
        tmp_path / "manifests" / "proposal_portfolio_manifest.json",
        {
            "status": "PASS",
            "months": 120,
            "net": {"mean_monthly_return": -0.001},
            "taq_net": {"mean_monthly_return": -0.002},
        },
    )
    write_json(
        tmp_path / "manifests" / "asset_pricing_inference_manifest.json",
        {
            "status": "PASS",
            "proposal_net_ff5_alpha_t": -1.9,
            "proposal_taq_net_ff5_alpha_t": -3.0,
            "bootstrap_net_ls": {"status": "PASS"},
        },
    )

    items = evaluate_readiness(tmp_path)
    summary = readiness_summary(items)

    assert "Sector-balanced beta-hedged buffered portfolio" in summary["negative_result_items"]
    assert "Factor alphas, HJ/GRS, Newey-West, bootstrap inference" in summary["negative_result_items"]


def test_readiness_characteristic_threshold(tmp_path):
    write_json(
        tmp_path / "manifests" / "characteristic_library_manifest.json",
        {"status": "PASS", "panel_rows": 100, "characteristics": [f"c{i}" for i in range(33)]},
    )

    item = next(row for row in evaluate_readiness(tmp_path) if row.requirement == "Expanded CRSP-Compustat characteristics")

    assert item.status == "PASS"
    assert item.score == 1.0
