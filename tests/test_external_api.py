from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from surface_returns import external_api
from scripts.build_external_api_controls import lag_for_availability, merge_state_tables, panel_output_path


def test_load_env_file_strips_quotes_and_does_not_override(tmp_path, monkeypatch):
    path = tmp_path / ".env.local"
    path.write_text("BLS_API_KEY='from-file'\nEIA_API_KEY=\"abc123\"\n# ignored\n", encoding="utf-8")
    monkeypatch.setenv("BLS_API_KEY", "already-set")

    external_api.load_env_file(path)

    assert os.environ["BLS_API_KEY"] == "already-set"
    assert os.environ["EIA_API_KEY"] == "abc123"


def test_year_chunks_respects_end_year():
    assert external_api._year_chunks(1996, 2024, width=10) == [(1996, 2005), (2006, 2015), (2016, 2024)]


def test_fetch_fred_series_prefers_official_api(monkeypatch):
    calls = []

    def fake_urlopen_json(url: str, **_kwargs):
        calls.append(url)
        assert "api_key=fred-key" in url
        assert "series_id=UNRATE" in url
        return {
            "observations": [
                {"date": "2020-01-01", "value": "3.5"},
                {"date": "2020-02-01", "value": "."},
            ]
        }

    monkeypatch.setattr(external_api, "_urlopen_json", fake_urlopen_json)

    frame, route = external_api.fetch_fred_series("UNRATE", "2020-01-01", "2020-12-31", api_key="fred-key")

    assert route == "official API"
    assert calls
    assert frame["date"].tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    assert frame["UNRATE"].tolist()[0] == 3.5
    assert pd.isna(frame["UNRATE"].tolist()[1])


def test_fetch_fred_series_falls_back_without_leaking_key(monkeypatch):
    hidden_key = "secret-fred-key"

    def fake_urlopen_json(*_args, **_kwargs):
        raise RuntimeError(f"bad key {hidden_key}")

    def fake_csv(series_id: str, _start: str, _end: str):
        return pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], series_id: [3.5]})

    monkeypatch.setattr(external_api, "_urlopen_json", fake_urlopen_json)
    monkeypatch.setattr(external_api, "fetch_fred_series_csv", fake_csv)
    monkeypatch.setenv("FRED_API_KEY", hidden_key)

    frame, route = external_api.fetch_fred_series("UNRATE", "2020-01-01", "2020-12-31", api_key=hidden_key)

    assert route == "fredgraph CSV"
    assert frame.loc[0, "UNRATE"] == 3.5


def test_fetch_eia_controls_v2_parses_monthly_spot(monkeypatch):
    def fake_urlopen_json(url: str, **_kwargs):
        assert "api_key=eia-key" in url
        assert "petroleum%2Fpri%2Fspt" not in url
        return {
            "response": {
                "data": [
                    {"period": "2020-01", "value": "57.52"},
                    {"period": "2020-02", "value": "50.54"},
                ]
            }
        }

    monkeypatch.setattr(external_api, "_urlopen_json", fake_urlopen_json)

    frame, status = external_api.fetch_eia_controls_v2(2020, 2020, "eia-key")

    assert status.status == "PASS"
    assert status.rows == 2
    assert frame["month"].tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    assert frame["eia_wti_spot"].tolist() == [57.52, 50.54]
    assert pd.isna(frame.loc[0, "eia_wti_spot_return"])
    assert frame.loc[1, "eia_wti_spot_return"] < 0


def test_lag_for_availability_moves_observation_to_signal_month():
    raw = pd.DataFrame(
        {
            "month": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "fred_unrate": [3.5, 3.6],
        }
    )

    out = lag_for_availability(raw, lag_months=1, start_year=2020, end_year=2020)

    assert out["month"].tolist() == [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01")]
    assert out["fred_unrate"].tolist() == [3.5, 3.6]


def test_merge_state_tables_renames_overlapping_external_columns():
    base = pd.DataFrame({"month": [pd.Timestamp("2020-01-01")], "mktrf": [0.01]})
    external = pd.DataFrame({"month": [pd.Timestamp("2020-01-01")], "mktrf": [0.02], "fred_unrate": [3.5]})

    out = merge_state_tables(base, external)

    assert "mktrf" in out
    assert "external_mktrf" in out
    assert out.loc[0, "fred_unrate"] == 3.5


def test_panel_output_path_keeps_external_panel_suffix():
    path = panel_output_path(Path("data/processed/panel/surface_characteristic_state_ibes_regsho_panel.parquet"))

    assert path.name == "surface_characteristic_state_ibes_regsho_external_panel.parquet"


def test_build_external_controls_sanitizes_failure_details(monkeypatch):
    hidden_value = "hiddenblsvalue"

    def fake_fred(_start: str, _end: str):
        frame = pd.DataFrame({"month": [pd.Timestamp("2020-01-01")], "fred_unrate": [3.5]})
        return frame, external_api.SourceStatus("FRED", "PASS", rows=1)

    def fake_bls(*_args, **_kwargs):
        raise RuntimeError(f"bad credential {hidden_value}")

    monkeypatch.setattr(external_api, "fetch_fred_controls", fake_fred)
    monkeypatch.setattr(external_api, "fetch_bls_controls", fake_bls)
    monkeypatch.setenv("BLS_API_KEY", hidden_value)
    monkeypatch.delenv("BEA_API_KEY", raising=False)
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)

    controls, statuses = external_api.build_external_controls(2020, 2020)

    assert len(controls) == 1
    bls = next(status for status in statuses if status["source"] == "BLS")
    assert bls["status"] == "FAILED"
    assert hidden_value not in str(bls["detail"])
    assert "<redacted>" in str(bls["detail"])
