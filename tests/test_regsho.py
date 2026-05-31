import numpy as np
import pandas as pd

from surface_returns.regsho import (
    fit_regsho_did,
    parse_appendix_a_from_order_html,
    parse_category_a_pilot_text,
    prepare_regsho_did_frame,
)


def test_parse_category_a_pipe_text():
    text = "Symbol|Security_Name|Exchange\nABC|Example Corp|NYSE\nXYZ|Other Inc|NNM\n"
    out = parse_category_a_pilot_text(text)

    assert out["ticker"].tolist() == ["ABC", "XYZ"]
    assert out["pilot_category"].eq("A").all()


def test_parse_appendix_a_html_fallback():
    html = "<html><body><h2>APPENDIX A</h2><p>Ticker</p><p>Symbol Company Name</p><p>ABC EXAMPLE CORP</p><p>XYZ OTHER INC</p><p>17 CFR 240.10a-1.</p></body></html>"
    out = parse_appendix_a_from_order_html(html)

    assert out["ticker"].tolist() == ["ABC", "XYZ"]


def test_regsho_did_recovers_negative_triple_effect():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2004-01-31", periods=36, freq="ME")
    pilot_permnos = list(range(100, 120))
    control_permnos = list(range(200, 240))
    rows = []
    for permno in pilot_permnos + control_permnos:
        pilot = permno in pilot_permnos
        ticker = f"P{permno}" if pilot else f"C{permno}"
        firm_alpha = rng.normal(0, 0.01)
        for date in dates:
            signal = rng.normal()
            post = pd.Timestamp("2005-05-01") <= date.to_period("M").to_timestamp() <= pd.Timestamp("2006-04-01")
            effect = -0.03 * signal * float(pilot) * float(post)
            rows.append(
                {
                    "date": date,
                    "permno": permno,
                    "ticker": ticker,
                    "put_call_iv_spread": signal,
                    "next_ret": firm_alpha + 0.01 * signal + effect + rng.normal(0, 0.01),
                    "log_market_equity": rng.normal(),
                }
            )
    panel = pd.DataFrame(rows)
    pilot_list = pd.DataFrame({"ticker": [f"P{permno}" for permno in pilot_permnos]})

    did = prepare_regsho_did_frame(panel, pilot_list, controls=["log_market_equity"])
    result = fit_regsho_did(did, controls=["log_market_equity"])

    assert result.nobs == len(did)
    assert result.params["signal_x_pilot_x_post"] < -0.015
    assert result.tstat("signal_x_pilot_x_post", "cluster_date") < -2.0
