from __future__ import annotations

import pandas as pd


def accounting_available_date(datadate: pd.Timestamp, lag_months: int = 6) -> pd.Timestamp:
    return pd.Timestamp(datadate) + pd.DateOffset(months=lag_months)


def is_accounting_known_by(datadate: pd.Timestamp, signal_date: pd.Timestamp, lag_months: int = 6) -> bool:
    return accounting_available_date(datadate, lag_months) <= pd.Timestamp(signal_date)


def next_month_return_window(signal_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    signal = pd.Timestamp(signal_date)
    start = signal + pd.offsets.MonthBegin(1)
    end = start + pd.offsets.MonthEnd(0)
    return pd.Timestamp(start), pd.Timestamp(end)
