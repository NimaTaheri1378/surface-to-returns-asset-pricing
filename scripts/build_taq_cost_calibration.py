from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from surface_returns.manifest import write_json_atomic
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job, ensure_project_dirs
from surface_returns.trading_costs import (
    aggregate_taq_daily_costs,
    fill_taq_costs_with_calibration,
    merge_taq_costs_to_panel,
    standardize_ticker,
)
from surface_returns.wrds_helpers import TableRef, connect_wrds, describe_columns, list_tables_safe, validate_identifier


TAQ_LIBRARIES = ["taqmsec", "taqmsamp", "taq", "taqmsec_nbbo", "taqmsamp_nbbo"]
QUOTE_PREFIXES = ["cq", "cqm", "nbbo", "quote", "quotes"]
TRADE_PREFIXES = ["ct", "ctm", "trade", "trades"]
SYMBOL_COLUMNS = ["sym_root", "symbol", "ticker", "sym", "root"]
QUOTE_COLUMNS = ["bid", "bid_price", "best_bid", "bbid", "ofr", "ask", "offer", "ask_price", "best_ask", "bask"]
TRADE_COLUMNS = ["price", "tr_scond_price", "trade_price", "prc", "size", "tr_siz", "trade_size", "volume", "shares"]


def safe_suffix(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    if text and not text.startswith("_"):
        text = "_" + text
    return text


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def safe_string_literals(values: list[str]) -> str:
    clean = []
    for value in values:
        ticker = standardize_ticker(value)
        if ticker:
            clean.append("'" + ticker.replace("'", "''") + "'")
    if not clean:
        return "''"
    return ", ".join(sorted(set(clean)))


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    colset = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in colset:
            return colset[candidate.lower()]
    return None


def find_taq_table(conn, trade_date: pd.Timestamp, kind: str) -> tuple[TableRef | None, list[str]]:
    ymd = trade_date.strftime("%Y%m%d").lower()
    prefixes = QUOTE_PREFIXES if kind == "quote" else TRADE_PREFIXES
    for library in TAQ_LIBRARIES:
        tables = list_tables_safe(conn, library)
        lower_to_actual = {table.lower(): table for table in tables}
        candidates: list[str] = []
        for prefix in prefixes:
            for pattern in [f"{prefix}_{ymd}", f"{prefix}{ymd}", f"{prefix}_{ymd[-6:]}", f"{prefix}{ymd[-6:]}"]:
                if pattern in lower_to_actual:
                    candidates.append(lower_to_actual[pattern])
        for table in tables:
            low = table.lower()
            if ymd in low and any(prefix in low for prefix in prefixes):
                candidates.append(table)
        for table in dict.fromkeys(candidates):
            ref = TableRef(library=library, table=table)
            columns = describe_columns(conn, ref)
            symbol_col = first_existing(columns, SYMBOL_COLUMNS)
            if kind == "quote":
                has_payload = first_existing(columns, QUOTE_COLUMNS[:4]) and first_existing(columns, QUOTE_COLUMNS[4:])
            else:
                has_payload = first_existing(columns, TRADE_COLUMNS[:4]) and first_existing(columns, TRADE_COLUMNS[4:])
            if symbol_col and has_payload:
                return ref, columns
    return None, []


def query_taq_table(
    conn,
    ref: TableRef,
    columns: list[str],
    trade_date: pd.Timestamp,
    symbols: list[str],
    kind: str,
) -> pd.DataFrame:
    symbol_col = first_existing(columns, SYMBOL_COLUMNS)
    if symbol_col is None:
        raise KeyError(f"No symbol column found for {ref.qualified}")
    validate_identifier(symbol_col)
    if kind == "quote":
        selected = [symbol_col] + [
            col
            for col in [
                first_existing(columns, ["date", "trade_date", "datetime"]),
                first_existing(columns, ["bid", "bid_price", "best_bid", "bbid"]),
                first_existing(columns, ["ofr", "ask", "offer", "ask_price", "best_ask", "bask"]),
                first_existing(columns, ["bidsiz", "bid_size", "bidsize", "bid_shares"]),
                first_existing(columns, ["ofrsiz", "ask_size", "asksize", "ask_shares"]),
            ]
            if col
        ]
    else:
        selected = [symbol_col] + [
            col
            for col in [
                first_existing(columns, ["date", "trade_date", "datetime"]),
                first_existing(columns, ["price", "tr_scond_price", "trade_price", "prc"]),
                first_existing(columns, ["size", "tr_siz", "trade_size", "volume", "shares"]),
            ]
            if col
        ]
    selected = list(dict.fromkeys(selected))
    for col in selected:
        validate_identifier(col)
    date_col = first_existing(columns, ["date", "trade_date"])
    date_filter = ""
    if date_col:
        validate_identifier(date_col)
        date_filter = f" and {date_col} = '{trade_date.date()}'"
    parts: list[pd.DataFrame] = []
    for symbol_chunk in chunked(symbols, 400):
        literals = safe_string_literals(symbol_chunk)
        sql = (
            f"select {', '.join(selected)} from {ref.qualified} "
            f"where {symbol_col} in ({literals}){date_filter}"
        )
        part = conn.raw_sql(sql, date_cols=[date_col] if date_col else None)
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=selected)


def query_stocknames(conn, permnos: list[int], start_year: int, end_year: int) -> pd.DataFrame:
    rows = []
    for idx in range(0, len(permnos), 1000):
        chunk = permnos[idx : idx + 1000]
        permno_sql = ", ".join(str(int(item)) for item in chunk)
        sql = f"""
            select permno, namedt, nameenddt, ticker
            from crsp.stocknames
            where permno in ({permno_sql})
              and namedt <= '{end_year}-12-31'
              and coalesce(nameenddt, '2099-12-31') >= '{start_year}-01-01'
              and ticker is not null
        """
        part = conn.raw_sql(sql, date_cols=["namedt", "nameenddt"])
        if not part.empty:
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["permno", "namedt", "nameenddt", "ticker"])


def identifier_map_for_dates(panel_keys: pd.DataFrame, stocknames: pd.DataFrame) -> pd.DataFrame:
    keys = panel_keys[["permno", "date"]].drop_duplicates().copy()
    names = stocknames.copy()
    keys["date"] = pd.to_datetime(keys["date"])
    names["namedt"] = pd.to_datetime(names["namedt"])
    names["nameenddt"] = pd.to_datetime(names["nameenddt"]).fillna(pd.Timestamp("2099-12-31"))
    merged = keys.merge(names, on="permno", how="left")
    merged = merged[(merged["namedt"] <= merged["date"]) & (merged["date"] <= merged["nameenddt"])]
    merged["ticker"] = merged["ticker"].map(standardize_ticker)
    return merged.dropna(subset=["ticker"]).sort_values(["permno", "date", "namedt"]).drop_duplicates(["permno", "date"], keep="last")[
        ["permno", "date", "ticker"]
    ]


def select_sample_dates(dates: list[pd.Timestamp], sample_months: int, strategy: str) -> list[pd.Timestamp]:
    if not sample_months or sample_months >= len(dates):
        return dates
    if strategy == "last":
        return dates[-sample_months:]
    if strategy == "even":
        positions = pd.Series(range(len(dates))).quantile(
            [idx / max(sample_months - 1, 1) for idx in range(sample_months)]
        )
        chosen = sorted({int(round(value)) for value in positions})
        return [dates[min(idx, len(dates) - 1)] for idx in chosen][:sample_months]
    return dates[:sample_months]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--sample-months", type=int, default=0)
    parser.add_argument("--sample-date-strategy", choices=["first", "last", "even"], default="first")
    parser.add_argument("--sample-symbols", type=int, default=0)
    parser.add_argument("--output-suffix", default="", help="Safe suffix for smoke outputs, e.g. _smoke.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--calibrate-full-panel", action="store_true")
    parser.add_argument(
        "--panel",
        default="data/processed/panel/surface_characteristic_state_daily_risk_panel.parquet",
        help="Panel to augment with TAQ-calibrated costs, relative to project root.",
    )
    args = parser.parse_args()

    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    panel_path = root / args.panel
    if not panel_path.exists():
        fallback = root / "data" / "processed" / "panel" / "surface_characteristic_state_panel.parquet"
        if fallback.exists():
            panel_path = fallback
        else:
            raise FileNotFoundError(f"Missing panel for TAQ cost merge: {panel_path}")
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    target = panel[panel["date"].dt.year.between(args.start_year, args.end_year)][["permno", "date"]].drop_duplicates()
    dates = sorted(target["date"].dropna().unique())
    dates = select_sample_dates([pd.Timestamp(date) for date in dates], args.sample_months, args.sample_date_strategy)

    conn = connect_wrds()
    suffix = safe_suffix(args.output_suffix) or (
        f"_sample_m{args.sample_months}_s{args.sample_symbols}" if args.sample_months or args.sample_symbols else ""
    )
    month_stats: list[dict[str, object]] = []
    shard_dir = root / "data" / "processed" / "panel" / f"taq_costs{suffix}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = dirs["manifests"] / f"taq_costs{suffix}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    try:
        stocknames = query_stocknames(conn, sorted(target["permno"].dropna().astype(int).unique()), args.start_year, args.end_year)
        id_map = identifier_map_for_dates(target, stocknames)
        for date_value in dates:
            trade_date = pd.Timestamp(date_value)
            date_key = trade_date.strftime("%Y%m%d")
            shard_path = shard_dir / f"taq_costs_{date_key}.parquet"
            if args.resume and shard_path.exists():
                month_costs = pd.read_parquet(shard_path)
                stats = {"date": str(trade_date.date()), "status": "PASS_RESUMED", "panel_rows": int(len(month_costs))}
            else:
                mapping_date = id_map[id_map["date"].eq(trade_date)].copy()
                symbols = sorted(mapping_date["ticker"].dropna().unique().tolist())
                if args.sample_symbols:
                    symbols = symbols[: args.sample_symbols]
                    mapping_date = mapping_date[mapping_date["ticker"].isin(symbols)]
                quote_ref, quote_columns = find_taq_table(conn, trade_date, "quote")
                trade_ref, trade_columns = find_taq_table(conn, trade_date, "trade")
                if quote_ref is None:
                    stats = {"date": str(trade_date.date()), "status": "BLOCKED_NO_TAQ_QUOTE_TABLE", "symbols": len(symbols)}
                    write_json_atomic(manifest_dir / f"{date_key}.json", stats)
                    month_stats.append(stats)
                    print(f"taq_cost_date={date_key} status=BLOCKED_NO_TAQ_QUOTE_TABLE")
                    continue
                quotes = query_taq_table(conn, quote_ref, quote_columns, trade_date, symbols, "quote")
                trades = (
                    query_taq_table(conn, trade_ref, trade_columns, trade_date, symbols, "trade")
                    if trade_ref is not None
                    else pd.DataFrame()
                )
                daily_costs = aggregate_taq_daily_costs(quotes, trades, trade_date=trade_date)
                month_costs = merge_taq_costs_to_panel(mapping_date[["permno", "date"]], daily_costs, mapping_date)
                month_costs.to_parquet(shard_path, index=False)
                stats = {
                    "date": str(trade_date.date()),
                    "status": "PASS",
                    "symbols": len(symbols),
                    "quote_table": quote_ref.qualified,
                    "trade_table": trade_ref.qualified if trade_ref else None,
                    "quote_rows": int(len(quotes)),
                    "trade_rows": int(len(trades)),
                    "panel_rows": int(len(month_costs)),
                    "half_spread_coverage": float(month_costs["taq_half_spread"].notna().mean()) if len(month_costs) else 0.0,
                }
            write_json_atomic(manifest_dir / f"{date_key}.json", stats)
            month_stats.append(stats)
            print(f"taq_cost_date={date_key} status={stats['status']} rows={stats.get('panel_rows', 0)}")
    except Exception as exc:
        manifest = {
            "status": "BLOCKED_TAQ_COSTS",
            "output_suffix": suffix,
            "error": type(exc).__name__,
            "error_message": str(exc)[:1000],
            "months": month_stats,
        }
        write_json_atomic(dirs["manifests"] / f"taq_cost_manifest{suffix}.json", manifest)
        print("taq_cost_status=BLOCKED_TAQ_COSTS")
        return 37
    finally:
        conn.close()

    shards = sorted(shard_dir.glob("taq_costs_*.parquet"))
    costs = pd.concat((pd.read_parquet(path) for path in shards), ignore_index=True) if shards else pd.DataFrame()
    cost_path = root / "data" / "processed" / "panel" / f"taq_cost_panel{suffix}.parquet"
    costs.to_parquet(cost_path, index=False)
    merged = panel.merge(
        costs.drop(columns=["ticker"], errors="ignore"),
        on=["permno", "date"],
        how="left",
        suffixes=("", "_taq"),
    )
    out_path = root / "data" / "processed" / "panel" / f"surface_characteristic_state_daily_risk_taq_panel{suffix}.parquet"
    merged.to_parquet(out_path, index=False)
    taq_cols = [col for col in merged.columns if col.startswith("taq_")]
    manifest = {
        "status": "PASS",
        "output_suffix": suffix,
        "sample_date_strategy": args.sample_date_strategy,
        "calibrate_full_panel": bool(args.calibrate_full_panel),
        "months": month_stats,
        "cost_rows": int(len(costs)),
        "panel_rows": int(len(merged)),
        "taq_columns": taq_cols,
        "missingness": {col: float(merged[col].isna().mean()) for col in taq_cols},
        "artifacts": {"cost_panel": str(cost_path.relative_to(root)), "panel": str(out_path.relative_to(root))},
    }
    if args.calibrate_full_panel:
        merged = fill_taq_costs_with_calibration(merged)
        merged.to_parquet(out_path, index=False)
        exact_share = float(merged["taq_cost_source"].eq("exact_taq").mean()) if len(merged) else 0.0
        manifest["exact_taq_share"] = exact_share
        manifest["calibrated_taq_share"] = float(merged["taq_cost_source"].eq("calibrated_taq").mean()) if len(merged) else 0.0
        manifest["missingness"] = {col: float(merged[col].isna().mean()) for col in taq_cols if col in merged}
    write_json_atomic(dirs["manifests"] / f"taq_cost_manifest{suffix}.json", manifest)
    print("taq_cost_status=PASS")
    print(f"panel_rows={manifest['panel_rows']} taq_columns={len(taq_cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
