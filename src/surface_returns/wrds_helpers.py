from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class TableRef:
    library: str
    table: str

    @property
    def qualified(self) -> str:
        validate_identifier(self.library)
        validate_identifier(self.table)
        return f"{self.library}.{self.table}"


def validate_identifier(value: str) -> None:
    if not IDENTIFIER_RE.match(value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")


def connect_wrds():
    import wrds

    return wrds.Connection()


def list_tables_safe(conn, library: str) -> list[str]:
    try:
        return sorted(conn.list_tables(library=library))
    except Exception:
        return []


def describe_columns(conn, ref: TableRef) -> list[str]:
    desc = conn.describe_table(library=ref.library, table=ref.table)
    if isinstance(desc, pd.DataFrame):
        for col in ["name", "column_name", "Column", "column"]:
            if col in desc.columns:
                return [str(item).lower() for item in desc[col].tolist()]
        return [str(item).lower() for item in desc.index.tolist()]
    return []


def choose_table(tables: Iterable[str], preferred: list[str], contains: list[str] | None = None) -> str | None:
    table_set = {table.lower(): table for table in tables}
    for table in preferred:
        if table.lower() in table_set:
            return table_set[table.lower()]
    if contains:
        for table in sorted(table_set):
            if all(piece.lower() in table for piece in contains):
                return table_set[table]
    return None


def candidate_tables(
    tables: Iterable[str],
    preferred: list[str],
    contains: list[str] | None = None,
) -> list[str]:
    table_set = {table.lower(): table for table in tables}
    ordered: list[str] = []
    for table in preferred:
        found = table_set.get(table.lower())
        if found and found not in ordered:
            ordered.append(found)
    if contains:
        for table in sorted(table_set):
            if all(piece.lower() in table for piece in contains):
                found = table_set[table]
                if found not in ordered:
                    ordered.append(found)
    return ordered


def select_existing(columns: list[str], desired: list[str]) -> list[str]:
    colset = {col.lower(): col for col in columns}
    return [colset[col.lower()] for col in desired if col.lower() in colset]


def safe_limit_sql(ref: TableRef, columns: list[str], limit: int = 5) -> str:
    validate_identifier(ref.library)
    validate_identifier(ref.table)
    for col in columns:
        validate_identifier(col)
    col_sql = ", ".join(columns) if columns else "*"
    return f"select {col_sql} from {ref.qualified} limit {int(limit)}"
