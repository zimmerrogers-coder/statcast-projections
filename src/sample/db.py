"""Database access. No password here: libpq reads it from pgpass.conf."""

import os
from pathlib import Path

import pandas as pd
import psycopg

DB_NAME = "advanced_metrics"
DB_USER = "metrics_app"
DB_HOST = "localhost"
DB_PORT = 5432

ROOT = Path(__file__).resolve().parent.parent.parent
SQL_DIR = ROOT / "sql"
REPORTS = ROOT / "reports"


def connect() -> psycopg.Connection:
    return psycopg.connect(host=DB_HOST, port=DB_PORT, user=DB_USER,
                           dbname=os.environ.get("SAMPLE_DB_NAME", DB_NAME))


def apply_schema(conn: psycopg.Connection) -> list[str]:
    """Run every sql/NN_*.sql file in name order. Each is safe to run again."""
    ran = []
    for path in sorted(SQL_DIR.glob("[0-9][0-9]_*.sql")):
        conn.execute(path.read_text(encoding="utf-8"))
        ran.append(path.name)
    conn.commit()
    return ran


def read(conn: psycopg.Connection, sql: str, params=None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def records(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """Rows as plain Python tuples, with every kind of 'missing' turned into None."""
    plain = frame[columns].astype(object).where(frame[columns].notna(), None)
    return [tuple(v.item() if hasattr(v, "item") else v for v in row)
            for row in plain.itertuples(index=False, name=None)]


def insert(conn: psycopg.Connection, table: str, frame: pd.DataFrame, columns: list[str],
           on_conflict: str = "") -> None:
    sql = (f"INSERT INTO {table} ({', '.join(columns)}) "
           f"VALUES ({', '.join(['%s'] * len(columns))}) {on_conflict}")
    with conn.cursor() as cur:
        cur.executemany(sql, records(frame, columns))
