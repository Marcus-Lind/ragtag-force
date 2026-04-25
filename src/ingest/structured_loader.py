"""Load structured CSV data into SQLite.

Reads BAH rates, base pay tables, and BAS rates from CSV files and creates
corresponding SQLite tables. Idempotent — safe to run multiple times.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import SQLITE_PATH, DATA_STRUCTURED_PATH


def _ensure_parent_dir(path: Path) -> None:
    """Create parent directories if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_bah_rates(conn: sqlite3.Connection) -> int:
    """Load BAH rates CSV into the bah_rates table.

    Returns:
        Number of rows loaded.
    """
    csv_path = DATA_STRUCTURED_PATH / "bah_rates_2026.csv"
    df = pd.read_csv(csv_path)
    df.to_sql("bah_rates", conn, if_exists="replace", index=False)
    return len(df)


def load_enlisted_base_pay(conn: sqlite3.Connection) -> int:
    """Load enlisted base pay CSV into the enlisted_base_pay table.

    Returns:
        Number of rows loaded.
    """
    csv_path = DATA_STRUCTURED_PATH / "enlisted_base_pay_2026.csv"
    df = pd.read_csv(csv_path)
    df.to_sql("enlisted_base_pay", conn, if_exists="replace", index=False)
    return len(df)


def load_officer_base_pay(conn: sqlite3.Connection) -> int:
    """Load officer base pay CSV into the officer_base_pay table.

    Returns:
        Number of rows loaded.
    """
    csv_path = DATA_STRUCTURED_PATH / "officer_base_pay_2026.csv"
    df = pd.read_csv(csv_path)
    df.to_sql("officer_base_pay", conn, if_exists="replace", index=False)
    return len(df)


def load_bas_rates(conn: sqlite3.Connection) -> int:
    """Load BAS rates CSV into the bas_rates table.

    Returns:
        Number of rows loaded.
    """
    csv_path = DATA_STRUCTURED_PATH / "bas_rates_2026.csv"
    df = pd.read_csv(csv_path)
    df.to_sql("bas_rates", conn, if_exists="replace", index=False)
    return len(df)


def load_perdiem_rates(conn: sqlite3.Connection) -> int:
    """Load per diem rates CSV into the perdiem_rates table.

    Returns:
        Number of rows loaded.
    """
    csv_path = DATA_STRUCTURED_PATH / "perdiem_rates_2026.csv"
    df = pd.read_csv(csv_path)
    df.to_sql("perdiem_rates", conn, if_exists="replace", index=False)
    return len(df)


def load_all_structured_data(db_path: Path | None = None) -> dict[str, int]:
    """Load all structured CSV data into SQLite.

    Args:
        db_path: Path to the SQLite database. Defaults to config SQLITE_PATH.

    Returns:
        Dictionary mapping table name to row count.
    """
    db_path = db_path or SQLITE_PATH
    _ensure_parent_dir(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        counts = {
            "bah_rates": load_bah_rates(conn),
            "enlisted_base_pay": load_enlisted_base_pay(conn),
            "officer_base_pay": load_officer_base_pay(conn),
            "bas_rates": load_bas_rates(conn),
            "perdiem_rates": load_perdiem_rates(conn),
        }
        conn.commit()
        return counts
    finally:
        conn.close()


if __name__ == "__main__":
    counts = load_all_structured_data()
    for table, count in counts.items():
        print(f"  {table}: {count} rows")
    print("Structured data loaded successfully.")
