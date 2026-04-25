"""Structured data lookup from SQLite.

Queries the entitlements database for BAH rates, base pay, and BAS
rates. Never embeds these — always queries directly per design rules.
"""

import sqlite3
from pathlib import Path
from typing import Optional

from src.config import SQLITE_PATH


def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a SQLite connection.

    Args:
        db_path: Path to SQLite database. Defaults to config SQLITE_PATH.

    Returns:
        SQLite connection.
    """
    db_path = db_path or SQLITE_PATH
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def lookup_bah(
    grade: str,
    dependency_status: str,
    locality_code: str,
    db_path: Optional[Path] = None,
) -> Optional[dict]:
    """Look up BAH rate for a given grade, dependency status, and locality.

    Args:
        grade: Pay grade (e.g., 'E-4', 'O-3').
        dependency_status: 'with_dependents' or 'without_dependents'.
        locality_code: BAH locality code (e.g., 'CLARKSVILLE_TN').
        db_path: Optional database path override.

    Returns:
        Dict with rate info or None if not found.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """SELECT grade, dependency_status, locality, locality_code, monthly_rate
               FROM bah_rates
               WHERE grade = ? AND dependency_status = ? AND locality_code = ?""",
            (grade, dependency_status, locality_code),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def lookup_base_pay(
    grade: str,
    years_of_service: int = 0,
    db_path: Optional[Path] = None,
) -> Optional[dict]:
    """Look up base pay for a given grade and years of service.

    Args:
        grade: Pay grade (e.g., 'E-4', 'O-3').
        years_of_service: Years of service (defaults to 0).
        db_path: Optional database path override.

    Returns:
        Dict with pay info or None if not found.
    """
    conn = _get_connection(db_path)
    try:
        # Determine which table to query
        if grade.upper().startswith("E"):
            table = "enlisted_base_pay"
        elif grade.upper().startswith("O"):
            table = "officer_base_pay"
        else:
            return None

        cursor = conn.execute(
            f"""SELECT grade, yos_min, yos_max, monthly_rate
                FROM {table}
                WHERE grade = ? AND yos_min <= ? AND yos_max > ?
                ORDER BY yos_min DESC
                LIMIT 1""",
            (grade, years_of_service, years_of_service),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def lookup_bas(
    category: str = "Enlisted",
    db_path: Optional[Path] = None,
) -> Optional[dict]:
    """Look up BAS rate for enlisted or officer.

    Args:
        category: 'Enlisted' or 'Officer'.
        db_path: Optional database path override.

    Returns:
        Dict with BAS rate info or None if not found.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT category, monthly_rate FROM bas_rates WHERE category = ?",
            (category,),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def lookup_all_bah_for_grade(
    grade: str,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """Look up all BAH rates for a given grade across all localities.

    Args:
        grade: Pay grade (e.g., 'E-4').
        db_path: Optional database path override.

    Returns:
        List of dicts with rate info.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """SELECT grade, dependency_status, locality, locality_code, monthly_rate
               FROM bah_rates
               WHERE grade = ?
               ORDER BY locality_code, dependency_status""",
            (grade,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def lookup_perdiem(
    city: str,
    state: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Optional[dict]:
    """Look up per diem rate for a city/state.

    Tries exact city match first, falls back to Standard CONUS rate.

    Args:
        city: City name (e.g., 'San Diego', 'Fayetteville').
        state: Optional 2-letter state code (e.g., 'CA', 'NC').
        db_path: Optional database path override.

    Returns:
        Dict with lodging_rate, mie_rate, etc., or None if not found.
    """
    conn = _get_connection(db_path)
    try:
        if state:
            cursor = conn.execute(
                """SELECT city, state, county, lodging_rate, mie_rate, fiscal_year, notes
                   FROM perdiem_rates
                   WHERE LOWER(city) = LOWER(?) AND LOWER(state) = LOWER(?)""",
                (city, state),
            )
        else:
            cursor = conn.execute(
                """SELECT city, state, county, lodging_rate, mie_rate, fiscal_year, notes
                   FROM perdiem_rates
                   WHERE LOWER(city) = LOWER(?) AND city != 'Standard CONUS'""",
                (city,),
            )
        row = cursor.fetchone()
        if row:
            d = dict(row)
            d["total_perdiem"] = d["lodging_rate"] + d["mie_rate"]
            return d

        # Fall back to standard CONUS rate
        cursor = conn.execute(
            """SELECT city, state, county, lodging_rate, mie_rate, fiscal_year, notes
               FROM perdiem_rates
               WHERE city = 'Standard CONUS'"""
        )
        row = cursor.fetchone()
        if row:
            d = dict(row)
            d["total_perdiem"] = d["lodging_rate"] + d["mie_rate"]
            return d
        return None
    finally:
        conn.close()
