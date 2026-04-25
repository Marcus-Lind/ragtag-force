"""Live GSA Per Diem API client.

Queries the official GSA Per Diem API (api.gsa.gov) for real-time
federal travel per diem rates by city and state. Falls back to
Standard CONUS rates on any error.
"""

import datetime
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GSA_API_BASE = "https://api.gsa.gov/travel/perdiem/v2/rates"
GSA_API_KEY = "DEMO_KEY"
REQUEST_TIMEOUT = 5  # seconds

# Standard CONUS fallback (FY2025)
STANDARD_CONUS = {
    "city": "Standard CONUS",
    "state": "",
    "lodging_rate": 110,
    "mie_rate": 68,
    "total_perdiem": 178,
    "source": "Standard CONUS Rate (fallback)",
    "fiscal_year": None,
    "seasonal_note": None,
}


def _current_fiscal_year() -> int:
    """Return the current federal fiscal year (Oct 1 start)."""
    now = datetime.date.today()
    return now.year + 1 if now.month >= 10 else now.year


def _current_month() -> int:
    """Return the current month number (1-12)."""
    return datetime.date.today().month


def lookup_perdiem_gsa(
    city: str,
    state: Optional[str] = None,
) -> dict:
    """Query the live GSA Per Diem API for a city/state.

    Args:
        city: City name (e.g., "San Diego", "Huntsville").
        state: Two-letter state code (e.g., "CA", "AL", "DC").

    Returns:
        Dictionary with lodging_rate (current month), mie_rate,
        total_perdiem, source info, and optional seasonal_note.
        Falls back to Standard CONUS on any error.
    """
    if not state:
        state = "DC" if city.lower() in ("washington", "dc", "pentagon") else ""

    if not state:
        logger.warning("No state provided for city '%s', returning standard rate", city)
        return {**STANDARD_CONUS, "city": city}

    fy = _current_fiscal_year()
    url = f"{GSA_API_BASE}/city/{requests.utils.quote(city)}/state/{state}/year/{fy}"

    try:
        resp = requests.get(
            url,
            headers={"X-API-KEY": GSA_API_KEY},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        rates = data.get("rates", [])
        if not rates or not rates[0].get("rate"):
            logger.warning("GSA API returned empty rates for %s, %s", city, state)
            return {**STANDARD_CONUS, "city": city, "state": state}

        rate = rates[0]["rate"][0]
        mie = rate.get("meals", 68)
        months = rate.get("months", {}).get("month", [])

        # Get current month's lodging rate
        current_month = _current_month()
        lodging = 110  # default
        for m in months:
            if m.get("number") == current_month:
                lodging = m.get("value", 110)
                break

        # Check if lodging varies by season
        lodging_values = [m.get("value", 0) for m in months]
        is_seasonal = len(set(lodging_values)) > 1
        seasonal_note = None
        if is_seasonal:
            low = min(lodging_values)
            high = max(lodging_values)
            seasonal_note = f"Seasonal rates: ${low}-${high}/night (showing current month)"

        gsa_city = rate.get("city", city)
        is_standard = rate.get("standardRate", "false") == "true"

        result = {
            "city": gsa_city,
            "state": state,
            "lodging_rate": lodging,
            "mie_rate": mie,
            "total_perdiem": lodging + mie,
            "source": f"GSA Per Diem API (FY{fy}, live)" if not is_standard else f"GSA Standard CONUS Rate (FY{fy})",
            "fiscal_year": fy,
            "seasonal_note": seasonal_note,
        }

        logger.info("GSA API: %s, %s → $%d lodging + $%d M&IE", gsa_city, state, lodging, mie)
        return result

    except requests.RequestException as e:
        logger.error("GSA API request failed for %s, %s: %s", city, state, e)
        return {**STANDARD_CONUS, "city": city, "state": state or ""}

    except (KeyError, IndexError, ValueError) as e:
        logger.error("GSA API response parse error for %s, %s: %s", city, state, e)
        return {**STANDARD_CONUS, "city": city, "state": state or ""}
