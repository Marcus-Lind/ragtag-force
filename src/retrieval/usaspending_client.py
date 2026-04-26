"""USAspending.gov API client for federal contract intelligence.

Searches the USAspending awards API to find defense contracts by keyword,
agency, NAICS code, and location. No authentication required.
"""

from typing import Optional

import httpx
from datetime import datetime

USASPENDING_BASE = "https://api.usaspending.gov/api/v2"
SEARCH_ENDPOINT = f"{USASPENDING_BASE}/search/spending_by_award/"

# Contract award type codes: A=BPA Call, B=Purchase Order, C=Delivery Order, D=Definitive
CONTRACT_TYPES = ["A", "B", "C", "D"]

FIELDS = [
    "Award ID",
    "Recipient Name",
    "Award Amount",
    "Description",
    "Start Date",
]


def search_contracts(
    keywords: list[str] | None = None,
    agency_name: str | None = None,
    naics_codes: list[str] | None = None,
    state_code: str | None = None,
    recipient_name: str | None = None,
    limit: int = 5,
) -> dict:
    """Search USAspending for defense contracts.

    Args:
        keywords: Search terms (matched against contract descriptions).
        agency_name: Toptier agency name (e.g., "Department of Defense").
        naics_codes: NAICS industry codes to filter by.
        state_code: Two-letter state code for place of performance.
        recipient_name: Contractor/recipient name to search.
        limit: Max results to return.

    Returns:
        Dict with 'contracts' list and 'query_params' metadata.
    """
    now = datetime.now()
    # Search last 3 fiscal years for good coverage
    start_date = f"{now.year - 3}-10-01"
    end_date = f"{now.year}-09-30"

    filters: dict = {
        "time_period": [{"start_date": start_date, "end_date": end_date}],
        "award_type_codes": CONTRACT_TYPES,
    }

    if keywords:
        filters["keywords"] = keywords

    if agency_name:
        filters["agencies"] = [
            {"type": "awarding", "tier": "toptier", "name": agency_name}
        ]

    if naics_codes:
        filters["naics_codes"] = {"require": naics_codes}

    if state_code:
        filters["recipient_locations"] = [
            {"country": "USA", "state": state_code}
        ]

    if recipient_name:
        filters["recipient_search_text"] = [recipient_name]

    payload = {
        "filters": filters,
        "fields": FIELDS,
        "limit": limit,
        "sort": "Award Amount",
        "order": "desc",
    }

    query_params = {
        "keywords": keywords or [],
        "agency": agency_name or "Any",
        "naics_codes": naics_codes or [],
        "state": state_code or "Any",
        "recipient": recipient_name or "Any",
        "time_range": f"{start_date} to {end_date}",
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(SEARCH_ENDPOINT, json=payload)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        contracts = []
        for r in results:
            amount = r.get("Award Amount", 0)
            contracts.append({
                "award_id": r.get("Award ID", "N/A"),
                "recipient": r.get("Recipient Name", "Unknown"),
                "amount": amount,
                "amount_display": _format_amount(amount),
                "description": (r.get("Description") or "No description")[:200],
                "start_date": r.get("Start Date", "N/A"),
            })

        return {
            "contracts": contracts,
            "total_found": len(contracts),
            "has_more": data.get("page_metadata", {}).get("hasNext", False),
            "source": "USAspending.gov API (live)",
            "query_params": query_params,
        }

    except Exception as e:
        return {
            "contracts": [],
            "total_found": 0,
            "has_more": False,
            "source": "USAspending.gov API (error)",
            "error": str(e),
            "query_params": query_params,
        }


def _format_amount(amount: float) -> str:
    """Format dollar amount for display."""
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.2f}B"
    elif amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount / 1_000:.0f}K"
    else:
        return f"${amount:,.2f}"
