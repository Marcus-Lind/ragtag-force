"""Ingest TDY travel reference content into ChromaDB.

Creates JTR-based reference content and ingests it into a separate
'tdy_documents' ChromaDB collection. Idempotent — safe to run multiple times.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.ingest.embeddings import embed_texts
from src.ingest.vector_store import get_chroma_client

# Representative JTR chapter content organized by section
JTR_CONTENT = [
    # ── Chapter 2: TDY Travel ──────────────────────────────────────────
    {
        "text": """JTR Chapter 2 - Temporary Duty (TDY) Travel Allowances

Section 020101: A Service member or civilian employee is authorized TDY travel allowances when directed to travel away from the permanent duty station (PDS) on official business. TDY travel must be authorized or approved in writing before travel begins. The travel authorization must specify the purpose of travel, estimated costs, transportation mode, and TDY location.

A Service member on TDY is authorized per diem (lodging, meals, and incidental expenses), transportation costs, and other necessary expenses. The traveler must use the most economical means of transportation that meets mission requirements.""",
        "metadata": {"source_doc": "Joint Travel Regulations", "section_heading": "Chapter 2 - TDY Travel Allowances", "chapter": "2"},
    },
    {
        "text": """JTR Section 020201: Transportation Allowances for TDY Travel

The traveler is authorized the transportation mode that is most advantageous to the Government. The order of preference for transportation is:
1. Government vehicle (GOV) when available
2. Common carrier (commercial air, rail, or bus)
3. Rental car when authorized
4. Privately Owned Vehicle (POV)

Commercial air travel should be booked through the Commercial Travel Office (CTO) using City Pair fares when available. City Pair fares are discounted government contract fares that are fully refundable and do not require advance purchase.""",
        "metadata": {"source_doc": "Joint Travel Regulations", "section_heading": "Section 020201 - Transportation", "chapter": "2"},
    },
    {
        "text": """JTR Section 020205: Privately Owned Vehicle (POV) Mileage

When a POV is authorized for TDY travel, the traveler is reimbursed at the current standard mileage rate set by the GSA. For FY2026, the standard mileage reimbursement rate is $0.67 per mile. The traveler may also be reimbursed for parking fees, tolls, and ferry fares.

POV mileage is computed using the Defense Table of Official Distances (DTOD). If the POV is used as advantageous to the Government, the traveler receives the standard mileage rate. If POV use is for the traveler's convenience and not advantageous to the Government, reimbursement is limited to the cost of the authorized transportation mode (constructive cost comparison).""",
        "metadata": {"source_doc": "Joint Travel Regulations", "section_heading": "Section 020205 - POV Mileage", "chapter": "2"},
    },
    {
        "text": """JTR Section 020301: Rental Car Authorization

A rental car may be authorized when it is to the Government's advantage, considering total transportation costs, the traveler's needs at the TDY location, and available alternatives. A rental car is generally authorized when:
- Public transportation is not available or practical
- Multiple trips to different locations are required
- The rental cost is less than taxi or rideshare costs
- The TDY location lacks adequate public transit

The traveler must use the Defense Travel Management Office (DTMO) rental car program and obtain the Government rate. Personal accident insurance and liability coverage are provided under the Federal Tort Claims Act and should not be purchased.""",
        "metadata": {"source_doc": "Joint Travel Regulations", "section_heading": "Section 020301 - Rental Car", "chapter": "2"},
    },
    {
        "text": """JTR Section 020401: Travel Advances

A traveler may request a travel advance for estimated TDY expenses through the Defense Travel System (DTS). The advance is limited to the estimated per diem and other reimbursable expenses for the TDY period. Government Travel Charge Card (GTCC) holders are generally expected to use the GTCC for travel expenses rather than requesting cash advances.

Travel advances must be liquidated within 5 business days of completing travel by filing a travel voucher in DTS. Failure to file a timely voucher may result in automatic salary offset to collect the outstanding advance.""",
        "metadata": {"source_doc": "Joint Travel Regulations", "section_heading": "Section 020401 - Travel Advances", "chapter": "2"},
    },

    # ── Chapter 3: Per Diem ────────────────────────────────────────────
    {
        "text": """JTR Chapter 3 - Per Diem Rates and Allowances

Section 030101: Per diem is the daily allowance for lodging, meals, and incidental expenses (M&IE) when a traveler is on TDY. Per diem rates are set by the General Services Administration (GSA) for CONUS locations and by the Department of State for OCONUS locations.

Per diem has two components:
1. Lodging: Actual expense up to the maximum lodging rate for the TDY location
2. Meals and Incidental Expenses (M&IE): A flat daily rate based on the TDY location

The total per diem rate equals the maximum lodging rate plus the M&IE rate for that location. For FY2026, the standard CONUS per diem rate is $178/day ($110 lodging + $68 M&IE).""",
        "metadata": {"source_doc": "Joint Travel Regulations", "section_heading": "Chapter 3 - Per Diem Rates", "chapter": "3"},
    },
    {
        "text": """JTR Section 030201: CONUS Per Diem Rates

CONUS per diem rates are published annually by GSA and are effective from October 1 through September 30 of each fiscal year. There are two types of rates:
1. Standard Rate: Applies to all locations not specifically listed as Non-Standard Areas (NSAs). For FY2026, the standard rate is $110/night lodging and $68/day M&IE.
2. Non-Standard Area (NSA) Rate: Higher rates for designated high-cost areas. Examples include Washington DC ($218 lodging, $79 M&IE), San Diego ($194 lodging, $74 M&IE), and Honolulu ($253 lodging, $79 M&IE).

Lodging is reimbursed at actual cost up to the maximum rate. If the traveler's lodging cost exceeds the maximum rate, the excess is the traveler's responsibility unless a lodging rate waiver is approved.""",
        "metadata": {"source_doc": "Joint Travel Regulations", "section_heading": "Section 030201 - CONUS Per Diem", "chapter": "3"},
    },
    {
        "text": """JTR Section 030301: M&IE Breakdown and First/Last Day Rules

The M&IE rate includes breakfast, lunch, dinner, and incidental expenses. The incidental expenses portion covers tips to porters, baggage handlers, and hotel staff. For FY2026, the M&IE breakdown at the standard CONUS rate ($68) is:
- Breakfast: $16
- Lunch: $18
- Dinner: $27
- Incidental Expenses: $7

On the first and last day of TDY travel, the traveler receives 75% of the applicable M&IE rate. This is known as the "travel day" rate. For the standard CONUS rate, the travel day M&IE is $51 (75% of $68).

If Government meals are provided (e.g., meals included in conference registration or available in a dining facility), the corresponding meal portion is deducted from the M&IE rate.""",
        "metadata": {"source_doc": "Joint Travel Regulations", "section_heading": "Section 030301 - M&IE Breakdown", "chapter": "3"},
    },
    {
        "text": """JTR Section 030401: Lodging Receipts and Actual Expense

Lodging is reimbursed on an actual expense basis up to the maximum lodging rate. The traveler must provide receipts for lodging expenses. If receipts are not available, the traveler must provide a statement explaining why.

Actual Expense Allowance (AEA): When lodging costs at the TDY location exceed the maximum rate and no suitable lodging is available within the rate, the authorizing official may approve an Actual Expense Allowance up to 300% of the maximum per diem rate. AEA requests must document the non-availability of lodging at the standard rate.

Government Quarters: When government quarters are available and adequate, the traveler must use them. If directed to use government quarters but commercial lodging is used instead, lodging reimbursement is limited to the government quarters rate.""",
        "metadata": {"source_doc": "Joint Travel Regulations", "section_heading": "Section 030401 - Lodging", "chapter": "3"},
    },
    {
        "text": """JTR Section 030501: Long-Term TDY (Flat Rate Per Diem)

For TDY assignments of 30 days or more at a single location, a flat rate per diem may apply. The flat rate per diem is 75% of the locality per diem rate for the first 30 days, reducing to 55% after 180 days. This reduced rate reflects the assumption that travelers on long-term TDY can find more economical lodging and meal arrangements.

The authorizing official may waive the flat rate reduction if the traveler demonstrates that suitable long-term lodging is not available or that the reduced rate would cause hardship. Extended TDY beyond 180 days requires senior-level approval and justification that PCS is not appropriate.""",
        "metadata": {"source_doc": "Joint Travel Regulations", "section_heading": "Section 030501 - Long-Term TDY", "chapter": "3"},
    },

    # ── Chapter 4: Government Quarters ─────────────────────────────────
    {
        "text": """JTR Chapter 4 - Government Quarters and Lodging

Section 040101: Government quarters (billeting, visiting officer/enlisted quarters, or transient lodging) must be used when available and adequate at the TDY installation. The traveler must check availability before booking commercial lodging.

A Certificate of Non-Availability (CNA) is required when government quarters are available but the traveler uses commercial lodging. The CNA must be issued by the installation billeting office. If government quarters are not available, the traveler may use commercial lodging and receive the standard per diem lodging rate.

Government quarters rates are typically $0-$25 per night, significantly less than commercial lodging rates. Travelers using government quarters still receive the full M&IE rate.""",
        "metadata": {"source_doc": "Joint Travel Regulations", "section_heading": "Chapter 4 - Government Quarters", "chapter": "4"},
    },

    # ── Chapter 5: PCS Travel ──────────────────────────────────────────
    {
        "text": """JTR Chapter 5 - Permanent Change of Station (PCS) Travel

Section 050101: PCS travel is authorized when a Service member receives orders to relocate to a new permanent duty station. PCS entitlements include:
1. Transportation of the Service member and dependents
2. Per diem for travel days
3. Dislocation Allowance (DLA) - a one-time payment to offset relocation costs
4. Temporary Lodging Expense (TLE) - up to 14 days of lodging at the old and new duty station
5. Household goods (HHG) transportation and storage

DLA rates for FY2026 vary by pay grade and dependency status. For an E-4 without dependents, DLA is approximately $1,014.81. For an O-3 with dependents, DLA is approximately $3,174.03.""",
        "metadata": {"source_doc": "Joint Travel Regulations", "section_heading": "Chapter 5 - PCS Travel", "chapter": "5"},
    },
    {
        "text": """JTR Section 050301: PCS Per Diem and Travel Days

PCS per diem is authorized for each day of travel between the old and new duty stations. The number of authorized travel days is based on the official distance:
- 1 day for each 350 miles of official distance
- Plus 1 additional day for any remaining distance over 50 miles

Per diem during PCS travel is calculated at the standard CONUS rate unless the traveler stops at a non-standard area. The lodging portion is actual expense up to the maximum rate; M&IE is the flat locality rate.

Service members traveling with dependents receive per diem for both the member and dependents. Dependent per diem is typically 75% of the member's rate.""",
        "metadata": {"source_doc": "Joint Travel Regulations", "section_heading": "Section 050301 - PCS Per Diem", "chapter": "5"},
    },

    # ── DTS and Voucher Filing ─────────────────────────────────────────
    {
        "text": """Defense Travel System (DTS) - Travel Authorization and Voucher Filing

All official travel must be authorized through DTS before travel begins. The travel authorization (TA) specifies:
- Purpose and justification for travel
- TDY location and dates
- Estimated costs (transportation, per diem, other expenses)
- Funding citation (line of accounting)
- Transportation mode and routing
- Rental car authorization (if applicable)

After travel is complete, the traveler must file a travel voucher in DTS within 5 business days. The voucher must include:
- Actual lodging receipts
- Any receipts for expenses over $75
- Airline ticket receipt/itinerary
- Rental car receipt (if applicable)
- Any other required documentation

The travel voucher is routed to the authorizing official for approval, then to the finance office for payment. Payment is typically made within 30 days of voucher approval.""",
        "metadata": {"source_doc": "Defense Travel System Guide", "section_heading": "Travel Authorization and Vouchers", "chapter": "DTS"},
    },
    {
        "text": """Government Travel Charge Card (GTCC) - Policy and Use

All DoD travelers are required to use the Government Travel Charge Card (GTCC) for official travel expenses. The GTCC must be used for:
- Commercial lodging
- Rental cars
- Commercial air travel (when not booked through CTO)
- Other travel-related expenses

ATM cash advances on the GTCC should only be used for expenses that cannot be charged (e.g., meals, tips, taxi fares in cash-only situations). The maximum ATM advance is typically limited to the estimated cash expenses for the trip.

The GTCC statement must be paid in full within 30 days of the billing date. Split disbursement through DTS can route per diem payments directly to the GTCC account. Delinquent accounts may result in counseling, suspension of travel privileges, or adverse action.""",
        "metadata": {"source_doc": "DoD Financial Management Regulation", "section_heading": "GTCC Policy", "chapter": "FMR"},
    },
]


def ingest_tdy_content() -> int:
    """Ingest TDY travel content into a separate ChromaDB collection.

    Returns:
        Number of chunks ingested.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name="tdy_documents",
        metadata={"hnsw:space": "cosine"},
    )

    # Clear existing content for idempotency
    existing = collection.count()
    if existing > 0:
        collection.delete(where={"source_doc": {"$ne": ""}})
        try:
            collection = client.get_or_create_collection(
                name="tdy_documents",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            pass

    texts = [item["text"] for item in JTR_CONTENT]
    metadatas = [item["metadata"] for item in JTR_CONTENT]
    ids = [f"tdy-{i:04d}" for i in range(len(texts))]

    print(f"Embedding {len(texts)} TDY document chunks...")
    embeddings = embed_texts(texts)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    final_count = collection.count()
    print(f"TDY collection: {final_count} chunks ingested")
    return final_count


if __name__ == "__main__":
    count = ingest_tdy_content()
    print(f"Done: {count} TDY chunks in ChromaDB")
