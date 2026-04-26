GenAI.mil
The US military runs on paperwork. Build the AI assistant that makes the rank-and-file faster, smarter, and less buried in bureaucracy, and does it offline.

The idea: While modern defense often focuses on the "digital battlefield," the most immediate friction for the three million men and women in uniform often occurs in the "administrative trenches." From navigating thousands of pages of convoluted regulations and policies to the grueling manual labor of filling out paperwork for housing, maintenance, or housing, the "bureaucratic tail" of the military significantly drains mission readiness. The launch of the GenAI.mil portal provides a secure environment to weaponize AI against this inefficiency. The challenge for this hackathon is to build an AI-enabled software application that solves the everyday problems of the rank-and-file and streamlines the logistical hurdles that keep warfighters behind desks instead of in the field.

A good starting point: Pick one user persona (e.g., a junior NCO planning a training trip) and build an end-to-end solution for them. Set up a RAG pipeline over a small corpus of Army publications or Field Manuals (FMs). Get accurate retrieval working before adding form generation or logistics planning.

Judges:

SF	Boston	DC
David Vernal	Dr. Ho-Chit Liu	Kevin McQueary
Neeraj Chandra	Dr. Sanjeev Mohindra	Raj Panth
		Stuart Wagner

Example project directions:
•	Regulation navigator: ingest Army Regulations and Field Manuals into a vector store; answer "what does AR 600-8-10 say about leave accrual?" accurately
•	Form auto-filler: take a natural language request ("I need to file a DA 31 for 10 days leave starting June 3") and populate the correct form fields
•	TDY planner: pull Joint Travel Regulations and GSA rates to generate a compliant travel itinerary with per diem calculations
•	Contract intel tool: query USAspending or SAM.gov to surface relevant past awards and help a contracting officer understand the landscape

Datasets & APIs:
•	DTIC Public STINET — DoD technical reports and TTPs, full-text searchable
•	Army Publishing Directorate — all ARs, ADPs, FMs, and DA forms (bulk downloadable PDFs)
•	Air Force e-Publishing — AFIs, AFMANs, and AF forms
•	Joint Travel Regulations — the canonical DoD travel and per diem rulebook
•	GSA Open APIs — federal procurement catalog and per diem rates
•	SAM.gov Public API — contract opportunities and entity registrations (no auth for public data)
•	USAspending.gov API — all federal contracts and spending, fully open
•	Federal Register API — regulations, executive orders, notices
•	eCFR Bulk Data — CFR Title 32 (National Defense) and Title 48 (Federal Acquisition) as XML/JSON

