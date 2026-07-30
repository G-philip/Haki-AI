"""
Coverage audit — lists every unique source document currently in the
database, with chunk counts, so you can see your actual coverage at a
glance and plan what to ingest next deliberately, rather than discovering
gaps one query at a time.

Usage:
    python audit_coverage.py
"""

import sys
import time
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))

from logic.rag_engine import RAGEngine
from logic.embeddings import embeddings_service

print("Loading...")
embeddings_service.load()
rag = RAGEngine()

while not rag.is_ready():
    time.sleep(0.5)

print(f"Database: {rag.collection.count()} total chunks\n")

all_docs = rag.collection.get(include=["metadatas"])

source_counts = Counter()
category_counts = Counter()
filename_to_source = {}

for meta in all_docs["metadatas"]:
    if not meta:
        continue
    source = meta.get("source", "Unknown")
    category = meta.get("category", "Unknown")
    filename = meta.get("filename", "")
    source_counts[source] += 1
    category_counts[category] += 1
    if filename:
        filename_to_source[filename] = source

print("=" * 70)
print(f"DOCUMENTS CURRENTLY INGESTED ({len(source_counts)} unique sources)")
print("=" * 70)
for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"  {count:4d} chunks  |  {source}")

print()
print("=" * 70)
print("REFERENCE: common Kenyan legal topics and their typical source docs")
print("Cross-check against the list above to spot likely gaps.")
print("=" * 70)

reference_topics = {
    "Civil procedure (filing, withdrawal, service of process)": [
        "Civil Procedure Act (Cap. 21)", "Civil Procedure Rules, 2010"
    ],
    "Criminal procedure (arrest, trial, bail)": [
        "Criminal Procedure Code (Cap. 75)", "already present" 
    ],
    "Constitutional rights": ["Constitution of Kenya 2010", "already present"],
    "Marriage / divorce": [
        "Marriage Act, 2014", "Matrimonial Property Act, 2013"
    ],
    "Succession / inheritance / wills": ["Law of Succession Act (Cap. 160)"],
    "Children / custody": ["Children Act, 2022"],
    "Employment disputes": ["Employment Act, 2007", "Labour Relations Act, 2007"],
    "Land disputes": [
        "Land Act, 2012", "Land Registration Act, 2012", "Sectional Properties Act"
    ],
    "Tenancy / landlord-tenant": ["Rent Restriction Act", "Landlord and Tenant Act (not yet in force, check current status)"],
    "Contracts": ["Law of Contract Act (Cap. 23)"],
    "Police conduct / oversight": [
        "National Police Service Act, 2011", "already present",
        "Independent Policing Oversight Authority Act, 2011"
    ],
    "Bail and bond": ["Bail and Bond Policy Guidelines", "already present"],
    "Penal offences": ["Penal Code (Cap. 63)", "already present"],
    "Small claims": ["Small Claims Court Act, 2016"],
    "Traffic offences": ["Traffic Act (Cap. 403)"],
    "Affidavits / oaths (matches your document generator)": [
        "Oaths and Statutory Declarations Act (Cap. 15)"
    ],
}

ingested_lower = " ".join(source_counts.keys()).lower()

for topic, docs in reference_topics.items():
    status_parts = []
    for doc in docs:
        if doc == "already present":
            continue
        key_terms = doc.lower().split("(")[0].strip().split(",")[0].strip()
        first_words = " ".join(key_terms.split()[:3])
        present = first_words.lower() in ingested_lower
        status_parts.append((doc, present))

    if not status_parts:
        continue

    any_present = any(p for _, p in status_parts)
    marker = "✓ covered" if any_present else "✗ GAP"
    print(f"\n[{marker}] {topic}")
    for doc, present in status_parts:
        tag = "have" if present else "missing"
        print(f"    ({tag}) {doc}")

print()
print("=" * 70)
print("Note: this reference list is illustrative, not exhaustive, and the")
print("presence check is a rough substring match — verify manually before")
print("concluding a document is truly absent. Use this to prioritize what")
print("to ingest next based on the questions your actual users ask.")
print("=" * 70)
