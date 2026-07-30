"""
Diagnostic — checks whether "withdrawing a case" content actually exists
anywhere in the database, and if so, why it's not surfacing within the
current distance threshold.

Usage:
    python diagnose_withdrawal.py
"""

import sys
import re
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from logic.rag_engine import RAGEngine
from logic.embeddings import embeddings_service

print("Loading...")
embeddings_service.load()
rag = RAGEngine()

while not rag.is_ready():
    time.sleep(0.5)

print(f"Database: {rag.collection.count()} documents\n")

# 1. Direct text search across ALL chunks for "withdraw" — bypasses
#    embeddings entirely, so this tells us if the content exists at all.
all_docs = rag.collection.get(include=["documents", "metadatas"])

print("=" * 70)
print("Direct text search for 'withdraw' across all chunks")
print("=" * 70)
hits = []
for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
    if re.search(r'withdraw', doc, re.IGNORECASE):
        hits.append((doc, meta))

print(f"Chunks containing 'withdraw': {len(hits)}\n")
for doc, meta in hits[:5]:
    source = meta.get("source", "Unknown")
    idx = doc.lower().find("withdraw")
    start = max(0, idx - 80)
    end = min(len(doc), idx + 300)
    print(f"[{source}]")
    print(f"  ...{doc[start:end]}...")
    print()

# 2. Same, but for "discontinu" (legal term often used instead of "withdraw"
#    for civil cases — "discontinuance" in Kenyan civil procedure)
print("=" * 70)
print("Direct text search for 'discontinu' across all chunks")
print("=" * 70)
hits2 = []
for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
    if re.search(r'discontinu', doc, re.IGNORECASE):
        hits2.append((doc, meta))

print(f"Chunks containing 'discontinu': {len(hits2)}\n")
for doc, meta in hits2[:5]:
    source = meta.get("source", "Unknown")
    idx = doc.lower().find("discontinu")
    start = max(0, idx - 80)
    end = min(len(doc), idx + 300)
    print(f"[{source}]")
    print(f"  ...{doc[start:end]}...")
    print()

# 3. What did embedding search actually return for the original query,
#    regardless of threshold? (top_k high to see the full picture)
print("=" * 70)
print("Embedding search results for 'steps for withdrawing case' (unfiltered)")
print("=" * 70)
results = rag.search_database("steps for withdrawing case", top_k=10)
for i, r in enumerate(results):
    source = r.get("metadata", {}).get("source", "Unknown")
    distance = r.get("distance")
    snippet = r.get("document", "")[:150].replace("\n", " ")
    print(f"  [{i+1}] distance={distance}  source={source}")
    print(f"      {snippet}...")

print()
print("=" * 70)
print("If 'withdraw' hits above show relevant content but the embedding")
print("search above doesn't surface them in range, this is a wording/")
print("vocabulary mismatch (like 'discontinuance' vs 'withdraw'), similar")
print("to the Article 49 case. If NO chunks contain 'withdraw' or")
print("'discontinu' at all, your source PDFs likely don't cover this")
print("topic, and the 'unavailable' response is actually correct.")
print("=" * 70)
