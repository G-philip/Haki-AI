"""
Calibration script — run this against your real, loaded database to find
the correct MAX_RELEVANT_DISTANCE for rag_engine.py.

Usage:
    python calibrate_threshold.py

It prints the full ranked distance list for a set of test queries you
already know the right answer for (edit the QUERIES list below to match
known-good test cases from your own data). Look at where genuinely
relevant documents (read the printed snippet) stop showing up, and set
MAX_RELEVANT_DISTANCE in rag_engine.py just above that point.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from logic.rag_engine import RAGEngine
from logic.embeddings import embeddings_service
import time

print("Loading...")
embeddings_service.load()
rag = RAGEngine()

while not rag.is_ready():
    time.sleep(0.5)

print(f"Database: {rag.collection.count()} documents\n")

# Edit these to queries where you already know which document SHOULD
# come back (ideally ones you manually verified earlier).
QUERIES = [
    "what does article 49 say",
    "what to do when arrested by police",
    "rights of arrested persons",
]

for q in QUERIES:
    print("=" * 70)
    print(f"Q: {q}")
    print("=" * 70)
    # top_k high on purpose so you can see the full spread, not just top 5
    results = rag.search_database(q, top_k=10)
    for i, r in enumerate(results):
        source = r.get("metadata", {}).get("source", "Unknown")
        distance = r.get("distance")
        snippet = r.get("document", "")[:120].replace("\n", " ")
        print(f"  [{i+1}] distance={distance:.4f}  source={source}")
        print(f"      {snippet}...")
    print()

print("=" * 70)
print("Look at the distances above. For each query, find the distance of")
print("the LAST result that is still genuinely relevant/correct. Take the")
print("highest such value across all your test queries and use that (plus")
print("a small margin, e.g. +0.05) as MAX_RELEVANT_DISTANCE in rag_engine.py.")
print("=" * 70)
