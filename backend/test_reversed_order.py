"""
Runs the same two queries in REVERSED order from your usual test_search.py
to check whether the timeout follows the QUERY or the POSITION (first vs
second call after script startup).

Usage:
    python test_reversed_order.py
"""

import sys
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

# Reversed order vs your usual test_search.py
questions = [
    "rights of arrested persons",   # now goes FIRST
    "steps for withdrawing case",   # now goes SECOND
]

for q in questions:
    print("=" * 60)
    print(f"Q: {q}")
    print("=" * 60)
    start = time.time()
    result = rag.generate_answer(q)
    elapsed = time.time() - start
    print(result)
    print(f"\n[took {elapsed:.1f}s]")
    print()
