"""
Compares the actual context size (number of chunks, total characters,
rough token estimate) that gets built for two specific queries — one that
times out consistently, one that's intermittent — to test whether prompt
size is the real driver of your timeout pattern, now that reload cost and
keep-alive are ruled out.

Usage:
    python compare_context_size.py
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

queries = [
    "steps for withdrawing case",   # consistently times out
    "rights of arrested persons",   # intermittent
]

for q in queries:
    print("=" * 70)
    print(f"Q: {q}")
    print("=" * 70)

    raw_results = rag.search_database(q, top_k=8)
    relevant = rag._filter_relevant(raw_results)
    relevant = sorted(relevant, key=lambda r: r.get("distance", 999))[:5]

    context = rag._build_context(relevant)

    print(f"  Chunks retrieved (raw):      {len(raw_results)}")
    print(f"  Chunks passing threshold:    {len(relevant)}")
    print(f"  Final context length:       {len(context)} characters")
    # Rough rule of thumb: ~4 characters per token for English legal text
    print(f"  Rough estimated tokens:      ~{len(context) // 4}")
    print()
    for i, r in enumerate(relevant):
        source = r.get("metadata", {}).get("source", "Unknown")
        doc_len = len(r.get("document", ""))
        distance = r.get("distance")
        print(f"    [{i+1}] {doc_len:5d} chars  distance={distance}  {source}")
    print()

print("=" * 70)
print("If 'steps for withdrawing case' has a meaningfully larger context")
print("(more chunks passing threshold, and/or more total characters) than")
print("'rights of arrested persons', that's a real, fixable lever:")
print("reducing MAX_RELEVANT_DISTANCE slightly, or capping context to")
print("3 chunks instead of 5, would directly cut its prompt size and")
print("likely its timeout rate, in a way that's now well-targeted rather")
print("than a guess.")
print("=" * 70)
