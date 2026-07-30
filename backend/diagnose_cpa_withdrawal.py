"""
Diagnostic — searches specifically within Civil Procedure Act chunks for
the actual withdrawal-of-suit provision, to see if it exists, and if so,
why "steps for withdrawing case" doesn't surface it within threshold.

Usage:
    python diagnose_cpa_withdrawal.py
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

all_docs = rag.collection.get(include=["documents", "metadatas"])

cpa_chunks = []
for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
    if meta and "civil procedure act" in str(meta.get("source", "")).lower():
        cpa_chunks.append((doc, meta))

print(f"Total Civil Procedure Act chunks: {len(cpa_chunks)}\n")

# Print every chunk's first 100 chars so you can see overall coverage/order
print("=" * 70)
print("All CPA chunks (first 100 chars each, in storage order)")
print("=" * 70)
for i, (doc, meta) in enumerate(cpa_chunks):
    print(f"[{i}] chunk_index={meta.get('chunk_index')}  len={len(doc)}")
    print(f"    {doc[:100]}...")
print()

# Search specifically for "withdraw" within CPA chunks only
print("=" * 70)
print("CPA chunks containing 'withdraw'")
print("=" * 70)
found_any = False
for i, (doc, meta) in enumerate(cpa_chunks):
    if re.search(r'withdraw', doc, re.IGNORECASE):
        found_any = True
        idx = doc.lower().find("withdraw")
        start = max(0, idx - 100)
        end = min(len(doc), idx + 400)
        print(f"Chunk [{i}] chunk_index={meta.get('chunk_index')}")
        print(f"  ...{doc[start:end]}...")
        print()

if not found_any:
    print("No CPA chunk contains the word 'withdraw' at all.")
    print("This means either: (a) the source PDF's withdrawal provision")
    print("uses different wording entirely, (b) it fell in the skipped")
    print("TOC pages (first 12 of 36), or (c) chunking split it badly.")
    print()
    print("Checking page coverage: first chunk starts with this text,")
    print("which tells us how much of the document was actually captured")
    print("after TOC-skipping:")
    print(repr(cpa_chunks[0][0][:300]) if cpa_chunks else "NO CHUNKS AT ALL")
