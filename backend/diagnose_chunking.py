"""
Diagnostic — checks whether Article 49 (or any given article number) is
fragmented across chunk boundaries in your live Chroma collection, or
missing/merged with unrelated content.

Usage:
    python diagnose_chunking.py
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

# Pull every chunk from the Constitution of Kenya 2010 and scan for
# where (and whether) "49." actually appears, and what surrounds it.
all_docs = rag.collection.get(include=["documents", "metadatas"])

constitution_chunks = []
for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
    if meta and "constitution" in str(meta.get("source", "")).lower():
        constitution_chunks.append((doc, meta))

print(f"Found {len(constitution_chunks)} Constitution chunks\n")

import re

# Look for chunks that contain "49." near "arrested" (the real Article 49)
hits = []
for i, (doc, meta) in enumerate(constitution_chunks):
    if re.search(r"rights of arrested persons", doc, re.IGNORECASE) or re.search(r"\b49\.\s*\(1\)\s*An arrested", doc):
        hits.append((i, doc, meta))

if not hits:
    print("No chunk found containing the actual Article 49 heading/text.")
    print("Searching more loosely for any chunk mentioning '49.' near arrest-related words...")
    for i, (doc, meta) in enumerate(constitution_chunks):
        if "49." in doc and ("arrest" in doc.lower() or "silent" in doc.lower()):
            hits.append((i, doc, meta))

print(f"\nChunks matching Article 49 content: {len(hits)}\n")
for i, doc, meta in hits:
    print("=" * 70)
    print(f"Chunk #{i}  |  title={meta.get('title')}  |  length={len(doc)} chars")
    print("=" * 70)
    # Show where in the chunk "49" actually sits, and what's around it
    idx = doc.find("49.")
    if idx == -1:
        idx = 0
    start = max(0, idx - 100)
    end = min(len(doc), idx + 400)
    print(f"...{doc[start:end]}...")
    print()

if not hits:
    print("\nNo chunk contains Article 49's actual text at all.")
    print("This means ingestion never produced a usable chunk for it —")
    print("the content may be split across other chunks, or lost during")
    print("chunking/cleaning. Re-ingestion with structure-aware chunking")
    print("is needed.")
