"""
Finds the specific chunk containing "25. Sentence of death" text and
prints its COMPLETE metadata dict — every key, not just a curated subset
— so we can see the 'title' field and confirm why section_number/
subsection_number are blank for it.

Usage:
    python find_chunk_title.py D:\\projects\\kenyan-legal-bot\\backend\\data\\chroma_db
"""

import sys
import chromadb

if len(sys.argv) != 2:
    print("Usage: python find_chunk_title.py <path_to_chroma_db>")
    sys.exit(1)

db_path = sys.argv[1]
print(f"Opening collection at: {db_path}")

client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection("kenyan_law")

all_docs = collection.get(include=["documents", "metadatas"])
documents = all_docs.get("documents", [])
metadatas = all_docs.get("metadatas", [])
ids = all_docs.get("ids", [])

target_phrase = "25. Sentence of death"

found = 0
for i, doc in enumerate(documents):
    if target_phrase in doc:
        found += 1
        print("\n" + "=" * 70)
        print(f"MATCH {found} — chunk id: {ids[i]}")
        print("=" * 70)
        print("Full metadata:")
        meta = metadatas[i] or {}
        for key, value in meta.items():
            print(f"  {key!r}: {value!r}")
        print(f"\nFull text ({len(doc)} chars):")
        print(doc[:600])
        if len(doc) > 600:
            print(f"... [{len(doc) - 600} more characters]")

if found == 0:
    print(f"\nNo chunk found containing '{target_phrase}'. Try a shorter/")
    print("different substring if the exact phrase doesn't match verbatim.")
else:
    print(f"\n\nTotal matches: {found}")
