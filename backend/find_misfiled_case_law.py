"""
Finds documents currently tagged category='kenyan_law_pdf' whose
source/filename looks like a case name (contains " v ", " vs ",
"versus", or "Republic") rather than a statute — i.e. judgments that
were ingested through the wrong pipeline (_ingest_pdf instead of
_ingest_case_law_pdf) and are masquerading as statute text.

Run from your backend directory:
    python find_misfiled_case_law.py
"""

import re
from logic.rag_engine import RAGEngine
from logic.embeddings import embeddings_service

CASE_LIKE_PATTERN = re.compile(
    r'\bv\.?\s|\bvs\.?\s|\bversus\b|\brepublic\b', re.IGNORECASE
)

embeddings_service.load()
eng = RAGEngine()
eng.wait_for_loading(30)

all_docs = eng.collection.get(include=["metadatas"])
metadatas = all_docs.get("metadatas", [])

misfiled_sources = set()
for m in metadatas:
    if not m or m.get("category") != "kenyan_law_pdf":
        continue
    source = m.get("source", "") or m.get("filename", "")
    if CASE_LIKE_PATTERN.search(source):
        misfiled_sources.add(source)

print(f"Found {len(misfiled_sources)} distinct source file(s) tagged "
      f"'kenyan_law_pdf' that look like case names:\n")
for s in sorted(misfiled_sources):
    print(f"  - {s}")

if not misfiled_sources:
    print("None found by this heuristic — the case citations may be coming")
    print("from somewhere else entirely, or use naming patterns this script")
    print("doesn't catch. Worth double-checking filenames in data/pdfs by hand.")
else:
    print(f"\nThese {len(misfiled_sources)} file(s) need to be:")
    print("  1. Moved to a separate folder (e.g. data/case_law_pdfs)")
    print("  2. Re-ingested via ingestor.ingest_case_law_folder(...) instead")
    print("     of ingest_folder(...)")
    print("  3. Their current (mistagged) chunks removed from the collection")
