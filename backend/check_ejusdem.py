"""Quick check: does the Mwaura judgment chunk actually contain anything
resembling 'ejusdem generis' or a 'read together' holding about sections
296(1)/296(2), or was that claim fabricated?

Run from your backend directory:
    python check_ejusdem.py
"""
import time
from logic.rag_engine import RAGEngine
from logic.embeddings import embeddings_service

embeddings_service.load()
eng = RAGEngine()
eng.wait_for_loading(30)

# Search case law specifically (now that DOC_TYPE_FIELD/CASE_LAW_TYPE
# actually match real metadata) for the Mwaura case content.
results = eng.search_database(
    "Mwaura robbery with violence section 296", top_k=8,
    doc_type=eng.CASE_LAW_TYPE
)

print(f"Retrieved {len(results)} case-law chunk(s).\n")
for i, r in enumerate(results, 1):
    doc = r.get("document", "")
    meta = r.get("metadata", {})
    print(f"[{i}] {meta.get('case_name', meta.get('source', 'Unknown'))} (distance={r.get('distance')})")
    if "ejusdem" in doc.lower():
        idx = doc.lower().find("ejusdem")
        print(f"    FOUND 'ejusdem' at: ...{doc[max(0,idx-100):idx+200]}...")
    else:
        print("    'ejusdem' NOT found in this chunk.")
    print()
