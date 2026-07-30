"""
Two-part diagnostic to settle, with certainty, why check_ejusdem.py
returned zero case-law chunks.

PART 1 — confirms which rag_engine.py is ACTUALLY being imported right
now, and prints its live DOC_TYPE_FIELD/CASE_LAW_TYPE values. If these
still say "doc_type"/"case_law" instead of "category"/"case_law_pdf",
the updated file hasn't actually been deployed/restarted yet — full
stop, that's the answer, no need to look further.

PART 2 — if the constants ARE correct, this bypasses RAGEngine's search
entirely and asks the raw ChromaDB collection: what metadata "category"
values actually exist across every stored document, and how many of
each? This tells us definitively whether case-law documents exist in
the collection at all, and what they're really tagged with — rather
than continuing to guess.

Run from your backend directory:
    python diagnose_case_law.py
"""

import time
from collections import Counter

from logic.rag_engine import RAGEngine
from logic.embeddings import embeddings_service


def part1_confirm_loaded_version():
    print("=" * 60)
    print("PART 1: WHICH rag_engine.py IS ACTUALLY LOADED")
    print("=" * 60)
    print(f"Imported from: {RAGEngine.__module__} -> ", end="")
    import logic.rag_engine as re_module
    print(re_module.__file__)
    print(f"DOC_TYPE_FIELD = {RAGEngine.DOC_TYPE_FIELD!r}")
    print(f"CASE_LAW_TYPE  = {RAGEngine.CASE_LAW_TYPE!r}")

    if RAGEngine.DOC_TYPE_FIELD != "category" or RAGEngine.CASE_LAW_TYPE != "case_law_pdf":
        print("\n*** STOP: this is still the OLD version of rag_engine.py. ***")
        print("The file on disk hasn't been updated/restarted with the fix yet.")
        print("Nothing past this point will be meaningful until that's resolved —")
        print("replace logic/rag_engine.py with the latest version and restart")
        print("the server (and re-run this diagnostic standalone too) before")
        print("continuing.")
        return False

    print("\nConstants look correct. Proceeding to Part 2...")
    return True


def part2_raw_metadata_inspection(eng):
    print("\n" + "=" * 60)
    print("PART 2: RAW METADATA CATEGORIES IN THE LIVE COLLECTION")
    print("=" * 60)

    if not eng.collection:
        print("No collection available — is the DB actually loaded?")
        return

    all_docs = eng.collection.get(include=["metadatas"])
    metadatas = all_docs.get("metadatas", [])
    total = len(metadatas)
    print(f"Total documents in collection: {total}\n")

    category_counts = Counter()
    missing_category = 0
    for m in metadatas:
        if not m or "category" not in m:
            missing_category += 1
        else:
            category_counts[m.get("category")] += 1

    print("Counts by 'category' metadata value:")
    for cat, count in category_counts.most_common():
        print(f"  {cat!r}: {count}")
    if missing_category:
        print(f"  (no 'category' key at all): {missing_category}")

    case_law_count = category_counts.get("case_law_pdf", 0)
    if case_law_count == 0:
        print("\n*** No documents are tagged category='case_law_pdf'. ***")
        print("This means case law was either never ingested, or was ingested")
        print("under a different category value than expected. Either way,")
        print("retrieve_examples() legitimately has nothing to return right now —")
        print("it isn't a bug in the filter, there's simply no case-law data")
        print("tagged the way the code expects.")
        print("\nNext step: check whether you've actually run case-law ingestion")
        print("at all (a separate step from the statute PDFs), and if so, print")
        print("a few raw metadata dicts from documents whose 'source' or")
        print("'title' looks like a case name to see what category they")
        print("actually got tagged with.")
    else:
        print(f"\n{case_law_count} document(s) ARE tagged 'case_law_pdf' — showing")
        print("a few samples to sanity-check their metadata:")
        shown = 0
        for m in metadatas:
            if m and m.get("category") == "case_law_pdf":
                print(f"  {dict(m)}")
                shown += 1
                if shown >= 5:
                    break


if __name__ == "__main__":
    ok = part1_confirm_loaded_version()
    if not ok:
        exit(1)

    embeddings_service.load()
    eng = RAGEngine()
    eng.wait_for_loading(30)

    part2_raw_metadata_inspection(eng)
