"""
Two-part test for the "robbery with violence" / "seven years" case.

PART 1 — log check: did this query actually pass the grounding gate on
attempt 1, get rescued on attempt 2, or fail and get rejected? Tells you
whether this is a substring-matching blind spot or a bug in the retry flow.

PART 2 — raw retrieval check: pulls the actual documents that would be
retrieved for this query and shows every place "seven years" appears in
them, so you can see with your own eyes what it's really describing.

USAGE:
    python3 test_seven_years_grounding.py /path/to/app.log

    (Part 2 will attempt to import your live RAGEngine to run a real
    search_database call. If the import path doesn't match your project
    layout, edit the IMPORT SECTION below — it's marked clearly — and
    re-run. Part 1 will still run fine even if Part 2's import fails.)
"""

import sys
import re

QUERY_SUBSTRING = "robbery with violence"
PHRASE_TO_TRACE = "seven years"


# ============================================
# PART 1 — log check
# ============================================

def check_log(log_path: str):
    print("=" * 60)
    print("PART 1: LOG CHECK")
    print("=" * 60)

    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Could not open log file '{log_path}': {e}")
        return

    # Grab every GROUNDING_ISSUES_FOUND / GROUNDING_ISSUES_PERSIST /
    # OLLAMA / HONEST_NO_ANSWER line that mentions this query, in order,
    # so we can see the full sequence of what happened to it.
    relevant = [
        line.strip() for line in lines
        if QUERY_SUBSTRING in line and (
            "GROUNDING_ISSUES" in line
            or "OLLAMA_NO_ANSWER" in line
            or "HONEST_NO_ANSWER" in line
        )
    ]

    if not relevant:
        print(f"No GROUNDING_ISSUES / OLLAMA_NO_ANSWER / HONEST_NO_ANSWER lines "
              f"mentioning '{QUERY_SUBSTRING}' were found in this log.")
        print("This likely means either:")
        print("  - this query never hit a logged issue (passed validation")
        print("    cleanly on attempt 1), or")
        print("  - the log file you pointed at doesn't cover the time this")
        print("    query ran.")
        return

    found_count = sum(1 for l in relevant if "GROUNDING_ISSUES_FOUND" in l)
    persist_count = sum(1 for l in relevant if "GROUNDING_ISSUES_PERSIST" in l)

    print(f"Found {len(relevant)} relevant log line(s) for this query:\n")
    for line in relevant:
        print(f"  {line}")

    print()
    if persist_count > 0:
        print("RESULT: This query PERSISTED after retry and should have")
        print("returned UNAVAILABLE_MESSAGE. If you actually received a full")
        print("answer with 'seven years' in it, that's a real bug in the")
        print("retry/return flow, not just a substring-matching gap — worth")
        print("checking generate_answer's control flow directly.")
    elif found_count > 0:
        print("RESULT: This query FAILED attempt 1, then passed after the")
        print("correction retry. The corrected answer still contains the")
        print("fabricated phrase, so the retry moved/rephrased the claim")
        print("without actually removing the ungrounded number.")
    else:
        print("RESULT: No GROUNDING_ISSUES line at all — this answer passed")
        print("validation cleanly on the very first attempt. That means")
        print(f"'{PHRASE_TO_TRACE}' really was found as a literal substring")
        print("somewhere in the retrieved context. Proceed to Part 2 to see")
        print("what it's actually attached to.")


# ============================================
# PART 2 — raw retrieval check
# ============================================

def check_retrieval():
    print("\n" + "=" * 60)
    print("PART 2: RAW RETRIEVAL CHECK")
    print("=" * 60)

    # --- IMPORT SECTION: adjust this to match your project layout ---
    # Try a couple of common locations before giving up.
    eng = None
    import_errors = []
    for module_path in ("logic.rag_engine", "rag_engine", "app.logic.rag_engine"):
        try:
            module = __import__(module_path, fromlist=["RAGEngine"])
            RAGEngine = getattr(module, "RAGEngine")
            eng = RAGEngine()
            print(f"Imported RAGEngine from '{module_path}'.")
            break
        except Exception as e:
            import_errors.append(f"{module_path}: {e}")

    if eng is None:
        print("Could not import RAGEngine from any of the usual locations:")
        for err in import_errors:
            print(f"  - {err}")
        print("\nEdit the IMPORT SECTION in this script to match your actual")
        print("project layout (the module path that contains `class RAGEngine`),")
        print("then re-run.")
        return

    print("Waiting for the vector DB to finish loading (up to 30s)...")
    if not eng.wait_for_loading(30):
        print("DB did not finish loading in time — can't run a live search.")
        return

    # The DB and the embeddings model are two separate things that both
    # need to be ready. wait_for_loading() only covers the DB thread
    # inside RAGEngine itself — embeddings_service is a separate global
    # object, and if your real app runs some startup/warm-up step on it
    # (e.g. loading a sentence-transformer model) that only happens in
    # your actual entry point, this standalone script won't trigger it.
    # Poll for a bit before giving up, in case it's just slow to init.
    import time
    try:
        from logic.embeddings import embeddings_service
    except Exception as e:
        print(f"Could not import embeddings_service directly: {e}")
        embeddings_service = None

    if embeddings_service is not None:
        if not embeddings_service.is_ready():
            print("embeddings_service not ready yet — calling .load() directly,")
            print("matching what your app's real lifespan() startup does...")
            try:
                embeddings_service.load()
            except Exception as e:
                print(f"embeddings_service.load() raised: {e}")

        print("Waiting for embeddings_service to become ready (up to 20s)...")
        waited = 0
        while not embeddings_service.is_ready() and waited < 20:
            time.sleep(1)
            waited += 1

        if not embeddings_service.is_ready():
            print("\nembeddings_service still never became ready after calling")
            print(".load() directly and waiting. Check logic/embeddings.py for")
            print("what .load() actually needs (a model file path, a download,")
            print("a specific working directory, etc.) — something about running")
            print("standalone is likely missing an environment assumption that")
            print("main.py's process satisfies.")
            return

    query = "robbery with violence"
    results = eng.search_database(query, top_k=8)
    print(f"\nRetrieved {len(results)} chunk(s) for query: '{query}'\n")

    matches_found = 0
    for i, r in enumerate(results, start=1):
        doc = r.get("document", "")
        source = r.get("metadata", {}).get("source", "Unknown")
        distance = r.get("distance")
        if PHRASE_TO_TRACE.lower() in doc.lower():
            matches_found += 1
            idx = doc.lower().find(PHRASE_TO_TRACE.lower())
            start = max(0, idx - 150)
            end = min(len(doc), idx + 150)
            print(f"[Chunk {i}] source={source} distance={distance}")
            print(f"  ...{doc[start:end]}...")
            print()

    if matches_found == 0:
        print(f"'{PHRASE_TO_TRACE}' was NOT found in any of the {len(results)} "
              f"retrieved chunks for this query.")
        print("That would be surprising given the log result above — if Part 1")
        print("said this passed validation cleanly, the phrase must be coming")
        print("from somewhere else (e.g. a different top_k, or case-law results")
        print("via retrieve_examples() rather than the main statute search).")
        print("Try also checking: eng.retrieve_examples(query, top_k=8)")
    else:
        print(f"Found '{PHRASE_TO_TRACE}' in {matches_found} chunk(s) above.")
        print("Check whether the surrounding text is actually about robbery")
        print("with violence, or about something else entirely (a different")
        print("offense, a procedural time limit, an unrelated case detail,")
        print("etc). If it's unrelated, that confirms the substring-matching")
        print("blind spot: the phrase existed *somewhere* in context, but was")
        print("never actually entailed by that text for this claim.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_seven_years_grounding.py /path/to/app.log")
        sys.exit(1)

    check_log(sys.argv[1])
    check_retrieval()
