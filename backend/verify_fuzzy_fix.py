#!/usr/bin/env python3
"""
verify_fuzzy_fix.py

Cheap, Ollama-free confirmation that the _fuzzy_quote_match fix actually
works against your REAL Section 14 chunk text -- not a synthetic
reconstruction of how the sentence might continue (the anchor-window fix
was verified against a guessed continuation; this checks it against the
real one).

Reuses check_source_text.py's exact-match lookup to pull the real chunk,
then calls the REAL _fuzzy_quote_match from your actual rag_engine.py
(not a reimplementation) with the exact quote that was flagged in your
log, to confirm it now passes.

No Ollama required, no retrieval replay required -- this only exercises
the one specific piece of logic that changed. If this passes, the fix is
confirmed against ground truth; if it still fails, we've isolated the
problem to something other than the anchor-window logic, before spending
any time on a live end-to-end run.

USAGE
    python3 verify_fuzzy_fix.py --db-path D:\\projects\\kenyan-legal-bot\\backend\\data\\chroma_db
"""

import argparse
import re
import sys
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-path", required=True, help="Path to the chroma_db folder")
    parser.add_argument("--collection", default="kenyan_law")
    parser.add_argument("--rag-engine-dir", default=None,
                         help="Folder containing rag_engine.py, if not the same folder as this script")
    args = parser.parse_args()

    engine_dir = Path(args.rag_engine_dir) if args.rag_engine_dir else Path(__file__).parent
    sys.path.insert(0, str(engine_dir))

    try:
        import chromadb
    except ImportError:
        print("chromadb not installed here -- run this on the machine with your real DB.")
        return 1

    try:
        from logic.rag_engine import RAGEngine
    except ImportError as e:
        print(f"Could not import rag_engine.py from {engine_dir}: {e}")
        return 1

    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_collection(args.collection)
    data = collection.get(include=["documents", "metadatas"])

    # Find the real Section 14 / Matrimonial Property Act chunk -- same
    # identification approach as check_source_text.py, but here we keep
    # the FULL text (no 220-char snippet truncation) since we need the
    # real continuation, not a preview of it.
    target_doc = None
    for doc, meta in zip(data.get("documents", []), data.get("metadatas", [])):
        if (meta.get("source", "").strip() == "Matrimonial Property Act"
                and str(meta.get("section_number", "")) == "14"):
            target_doc = doc
            break

    if not target_doc:
        print("Could not find the Matrimonial Property Act, Section 14 chunk in this DB.")
        print("Available Matrimonial Property Act sections found:")
        for doc, meta in zip(data.get("documents", []), data.get("metadatas", [])):
            if meta.get("source", "").strip() == "Matrimonial Property Act":
                print(f"  Section {meta.get('section_number')}")
        return 1

    print("=== REAL CHUNK TEXT (full, not truncated) ===")
    print(target_doc)
    print()

    # The exact phrase that was actually flagged as ungrounded in your log.
    flagged_quote = "rebuttable presumption that the property is held in trust for the other spouse."

    # RAGEngine's methods are plain functions on the class that don't
    # touch self except for the class-level regex/threshold constants,
    # so we can call _fuzzy_quote_match directly without a fully loaded
    # engine (no DB load, no embeddings, no Ollama needed for this check).
    engine = object.__new__(RAGEngine)

    exact = flagged_quote.lower() in normalize(target_doc)
    fuzzy = engine._fuzzy_quote_match(flagged_quote.lower(), normalize(target_doc))

    print(f'Flagged quote: "{flagged_quote}"')
    print(f"Exact substring match: {exact}")
    print(f"_fuzzy_quote_match result (the actual method used in production): {fuzzy}")
    print()
    if fuzzy:
        print("PASS -- the fix resolves this exact real-world case against your real corpus text.")
    else:
        print("STILL FAILING -- the anchor-window fix does not cover this real case. Needs another look")
        print("before trusting it in a live run -- worth pasting the real chunk text printed above so I")
        print("can see exactly how it diverges from the flagged quote.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
