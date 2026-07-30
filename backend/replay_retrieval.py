#!/usr/bin/env python3
"""
replay_retrieval.py

Follow-up to check_source_text.py. That script confirmed a phrase is
genuinely, verbatim in the ingested corpus -- so the open question is no
longer "is this real," it's "why didn't the pipeline use it that day."

This script answers that by reusing your ACTUAL RAGEngine class -- the
real search_database(), the real _filter_relevant(), the real top-5 cap --
rather than reimplementing the retrieval logic separately. Reimplementing
it here would risk the diagnostic quietly drifting from what production
actually does, which defeats the point. (This is the opposite tradeoff
from check_source_text.py, which deliberately reimplemented fuzzy-matching
independently so it couldn't share a bug with the validator it was
checking -- there, independence was the goal; here, exact fidelity is.)

It does NOT need Ollama running -- RAGEngine's constructor only loads the
Chroma DB and embeddings, so this only exercises retrieval, not generation.

USAGE

    python3 replay_retrieval.py "After i got married, i inherited land under my name . can my wife claim it during divorce?" --contains "held in trust"

    # Or match by source/section metadata instead of/as well as text content
    python3 replay_retrieval.py "some query" --source "Matrimonial Property Act" --section 14

WHAT IT PRINTS
    - Every one of the raw top-8 chunks returned by the real embedding
      search, in rank order, with distance
    - For each: whether it matched your --contains/--source/--section
      target, whether it survived _filter_relevant (the absolute ceiling
      + relative margin), and whether it made the final top-5 cap that
      actually gets sent to the model
    - A one-line verdict: IN CONTEXT / FILTERED OUT (too far from best
      match) / CAPPED OUT (relevant but ranked 6th-8th) / NOT RETRIEVED
      AT ALL (didn't even make the raw top 8)

That last distinction matters: "filtered out" and "capped out" are both
retrieval-quality problems fixable by adjusting the retrieval side
(RELEVANCE_MARGIN, top_k, embeddings). "Not retrieved at all" is a
bigger gap -- it means even an uncapped, unfiltered top-8 search doesn't
consider this chunk close enough to be worth returning, which points
more at the embedding model or query phrasing than at any of the
threshold constants.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="The exact user query to replay through retrieval")
    parser.add_argument("--contains", default=None,
                         help="Substring (case-insensitive) to look for in each chunk's text, "
                              "to identify the target chunk")
    parser.add_argument("--source", default=None,
                         help="Match chunks whose metadata 'source' contains this (e.g. 'Matrimonial Property Act')")
    parser.add_argument("--section", default=None,
                         help="Match chunks whose metadata 'section_number' equals this (e.g. '14')")
    parser.add_argument("--rag-engine-dir", default=None,
                         help="Folder containing rag_engine.py, if not the same folder as this script")
    parser.add_argument("--timeout", type=int, default=60,
                         help="Seconds to wait for the DB to load before giving up. Your production log "
                              "shows this genuinely taking ~20-24s even in the real app, so the default "
                              "here is generous. Default: 60")
    args = parser.parse_args()

    if not (args.contains or args.source or args.section):
        parser.error("Give at least one of --contains, --source, or --section to identify the target chunk.")

    engine_dir = Path(args.rag_engine_dir) if args.rag_engine_dir else Path(__file__).parent
    sys.path.insert(0, str(engine_dir))

    try:
        from logic.rag_engine import RAGEngine
    except ImportError as e:
        print(f"Could not import rag_engine.py from {engine_dir}: {e}")
        print("Pass the correct folder with --rag-engine-dir /path/to/backend")
        return 1

    print(f"Loading RAGEngine (Chroma DB + embeddings only -- Ollama not required for this check)...")
    import time
    start = time.time()
    engine = RAGEngine()
    # Poll in short slices instead of one blocking wait_for_loading(N) call,
    # so a slow-but-still-progressing load is visible rather than looking
    # identical to a fully stuck one.
    while not engine.is_ready() and (time.time() - start) < args.timeout:
        elapsed = time.time() - start
        print(f"  ...still loading ({elapsed:.0f}s elapsed)")
        time.sleep(3)

    if not engine.is_ready():
        print(f"\nDB did not finish loading in {args.timeout}s -- aborting.")
        print("\nMost likely cause: your backend server (main.py / uvicorn) is still running and")
        print("has this same Chroma DB file open. Chroma's persistent store is SQLite-backed, and")
        print("two separate processes opening the same DB file concurrently is a common source of")
        print("exactly this kind of silent hang on Windows. Try stopping the backend server first,")
        print("then re-run this script against the DB directly.")
        print(f"\nIf that's not it, try a longer --timeout (production's own log showed ~20-24s just")
        print("to open this DB even with nothing else contending for it).")
        return 1

    # RAGEngine.__init__ only loads the Chroma DB -- it does NOT trigger
    # embeddings_service's own model loading. Your production main.py does
    # that as an explicit separate step before logging EMBEDDINGS_LOADED,
    # which this script doesn't know the name of. Rather than guess, try
    # common method names, and if none exist, introspect the object so we
    # can see the real one instead of failing blind a second time.
    try:
        from logic.embeddings import embeddings_service
    except ImportError as e:
        print(f"Could not import logic.embeddings: {e}")
        return 1

    if not embeddings_service.is_ready():
        print("\nembeddings_service exists but isn't ready -- attempting common init method names...")
        loaded = False
        for method_name in ("load", "initialize", "init", "warm_up", "warmup", "start", "setup"):
            method = getattr(embeddings_service, method_name, None)
            if callable(method):
                print(f"  trying embeddings_service.{method_name}()...")
                try:
                    method()
                except Exception as e:
                    print(f"    {method_name}() raised: {e}")
                    continue
                if embeddings_service.is_ready():
                    print(f"  embeddings_service.{method_name}() worked.")
                    loaded = True
                    break

        if not loaded and not embeddings_service.is_ready():
            print("\nNone of the common method names worked. Here's what's actually on the object,")
            print("so we can find the right one instead of guessing further:")
            public_attrs = [a for a in dir(embeddings_service) if not a.startswith("_")]
            print(f"  {public_attrs}")
            print("\nTell me which of these looks like the load/init call (or share logic/embeddings.py)")
            print("and I'll wire it in directly rather than trying names blind.")
            return 1

    print(f"\nQuery: \"{args.query}\"\n")

    # Exactly what generate_answer() does for the primary statute search.
    raw_results = engine.search_database(args.query, top_k=8, exclude_doc_type=engine.CASE_LAW_TYPE)
    if not raw_results:
        print("search_database returned NOTHING for this query -- nothing to analyze.")
        return 0

    relevant = engine._filter_relevant(raw_results)
    relevant_sorted = sorted(relevant, key=lambda r: r.get("distance", 999))
    top5 = relevant_sorted[:5]

    # Identity, not text-content, is what decides "did this survive
    # filtering" -- so match each raw result against top5/relevant by
    # object identity (they're the same dicts, just filtered/reordered),
    # rather than re-comparing text.
    relevant_ids = {id(r) for r in relevant}
    top5_ids = {id(r) for r in top5}

    def matches_target(r) -> bool:
        meta = r.get("metadata", {}) or {}
        doc = (r.get("document") or "")
        if args.contains and args.contains.lower() in doc.lower():
            return True
        if args.source and args.source.lower() in str(meta.get("source", "")).lower():
            if args.section:
                return str(meta.get("section_number", "")) == str(args.section)
            return True
        if args.section and not args.source:
            return str(meta.get("section_number", "")) == str(args.section)
        return False

    print(f"{'Rank':<5}{'Distance':<12}{'Source':<30}{'Section':<12}{'Target?':<9}{'Verdict'}")
    print("-" * 100)

    best_distance = min((r.get("distance", 999) for r in raw_results), default=None)
    any_target_found = False

    for rank, r in enumerate(raw_results, start=1):
        meta = r.get("metadata", {}) or {}
        dist = r.get("distance")
        source = str(meta.get("source", "?"))[:28]
        section = str(meta.get("section_number", "") or "-")
        is_target = matches_target(r)
        if is_target:
            any_target_found = True

        if id(r) in top5_ids:
            verdict = "IN CONTEXT (sent to model)"
        elif id(r) in relevant_ids:
            verdict = "CAPPED OUT (relevant, but ranked 6th-8th)"
        else:
            margin = engine.RELEVANCE_MARGIN
            ceiling = engine.MAX_RELEVANT_DISTANCE
            if dist is not None and best_distance is not None and dist > best_distance + margin:
                verdict = f"FILTERED OUT (>{margin:.2f} beyond best match {best_distance:.3f})"
            elif dist is not None and dist > ceiling:
                verdict = f"FILTERED OUT (beyond absolute ceiling {ceiling:.2f})"
            else:
                verdict = "FILTERED OUT (reason unclear -- inspect manually)"

        flag = "<-- TARGET" if is_target else ""
        print(f"{rank:<5}{dist:<12.4f}{source:<30}{section:<12}{flag:<12}{verdict}")

    print()
    if not any_target_found:
        print("Target chunk did NOT appear in the raw top-8 at all for this query.")
        print("This means an uncapped, unfiltered search doesn't consider it close enough to")
        print("return -- that points at the embedding model or query phrasing, not at")
        print("MAX_RELEVANT_DISTANCE / RELEVANCE_MARGIN / the top-5 cap, since none of those")
        print("ever got a chance to act on it.")
    else:
        print("See the target row's verdict above for exactly where it fell out of the pipeline")
        print("(or confirmation that it made it into context, in which case the model had it")
        print("and the citation-tracking logic, not retrieval, is where to look next).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
