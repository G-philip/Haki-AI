#!/usr/bin/env python3
"""
check_source_text.py

Standalone ground-truth check: does a given phrase actually exist in your
ingested legal corpus, verbatim or near-verbatim -- independent of anything
the LLM said, and independent of rag_engine.py's own validator logic (so
this can't share a bug with the thing it's meant to double-check).

Answers exactly the question raised in the last conversation: is
"held in trust for the other spouse" really in the corpus (and just split
across a chunk boundary or slightly reworded), or was it never there and
the validator has been correctly catching a hallucination that recurred
often enough across sessions to look confirmed?

USAGE

    # Check against the ingested Chroma DB (the chunks the model actually sees)
    python3 check_source_text.py "held in trust for the other spouse"

    # Also cross-check against the raw source PDFs directly, independent of
    # chunking -- pass the folder your PDFs live in (same folder you'd pass
    # to pdf_ingestor.py's ingest_folder / ingest_case_law_folder)
    python3 check_source_text.py "held in trust for the other spouse" --pdf-dir ./data/statutes

    # If your Chroma DB lives somewhere other than the default relative path
    python3 check_source_text.py "some phrase" --db-path /path/to/chroma_db

WHAT IT DOES
    1. Connects to the same persistent Chroma collection rag_engine.py reads
       from ("kenyan_law"), pulls every chunk's text + metadata, and reports:
         - EXACT matches (case-insensitive, whitespace-normalized)
         - NEAR matches above a similarity floor (catches chunk-boundary
           splitting, minor rewording, OCR noise) -- reported with a score
           so you can judge each one, rather than a single opaque yes/no
    2. If --pdf-dir is given, also re-extracts text directly from the raw
       PDFs (same PyPDF2 approach pdf_ingestor.py uses) and runs the same
       exact/near check against THAT text. Comparing the two tells you
       whether a discrepancy is a chunking artifact (present in the raw
       PDF, but split/lost across chunk boundaries) or the phrase was never
       in the source at all.

This is deliberately a separate script, not a rag_engine.py method: the
whole point is to check the checker using an independent code path.
"""

import argparse
import difflib
import re
import sys
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def fuzzy_windowed_ratio(needle: str, haystack: str) -> float:
    """Best similarity ratio of `needle` against any equal-ish-length
    window of `haystack`. Same sliding-window idea as rag_engine.py's
    _fuzzy_quote_match, reimplemented independently here on purpose."""
    if not needle or not haystack:
        return 0.0
    if needle in haystack:
        return 1.0
    n_len = len(needle)
    window = n_len + max(20, n_len // 4)
    step = max(1, n_len // 6)
    best = 0.0
    for start in range(0, max(1, len(haystack) - min(n_len, len(haystack)) + 1), step):
        segment = haystack[start:start + window]
        ratio = difflib.SequenceMatcher(None, needle, segment, autojunk=False).ratio()
        if ratio > best:
            best = ratio
    return best


def check_chroma(phrase: str, db_path: Path, collection_name: str, near_floor: float):
    try:
        import chromadb
    except ImportError:
        print("chromadb not installed in this environment -- skipping DB check "
              "(run: pip install chromadb --break-system-packages)")
        return

    if not db_path.exists():
        print(f"DB path does not exist: {db_path}")
        print("Pass the correct location with --db-path, e.g.:")
        print("  python3 check_source_text.py \"phrase\" --db-path /path/to/data/chroma_db")
        return

    client = chromadb.PersistentClient(path=str(db_path))
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"Could not open collection '{collection_name}' at {db_path}: {e}")
        return

    total = collection.count()
    print(f"\n=== CHROMA DB CHECK ===")
    print(f"DB path: {db_path}")
    print(f"Collection: {collection_name}  (total chunks: {total})")

    data = collection.get(include=["documents", "metadatas"])
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    needle = normalize(phrase)
    exact_hits = []
    near_hits = []

    for doc, meta in zip(docs, metas):
        haystack = normalize(doc or "")
        if needle in haystack:
            exact_hits.append((doc, meta))
            continue
        ratio = fuzzy_windowed_ratio(needle, haystack)
        if ratio >= near_floor:
            near_hits.append((ratio, doc, meta))

    if exact_hits:
        print(f"\nEXACT match found in {len(exact_hits)} chunk(s):")
        for doc, meta in exact_hits:
            _print_chunk(doc, meta)
    else:
        print("\nNo EXACT (verbatim) match found in any ingested chunk.")

    near_hits.sort(key=lambda h: -h[0])
    if near_hits:
        print(f"\nNEAR matches (similarity >= {near_floor:.2f}), best first:")
        for ratio, doc, meta in near_hits[:8]:
            print(f"  similarity={ratio:.3f}")
            _print_chunk(doc, meta, indent="    ")
    elif not exact_hits:
        print(f"No near matches either (similarity < {near_floor:.2f} against every chunk).")
        print("This is the strongest signal the phrase simply isn't in the ingested corpus at all.")


def _print_chunk(doc: str, meta: dict, indent: str = "  "):
    source = meta.get("source", "?")
    section = meta.get("section_number", "")
    subsection = meta.get("subsection_number", "")
    category = meta.get("category", "?")
    loc = f"Section {section}" + (f"({subsection})" if subsection else "") if section else "(no section metadata)"
    print(f"{indent}source='{source}'  {loc}  category={category}")
    snippet = doc.strip().replace("\n", " ")
    if len(snippet) > 220:
        snippet = snippet[:220] + "..."
    print(f"{indent}text: {snippet}")


def check_raw_pdfs(phrase: str, pdf_dir: Path, near_floor: float):
    try:
        import PyPDF2
    except ImportError:
        print("\nPyPDF2 not installed -- skipping raw-PDF check "
              "(run: pip install PyPDF2 --break-system-packages)")
        return

    if not pdf_dir.exists():
        print(f"\n--pdf-dir does not exist: {pdf_dir}")
        return

    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"\n=== RAW PDF CHECK (independent of chunking) ===")
    print(f"Folder: {pdf_dir}  ({len(pdf_files)} PDF files)")

    needle = normalize(phrase)
    any_hit = False

    for pdf_path in pdf_files:
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ""
                for page in reader.pages:
                    full_text += (page.extract_text() or "") + " "
        except Exception as e:
            print(f"  [skip] {pdf_path.name}: could not read ({e})")
            continue

        haystack = normalize(full_text)
        if needle in haystack:
            print(f"  EXACT match in raw PDF text: {pdf_path.name}")
            any_hit = True
            continue

        ratio = fuzzy_windowed_ratio(needle, haystack)
        if ratio >= near_floor:
            print(f"  NEAR match (similarity={ratio:.3f}) in raw PDF text: {pdf_path.name}")
            any_hit = True

    if not any_hit:
        print("  No exact or near match found in any raw PDF in this folder.")
        print("  If this folder is the actual ingestion source, this is strong evidence")
        print("  the phrase was never in the source text at all.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phrase", help="The exact phrase to search for, e.g. \"held in trust for the other spouse\"")
    parser.add_argument("--db-path", default=None,
                         help="Path to the Chroma DB folder. Default: <this script's folder>/../data/chroma_db "
                              "(same relative path rag_engine.py uses).")
    parser.add_argument("--collection", default="kenyan_law", help="Chroma collection name (default: kenyan_law)")
    parser.add_argument("--pdf-dir", default=None,
                         help="Optional: folder of raw source PDFs to cross-check independent of chunking.")
    parser.add_argument("--near-floor", type=float, default=0.6,
                         help="Minimum similarity ratio (0-1) to report as a 'near match'. Lower = more results, "
                              "including weak ones. Default: 0.6")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else (Path(__file__).parent / "data" / "chroma_db")

    print(f"Searching for phrase: \"{args.phrase}\"")

    check_chroma(args.phrase, db_path, args.collection, args.near_floor)

    if args.pdf_dir:
        check_raw_pdfs(args.phrase, Path(args.pdf_dir), args.near_floor)
    else:
        print("\n(No --pdf-dir given, so this only checked the ingested/chunked text, "
              "not the original PDFs. Pass --pdf-dir to also rule out a chunking artifact.)")


if __name__ == "__main__":
    sys.exit(main())
