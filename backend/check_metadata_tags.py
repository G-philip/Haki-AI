"""
check_metadata_tags.py — one-off diagnostic for the chroma_db collection
used by rag_engine.py.

Purpose: confirm (or rule out) the ingestion metadata bug suspected from
production logs, where a chunk containing the literal robbery-with-violence
death-penalty text ("...he wounds, beats, strikes or uses any other personal
violence to any person, ... sentenced to death") appears to be tagged with
metadata section_number="205" instead of the correct "296".

This script does NOT modify the collection. It only reads and reports.

USAGE:
    Drop this file in the same directory as rag_engine.py (so the relative
    "../data/chroma_db" path below matches), then run:

        python check_metadata_tags.py

    Or pass an explicit path:

        python check_metadata_tags.py /path/to/chroma_db
"""

import sys
import re
from pathlib import Path

import chromadb

PHRASE_TO_CHECK = "sentenced to death"
SECTIONS_TO_INSPECT = ["296", "205"]  # the two numbers involved in the mismatch


def get_collection_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    # Mirrors RAGEngine._load_database()'s default relative path.
    return Path(__file__).parent.parent / "data" / "chroma_db"


def load_collection(path: Path):
    if not path.exists():
        print(f"ERROR: chroma_db path does not exist: {path}")
        print("Pass the correct path as an argument, e.g.:")
        print("    python check_metadata_tags.py /path/to/your/chroma_db")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(path))
    try:
        return client.get_collection("kenyan_law")
    except Exception as e:
        print(f"ERROR: could not open collection 'kenyan_law' at {path}: {e}")
        sys.exit(1)


def snippet(doc: str, around: str, width: int = 120) -> str:
    doc_norm = re.sub(r"\s+", " ", doc)
    idx = doc_norm.lower().find(around.lower())
    if idx == -1:
        return doc_norm[:width] + "..."
    start = max(0, idx - width // 2)
    end = min(len(doc_norm), idx + len(around) + width // 2)
    return ("..." if start > 0 else "") + doc_norm[start:end] + ("..." if end < len(doc_norm) else "")


def main():
    path = get_collection_path()
    print(f"Opening collection at: {path}\n")
    collection = load_collection(path)

    try:
        count = collection.count()
    except Exception as e:
        print(f"ERROR: could not count collection: {e}")
        sys.exit(1)

    print(f"Collection 'kenyan_law' loaded. Total chunks: {count}\n")

    all_docs = collection.get(include=["documents", "metadatas"])
    documents = all_docs.get("documents", [])
    metadatas = all_docs.get("metadatas", [])

    # --- Part 1: find every chunk containing the exact penalty phrase ---
    print("=" * 78)
    print(f'PART 1: chunks containing the phrase "{PHRASE_TO_CHECK}"')
    print("=" * 78)

    phrase_hits = []
    for doc, meta in zip(documents, metadatas):
        if PHRASE_TO_CHECK.lower() in (doc or "").lower():
            phrase_hits.append((doc, meta or {}))

    if not phrase_hits:
        print(f'No chunks found containing "{PHRASE_TO_CHECK}". '
              f"Either the phrasing differs slightly in your source text, "
              f"or this data hasn't been ingested under this collection/path.")
    else:
        for i, (doc, meta) in enumerate(phrase_hits, start=1):
            section = meta.get("section_number", "<missing>")
            subsection = meta.get("subsection_number", "<missing>")
            category = meta.get("category", "<missing>")
            source = meta.get("source", "<missing>")
            print(f"\n--- Hit {i} ---")
            print(f"  metadata.section_number    = {section}")
            print(f"  metadata.subsection_number = {subsection}")
            print(f"  metadata.category          = {category}")
            print(f"  metadata.source            = {source}")
            print(f"  text snippet: {snippet(doc, PHRASE_TO_CHECK)}")

            # Try to spot the actual heading number inside the text itself,
            # using the same "NUM. (1)" pattern rag_engine.py's
            # _exact_number_lookup uses, so we can compare it to what the
            # metadata claims.
            heading_match = re.search(r"\b(\d{1,3}[A-Za-z]?)\.\s*\(\d+\)", doc)
            if heading_match:
                print(f"  -> heading number found IN TEXT: {heading_match.group(1)} "
                      f"(compare to metadata.section_number above)")
            else:
                print("  -> no clear 'NUM. (1)' heading pattern found in this chunk's own text "
                      "(the heading may be in a preceding/adjacent chunk).")

    # --- Part 2: for each suspect section number, show what's actually tagged ---
    print("\n" + "=" * 78)
    print("PART 2: what IS tagged under each suspect section_number")
    print("=" * 78)

    for sec in SECTIONS_TO_INSPECT:
        matches = [
            (doc, meta) for doc, meta in zip(documents, metadatas)
            if str((meta or {}).get("section_number", "")) == sec
        ]
        print(f"\n--- metadata.section_number == '{sec}'  ({len(matches)} chunk(s)) ---")
        if not matches:
            print("  (no chunks tagged with this section_number)")
            continue
        for doc, meta in matches:
            subsection = (meta or {}).get("subsection_number", "<missing>")
            print(f"  subsection_number={subsection}  text: {snippet(doc, doc[:40] if doc else '')}")

    print("\n" + "=" * 78)
    print("WHAT TO LOOK FOR")
    print("=" * 78)
    print(
        "If a hit in PART 1 shows metadata.section_number='205' but the text\n"
        "snippet or the in-text heading number clearly reads as Section 296\n"
        "(the robbery-with-violence provision), that confirms the ingestion\n"
        "script mis-tagged this chunk's section_number/subsection_number\n"
        "fields — the fix belongs in your ingestion/heading-parsing logic,\n"
        "not in rag_engine.py's validators, which are correctly reporting\n"
        "what the metadata (wrongly) says.\n"
    )


if __name__ == "__main__":
    main()
