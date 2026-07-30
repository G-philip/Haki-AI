"""
check_section_drift.py — quantifies how far the section_number tagging bug
extends, following up on check_metadata_tags.py's finding that 161 chunks
are all tagged section_number="205" despite covering clearly unrelated
content (manslaughter, animals, vessels, aircraft, theft definitions, etc).

Hypothesis: the ingestion script's section-heading parser stops detecting
NEW section headings at some point and keeps stamping every subsequent
chunk with the last section number it successfully parsed (205), while a
separate subsection counter keeps incrementing off "(1)", "(2)", "(3)"...
patterns in the text regardless of which real section those belong to.

This script does NOT modify the collection. It only reads and reports.

USAGE:
    python check_section_drift.py D:\\projects\\kenyan-legal-bot\\backend\\data\\chroma_db
"""

import sys
import re
from collections import Counter
from pathlib import Path

import chromadb

STUCK_SECTION = "205"

# Matches a real section/article heading style like "296. Robbery with
# violence" or "25. Sentence of death" — a number, a period, then a
# capitalized title phrase. This is how Penal Code headings are actually
# written (confirmed in check_metadata_tags.py's Hit 3: "25. Sentence of
# death").
HEADING_PATTERN = re.compile(r'\b(\d{1,3}[A-Za-z]?)\.\s+([A-Z][a-zA-Z ,\'\-]{3,60})')


def get_collection_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return Path(__file__).parent.parent / "data" / "chroma_db"


def load_collection(path: Path):
    if not path.exists():
        print(f"ERROR: chroma_db path does not exist: {path}")
        sys.exit(1)
    client = chromadb.PersistentClient(path=str(path))
    try:
        return client.get_collection("kenyan_law")
    except Exception as e:
        print(f"ERROR: could not open collection 'kenyan_law' at {path}: {e}")
        sys.exit(1)


def main():
    path = get_collection_path()
    print(f"Opening collection at: {path}\n")
    collection = load_collection(path)

    all_docs = collection.get(include=["documents", "metadatas"])
    documents = all_docs.get("documents", [])
    metadatas = all_docs.get("metadatas", [])

    stuck_chunks = [
        (doc, meta) for doc, meta in zip(documents, metadatas)
        if str((meta or {}).get("section_number", "")) == STUCK_SECTION
    ]

    print("=" * 78)
    print(f"Chunks tagged section_number='{STUCK_SECTION}': {len(stuck_chunks)}")
    print("=" * 78)

    heading_counter = Counter()
    no_heading_found = 0
    examples_per_heading = {}

    for doc, meta in stuck_chunks:
        matches = HEADING_PATTERN.findall(doc or "")
        if not matches:
            no_heading_found += 1
            continue
        # Record every distinct heading-looking number found inside this
        # chunk's own text, so we can see the full spread of REAL section
        # numbers hiding under the single stuck "205" tag.
        for number, title in matches:
            key = f"{number}. {title.strip()}"
            heading_counter[number] += 1
            examples_per_heading.setdefault(number, key)

    print(f"\nChunks with no detectable heading pattern in their own text: {no_heading_found}")
    print(f"Distinct real section numbers found hiding under the '{STUCK_SECTION}' tag: "
          f"{len(heading_counter)}\n")

    print(f"{'Detected #':<12}{'Count':<8}Example heading text found in chunk")
    print("-" * 78)
    for number, count in heading_counter.most_common():
        print(f"{number:<12}{count:<8}{examples_per_heading[number]}")

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    if len(heading_counter) > 3:
        print(
            f"{len(heading_counter)} different real section numbers were found embedded\n"
            f"inside chunks that are ALL tagged section_number='{STUCK_SECTION}'. This\n"
            f"confirms the metadata tag is not being re-parsed per chunk — it's stuck\n"
            f"at '{STUCK_SECTION}' and carried forward across many genuinely different\n"
            f"sections. The fix belongs in the ingestion script's section-heading\n"
            f"detection logic (likely: it fails to match a new heading at some point\n"
            f"and falls back to / never updates from the last known value instead of\n"
            f"flagging the parse failure).\n"
        )
    else:
        print(
            "Few or no distinct headings found — the drift may be narrower than\n"
            "suspected, or headings in this section of the source PDF don't follow\n"
            "the 'NUMBER. Title' pattern this script looks for. Consider widening\n"
            "HEADING_PATTERN or manually inspecting a sample of these chunks.\n"
        )


if __name__ == "__main__":
    main()
