"""One-off script: clear the existing statute PDF index and rebuild it
using the new subsection-aware chunking logic in pdf_ingestor.py.

Run this from your backend directory (so relative paths like
"data/pdfs" resolve correctly):

    python reingest_statutes.py

Do NOT paste this code directly into a PowerShell prompt — PowerShell
doesn't understand Python syntax; you'll get a parser error like the one
you just saw. Run it as `python reingest_statutes.py` instead.
"""

import time

from logic.rag_engine import RAGEngine
from logic.embeddings import embeddings_service

# ADJUST THIS IMPORT if pdf_ingestor.py lives somewhere else in your
# project (e.g. `from logic.pdf_ingestor import PDFIngestor` if it's
# inside the logic/ package rather than at the top level of backend/).
from logic.pdf_ingestor import PDFIngestor


def main():
    print("Loading embeddings model...")
    embeddings_service.load()

    print("Initializing RAG engine (connecting to ChromaDB)...")
    rag = RAGEngine()

    waited = 0
    while not rag.is_ready() and waited < 30:
        time.sleep(0.5)
        waited += 0.5

    if not rag.is_ready():
        print("RAG engine did not become ready in time — aborting.")
        return

    before_count = rag.collection.count()
    print(f"Current document count before clearing: {before_count}")

    ingestor = PDFIngestor(rag_engine=rag)

    print("\nClearing existing statute PDF index...")
    removed = ingestor.clear_pdf_index()

    if removed is None:
        print("\nclear_pdf_index() failed (see error above) — ABORTING before")
        print("re-ingesting. Re-ingesting on top of a failed clear would leave")
        print("old, badly-chunked entries mixed in with new ones. Fix whatever")
        print("clear_pdf_index() reported and re-run this script from scratch.")
        return

    # Explicit verification, not just trusting the return value: actually
    # check the collection for any remaining kenyan_law_pdf-tagged entries
    # before proceeding. This is the same "don't just assume it worked"
    # principle behind every grounding check we've added elsewhere in this
    # project — a partial/silent failure here is worse than doing nothing,
    # since it would leave the old wrong-subsection chunks retrievable
    # alongside the new correct ones with no visible sign anything's wrong.
    remaining = rag.collection.get()
    remaining_old = sum(
        1 for m in remaining.get('metadatas', [])
        if m and m.get('category') == 'kenyan_law_pdf'
    )
    if remaining_old > 0:
        print(f"\n{remaining_old} old statute document(s) still remain after "
              f"clear_pdf_index() reported removing {removed} — ABORTING before "
              f"re-ingesting. Investigate before re-running.")
        return

    print(f"Confirmed: 0 old statute documents remain (removed {removed}).")

    print("\nRe-ingesting statute PDFs with the new subsection-aware chunker...")
    # ADJUST THIS PATH if your PDFs live somewhere other than
    # backend/data/pdfs — check main.py's PDF_DIR constant if unsure.
    ingestor.ingest_folder("data/pdfs")

    after_count = rag.collection.count()
    print(f"\nFinal document count: {after_count}")
    print(f"(was {before_count} before clearing/re-ingesting)")


if __name__ == "__main__":
    main()
