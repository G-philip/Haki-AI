"""Check what the James Kariuki Wagana case actually says"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from logic.rag_engine import RAGEngine
from logic.embeddings import embeddings_service

embeddings_service.load()
rag = RAGEngine()
while not rag.is_ready():
    time.sleep(0.5)

results = rag.search_database("James Kariuki Wagana", top_k=5)

for i, r in enumerate(results):
    print(f"\n{'='*60}")
    print(f"Result {i+1}")
    print(f"Source: {r['metadata'].get('source', '?')}")
    print(f"{'='*60}")
    print(r['document'][:2000])
    print()