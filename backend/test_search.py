"""Test RAG — simple, matches the working rag_engine.py"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from logic.rag_engine import RAGEngine
from logic.embeddings import embeddings_service

# Load
print("Loading...")
embeddings_service.load()
rag = RAGEngine()

while not rag.is_ready():
    time.sleep(0.5)

print(f"Database: {rag.collection.count()} documents\n")

# Test
questions = [
    # "can my wife get inherited property i own? I inherited property from my deseased father",
    "what to do when illegally arrested by police",
    # "rights of arrested persons",
]

for q in questions:
    print("=" * 60)
    print(f"Q: {q}")
    print("=" * 60)
    print(rag.generate_answer(q))
    print()