"""
Ollama health check — run this BEFORE test_search.py to find out whether
slowness/timeouts are an Ollama problem or a RAG problem, independent of
ChromaDB, embeddings, or anything else.

Usage:
    python check_ollama.py
"""

import time
import requests

OLLAMA_BASE = "http://localhost:11434"
MODEL = "llama3.2:latest"


def check_server_alive():
    print("1. Checking if Ollama server is reachable...")
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.ok:
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"   Server is up. Installed models: {models}")
            if MODEL not in models and MODEL.split(":")[0] not in [m.split(":")[0] for m in models]:
                print(f"   WARNING: '{MODEL}' not found in installed models list above.")
            return True
        else:
            print(f"   Server responded but with status {r.status_code}: {r.text[:200]}")
            return False
    except requests.exceptions.ConnectionError as e:
        print(f"   FAILED: cannot connect to Ollama at all. Is it running? err={e}")
        return False
    except Exception as e:
        print(f"   FAILED: unexpected error: {e}")
        return False


def check_generation_speed():
    print(f"\n2. Sending a minimal test prompt to '{MODEL}' (this may trigger a model load)...")
    start = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": MODEL,
                "prompt": "Reply with exactly the word: OK",
                "stream": False,
                "options": {"num_predict": 5, "temperature": 0}
            },
            timeout=180  # generous, since this run may include a cold model load
        )
        elapsed = time.time() - start
        if r.ok:
            text = r.json().get("response", "").strip()
            print(f"   SUCCESS in {elapsed:.1f}s. Response: {text!r}")
            if elapsed > 30:
                print("   NOTE: this took a while — likely a cold model load.")
                print("   Run this script again immediately; if the second run")
                print("   is fast (<5s), the issue is Ollama's idle/unload behavior,")
                print("   not a hardware/capacity problem.")
        else:
            print(f"   Server responded with status {r.status_code} after {elapsed:.1f}s: {r.text[:300]}")
    except requests.exceptions.ReadTimeout:
        elapsed = time.time() - start
        print(f"   TIMED OUT after {elapsed:.1f}s. Ollama is up but not generating in time.")
        print("   Likely causes: model too large for available RAM/VRAM,")
        print("   CPU-only inference being slow, or resource contention with")
        print("   another process (embeddings service, Chroma, etc).")
    except Exception as e:
        print(f"   FAILED: unexpected error: {e}")


def check_second_call_speed():
    print(f"\n3. Sending a SECOND test prompt immediately (model should now be warm)...")
    start = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": MODEL,
                "prompt": "Reply with exactly the word: OK",
                "stream": False,
                "options": {"num_predict": 5, "temperature": 0}
            },
            timeout=30
        )
        elapsed = time.time() - start
        if r.ok:
            print(f"   SUCCESS in {elapsed:.1f}s (warm). If this is still slow,")
            print("   the bottleneck is generation speed itself (hardware), not")
            print("   model loading.")
        else:
            print(f"   Status {r.status_code} after {elapsed:.1f}s")
    except requests.exceptions.ReadTimeout:
        print(f"   STILL TIMED OUT even on a warm call (>30s). This points to")
        print("   a genuine throughput/hardware problem, not just cold-load time.")
    except Exception as e:
        print(f"   FAILED: {e}")


if __name__ == "__main__":
    if check_server_alive():
        check_generation_speed()
        check_second_call_speed()
    else:
        print("\nFix the connection issue above before testing RAG — none of")
        print("RAGEngine's behavior matters until Ollama itself responds.")
