"""
Two checks in one:

1. Confirms whether logging is actually configured to print WARNING/INFO
   level messages to console. If test_search.py never showed
   NUMBER_MISMATCH, RETRIEVAL, EXACT_NUMBER_MATCH, etc. lines despite the
   code definitely calling logger.warning(...)/logger.info(...), this is
   almost certainly why — no handler was attached, so the messages were
   silently dropped rather than genuinely absent.

2. Checks whether OLLAMA_KEEP_ALIVE is set in the current environment,
   and queries Ollama's running model list to see if the model is
   currently loaded right now.

Usage:
    python check_setup.py
"""

import os
import logging
import requests

print("=" * 70)
print("1. LOGGING CONFIGURATION CHECK")
print("=" * 70)

root_logger = logging.getLogger()
print(f"Root logger level: {logging.getLevelName(root_logger.level)}")
print(f"Root logger handlers: {root_logger.handlers}")

rag_logger = logging.getLogger("haki_ai.rag_engine")
print(f"\n'haki_ai.rag_engine' logger level: {logging.getLevelName(rag_logger.level)}")
print(f"'haki_ai.rag_engine' logger handlers: {rag_logger.handlers}")
print(f"'haki_ai.rag_engine' effective level: {logging.getLevelName(rag_logger.getEffectiveLevel())}")
print(f"Propagate to root: {rag_logger.propagate}")

if not root_logger.handlers and not rag_logger.handlers:
    print("\n⚠ NO HANDLERS ATTACHED ANYWHERE.")
    print("  This means logger.warning(...) and logger.info(...) calls in")
    print("  rag_engine.py are being silently swallowed — not printed")
    print("  anywhere, not even to console. This is almost certainly why")
    print("  you haven't seen NUMBER_MISMATCH, RETRIEVAL, or other log")
    print("  lines even though the code path that logs them has run.")
    print()
    print("  Fix: add this near the top of main.py (or test_search.py),")
    print("  before RAGEngine() is instantiated:")
    print()
    print("      import logging")
    print("      logging.basicConfig(")
    print("          level=logging.INFO,")
    print("          format='%(asctime)s %(levelname)s %(name)s: %(message)s'")
    print("      )")
else:
    print("\n✓ At least one handler is attached — warnings should be visible")
    print("  somewhere (console or wherever the handler points).")

print()
print("=" * 70)
print("2. OLLAMA KEEP-ALIVE + LIVE MODEL STATUS CHECK")
print("=" * 70)

keep_alive_env = os.environ.get("OLLAMA_KEEP_ALIVE")
if keep_alive_env is None:
    print("⚠ OLLAMA_KEEP_ALIVE is NOT set in this process's environment.")
    print("  Note: this only checks the environment of THIS python process.")
    print("  What matters is whether it was set in the environment that")
    print("  ran 'ollama serve'. If you set it in a different terminal")
    print("  before starting the server, this check won't see it — but")
    print("  the server's own behavior (below) is the real test.")
else:
    print(f"OLLAMA_KEEP_ALIVE in this process's env: {keep_alive_env}")

try:
    r = requests.get("http://localhost:11434/api/ps", timeout=5)
    if r.ok:
        models = r.json().get("models", [])
        if models:
            print(f"\n✓ Ollama reports {len(models)} model(s) currently loaded:")
            for m in models:
                name = m.get("name", "unknown")
                expires = m.get("expires_at", "unknown")
                size_vram = m.get("size_vram", 0)
                print(f"   - {name}  (expires_at={expires}, size_vram={size_vram})")
            print()
            print("  If 'expires_at' is far in the future or this never")
            print("  empties between your test runs, keep-alive is working.")
            print("  If models show here but test_search.py STILL times out,")
            print("  the model is loaded and the bottleneck is pure")
            print("  generation throughput, not reload cost.")
        else:
            print("\n⚠ No models currently loaded in Ollama.")
            print("  This means the model has been unloaded (idle timeout)")
            print("  since your last call. The next request will pay a full")
            print("  cold-load cost again. If you've set OLLAMA_KEEP_ALIVE=-1")
            print("  and still see this, the setting likely isn't actually")
            print("  active in the server process — double check by setting")
            print("  it in the SAME terminal, in the SAME command, e.g.:")
            print()
            print("      set OLLAMA_KEEP_ALIVE=-1 && ollama serve")
            print()
            print("  (Windows cmd) or use 'setx' for a persistent system-wide")
            print("  env var, then restart the terminal/service so it's")
            print("  actually picked up.")
    else:
        print(f"\n✗ /api/ps returned status {r.status_code}")
except Exception as e:
    print(f"\n✗ Could not reach Ollama at all: {e}")
    print("  Server may not be running right now.")
