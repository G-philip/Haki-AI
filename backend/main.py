"""Main FastAPI Application for Kenyan Legal Assistant"""

import os
import sys
import time
import threading
import logging
import logging.handlers
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import traceback
import requests

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================
# LOGGING SETUP
# ============================================
# Configured here, before any of our own modules are imported below, so
# that RAGEngine (and anything else that grabs a logger at import/init
# time) is already writing to a real file by the time it starts logging.
# Without this, Python's logging defaults to WARNING-and-above-to-stderr
# only, and INFO-level lines (RETRIEVAL, EXACT_NUMBER_MATCH, etc.) were
# silently dropped rather than persisted anywhere.
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Windows' console defaults to the cp1252 codepage, which can't encode
# characters like '✓'/'❌' used elsewhere in this app's log messages
# (see logic/embeddings.py's "✓ Model loaded successfully" line). Without
# forcing UTF-8 here, logging.StreamHandler.emit() throws internally on
# every such line and prints a "--- Logging error ---" traceback instead
# of the actual message — noisy and easy to mistake for a real failure.
# Wrapping stdout in a UTF-8 TextIOWrapper (errors="replace" so a truly
# unencodable byte degrades to a replacement char instead of crashing)
# fixes the console handler; encoding="utf-8" on the file handler covers
# the same risk for the log file itself. This is a no-op on platforms
# that already default to UTF-8 (Linux/Mac), so it's safe everywhere.
import io
utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "app.log", maxBytes=10_000_000, backupCount=3, encoding="utf-8"
        ),
        logging.StreamHandler(utf8_stdout),
    ],
)

logger = logging.getLogger("haki_ai.main")

from logic.chat_manager import ChatManager
from logic.interview_engine import InterviewEngine
from logic.data_manager import DataManager
from logic.message_handler import MessageHandler
from logic.rag_engine import RAGEngine, UNAVAILABLE_MESSAGE as UNAVAILABLE_MESSAGE_TEXT
from logic.search_tool import LegalSearchTool
from logic.embeddings import embeddings_service
# ADJUST if pdf_ingestor.py lives elsewhere in your project (e.g.
# `from logic.pdf_ingestor import PDFIngestor`).
from logic.pdf_ingestor import PDFIngestor

# Create data directories
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PDF_DIR = DATA_DIR / "pdfs"
PDF_DIR.mkdir(exist_ok=True)
CHROMA_DIR = DATA_DIR / "chroma_db"
CHROMA_DIR.mkdir(exist_ok=True)

# Global shared instances
rag_engine = None
search_tool = None
data_manager = None
active_sessions: Dict[str, ChatManager] = {}


# ============================================
# HELPER FUNCTIONS
# ============================================

def _check_ollama():
    """Check if Ollama is available with retry.

    FIX: both bare `except:` clauses used to swallow every exception
    silently (including things unrelated to Ollama being down, like a
    typo introduced later in this function) and just return False with
    no trace of why. Now every failure path logs the real exception, so
    "health check says unhealthy" is debuggable from logs instead of a
    dead end.
    """
    for attempt in range(3):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                return True
            logger.warning(
                f"OLLAMA_CHECK_BAD_STATUS: attempt={attempt} status={response.status_code}"
            )
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"OLLAMA_CHECK_CONNECTION_ERROR: attempt={attempt} err={e}")
        except Exception as e:
            logger.error(f"OLLAMA_CHECK_UNEXPECTED_ERROR: attempt={attempt} err={e}")

        if attempt < 2:
            time.sleep(1)

    return False


def _check_ollama_fast():
    """Single-attempt Ollama check, no retries, no sleep — for endpoints
    that get polled repeatedly (/api/health, /api/stats), as opposed to
    _check_ollama()'s 3-attempt retry loop, which is the right amount of
    care for a ONE-TIME startup verification but pure waste when it's
    being called every 15 seconds indefinitely: a poll that fails just
    gets tried again by the NEXT poll 15s later anyway, so retrying 3x
    *inside* each individual poll adds latency and log noise (3 warnings
    per call instead of 1) without adding any real reliability the
    polling loop wasn't already providing at a higher level. Silent on
    failure — the caller already logs/surfaces unavailability at the
    response level (status: "degraded" etc.), so a second log line here
    for the exact same, expected-when-Ollama-is-down condition would
    just be the same noise this function exists to remove."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def _background_readiness_checks(rag_engine_instance: "RAGEngine"):
    """Polls RAG DB readiness and verifies Ollama, in a background thread
    started from lifespan() rather than awaited inline before `yield`.

    This is the direct fix for "frontend loads, then everything just
    sits there for several seconds before session creation / history /
    the first question work at all": FastAPI's lifespan `yield` gates
    EVERY route, including /api/new-session, /api/history, and
    /api/health, none of which touch rag_engine or Ollama at all. The
    previous code blocked right here — up to 30s waiting for the Chroma
    DB, plus Ollama's retry/backoff on top — before the app would accept
    ANY HTTP request whatsoever, session creation and history included.

    /api/chat still genuinely needs both to be ready to produce a real
    answer, but rag_engine.search_database() already fails soft
    (logs SEARCH_SKIPPED, returns []) rather than raising if it's called
    before this finishes — so a chat request landing in this brief
    window gets the existing "no answer" fallback, the same failure mode
    that already existed, just now confined to /api/chat specifically
    instead of blocking the whole app shell."""
    print("   Connecting to ChromaDB (in background)...")
    max_wait = 30
    waited = 0
    while not rag_engine_instance.is_ready() and waited < max_wait:
        time.sleep(0.5)
        waited += 0.5

    if rag_engine_instance.is_ready():
        print(f"✓ RAG Engine ready after {waited:.1f}s")
        if rag_engine_instance.collection:
            doc_count = rag_engine_instance.collection.count()
            print(f"  Database contains {doc_count} documents")
            logger.info(f"RAG_ENGINE_READY: waited={waited:.1f}s doc_count={doc_count}")
    else:
        print(f"⚠ RAG Engine not ready after {max_wait}s (will retry on queries)")
        logger.error(f"RAG_ENGINE_NOT_READY: gave up after {max_wait}s")

    print("\nVerifying Ollama connection (in background)...")
    if _check_ollama():
        print("✓ Ollama is running")
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.ok:
                models = response.json().get('models', [])
                if models:
                    model_names = [m.get('name', 'unknown') for m in models]
                    print(f"  Available models: {', '.join(model_names)}")
                if rag_engine_instance and hasattr(rag_engine_instance, 'ollama_model'):
                    print(f"  Using model: {rag_engine_instance.ollama_model}")
                    logger.info(
                        f"OLLAMA_READY: available_models={model_names if models else []} "
                        f"configured_model={rag_engine_instance.ollama_model}"
                    )
        except Exception as e:
            # Same rationale as the original inline version: purely
            # informational, shouldn't be fatal, but should be visible.
            logger.warning(f"OLLAMA_MODEL_LIST_CHECK_FAILED: err={e}")
    else:
        print("❌ Ollama is NOT running!")
        print("")
        print("  TO FIX THIS:")
        print("  1. Open a NEW terminal window")
        print("  2. Run: ollama serve")
        print("  3. Keep that terminal open")
        print("  4. In another terminal, run: ollama pull llama3.2")
        print("  5. Restart this backend server")
        print("")
        print("  Ollama is required for AI answers.")
        logger.error("OLLAMA_NOT_RUNNING: startup check failed after retries")

    print("\n" + "="*60)
    print("BACKGROUND STARTUP CHECKS COMPLETE (RAG DB + Ollama)")
    print("="*60 + "\n")
    logger.info("BACKGROUND_READINESS_CHECKS_COMPLETE")


def _save_session(session_id: str, chat_manager: ChatManager):
    """Save session to persistent storage"""
    try:
        chats = data_manager.load_chats()
        
        # Get title from first user message or doc type
        title = f"Chat {session_id}"
        for msg in chat_manager.messages:
            if msg.get('role') == 'user':
                title = msg.get('content', '')[:50]
                if len(msg.get('content', '')) > 50:
                    title += "..."
                break
        
        if chat_manager.doc_type:
            title = f"{chat_manager.doc_type} - {title}"
        
        chats[session_id] = {
            'title': title,
            'date': datetime.now().isoformat(),
            'messages': chat_manager.messages,
            'collected_fields': chat_manager.collected_fields,
            'doc_type': chat_manager.doc_type,
            'stage': chat_manager.stage,
            'has_document': bool(chat_manager.generated_document)
        }
        
        # Keep only last 50 sessions
        if len(chats) > 50:
            oldest = sorted(chats.keys(), key=lambda x: chats[x].get('date', ''))[0]
            del chats[oldest]
        
        data_manager.save_chats(chats)
    except Exception as e:
        # FIX: was print(f"Error saving session: {e}") — invisible once
        # the terminal scrolls past, and never reached a log file even
        # after logging was configured (print() and logging are separate
        # streams). exc_info=True captures the full traceback, same as
        # traceback.print_exc() would, but persisted.
        logger.error(f"SAVE_SESSION_FAILED: session_id={session_id} err={e}", exc_info=True)


# ============================================
# PYDANTIC MODELS
# ============================================

class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: Optional[List[Dict[str, Any]]] = None
    # True when this call is a retry of a previously-failed message (user
    # clicked the retry icon on an "unavailable" reply). On retry, a
    # second failure is NOT persisted to chat history — see chat() below —
    # the frontend shows a transient toast instead of a saved error bubble.
    is_retry: bool = False
    # True when the user edited a previous message and is resending the
    # revised text. When true, edit_index identifies which message in
    # history to overwrite (and everything after it to drop), rather than
    # appending a new user message.
    is_edit: bool = False
    edit_index: Optional[int] = None


class ChatResponse(BaseModel):
    # NOTE: was List[Dict[str, str]], which forces every value in every
    # message dict to be a str. ChatManager.add_message() actually
    # produces {'role': str, 'content': str, 'timestamp': str,
    # 'metadata': dict}, so 'metadata' (a dict, default {}) failed
    # validation as soon as real messages started flowing through (see
    # the ValidationError: "metadata ... Input should be a valid string").
    # Dict[str, Any] matches the real shape.
    messages: List[Dict[str, Any]]
    stage: str
    doc_type: Optional[str] = None
    collected_fields: Dict[str, Any] = {}
    document_text: Optional[str] = None
    has_document: bool = False
    # True only when this response represents a RETRY that failed again.
    # When true, `messages` is unchanged from before the retry (nothing
    # was persisted) and the frontend should show a transient toast
    # rather than render a new error bubble.
    transient_failure: bool = False
    user_type: Optional[str] = None
    # Set only when this turn was a mode-switch (or mode-switch denial,
    # or the fallback "choose question/document" nudge) — the literal
    # user message that triggered it and this confirmation text are NOT
    # persisted to chat history (see chat() route below); the frontend
    # should show this as a transient tooltip/toast, not a chat bubble.
    mode_notice: Optional[str] = None


class NewSessionRequest(BaseModel):
    # Mock login: which type of account this session belongs to. Real
    # auth isn't implemented yet — this mimics it by letting the
    # frontend's login screen specify the type directly. See
    # ChatManager.VALID_USER_TYPES / allowed_modes() for what each type
    # can access.
    user_type: Optional[str] = None


class NewSessionResponse(BaseModel):
    session_id: str
    welcome_message: str
    user_type: Optional[str] = None


class SessionData(BaseModel):
    # Same fix as ChatResponse.messages above.
    messages: List[Dict[str, Any]]
    collected_fields: Dict[str, Any]
    doc_type: Optional[str]
    stage: str
    has_document: bool = False
    user_type: Optional[str] = None


class LoadChatRequest(BaseModel):
    session_id: str
    chat_id: str


# ============================================
# LIFESPAN
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    global rag_engine, search_tool, data_manager
    
    # STARTUP
    print("\n" + "="*60)
    print("STARTING KENYAN LEGAL ASSISTANT API")
    print("="*60)
    logger.info("STARTUP: Kenyan Legal Assistant API starting")
    
    # 1. Initialize Data Manager
    print("\n[1/4] Initializing Data Manager...")
    data_manager = DataManager()
    print("✓ Data Manager ready")
    
    # 2. Load Shared Embeddings Model - FIXED: use load() not load_model()
    print("\n[2/4] Loading Embeddings Model (shared)...")
    try:
        embeddings_service.load()
        print("✓ Embeddings model loaded")
        logger.info("EMBEDDINGS_LOADED")
    except Exception as e:
        print(f"⚠ Embeddings model failed to load: {e}")
        # FIX: this failure used to only print — the app would then run
        # in a state where every single RAG query fails at
        # "SEARCH_SKIPPED: embeddings_service not ready" with no link
        # back to this root cause unless someone was watching the
        # terminal at the exact moment of startup. Logging it as an
        # error (with traceback) means it's findable later when someone
        # is instead staring at a pile of downstream SEARCH_SKIPPED /
        # UNAVAILABLE_MESSAGE log lines and wondering why.
        logger.error(f"EMBEDDINGS_LOAD_FAILED: err={e}", exc_info=True)
    
    # 3. Initialize RAG Engine
    print("\n[3/4] Initializing RAG Engine...")
    rag_engine = RAGEngine()
    # RAGEngine() only kicks off its OWN background thread for the actual
    # Chroma DB load (see rag_engine.py's _load_database) and returns
    # immediately — it does not block here. Readiness is polled below in
    # a separate background thread, not inline, so construction alone
    # doesn't hold up startup either.

    # 4. Initialize Search Tool
    print("\n[4/4] Initializing Legal Search Tool...")
    search_tool = LegalSearchTool()
    if hasattr(search_tool, 'get_stats'):
        stats = search_tool.get_stats()
        if isinstance(stats, tuple):
            known_topics = stats[0].get('known_topics', 0) if stats else 0
        else:
            known_topics = stats.get('known_topics', 0)
        print(f"✓ Search Tool ready (local topics: {known_topics})")
        logger.info(f"SEARCH_TOOL_READY: known_topics={known_topics}")
    else:
        print("✓ Search Tool ready")
        logger.info("SEARCH_TOOL_READY: known_topics=unknown (no get_stats)")

    # RAG DB readiness + Ollama verification run in a background thread
    # rather than blocking here — see _background_readiness_checks'
    # docstring. The app becomes reachable (session creation, history,
    # health checks — none of which need either) as soon as `yield`
    # below runs, typically well under a second from here, instead of
    # after up to 30s of Chroma polling plus Ollama's retry/backoff.
    threading.Thread(
        target=_background_readiness_checks, args=(rag_engine,), daemon=True
    ).start()

    print("\n" + "="*60)
    print("API IS READY!")
    print("="*60)
    print("\nAvailable endpoints:")
    print("  POST /api/new-session  - Start a new chat session")
    print("  POST /api/chat         - Send a message")
    print("  POST /api/load-chat    - Load a saved chat")
    print("  GET  /api/history      - Get all chat sessions")
    print("  GET  /api/history/{session_id} - Get specific session")
    print("  DELETE /api/history/{session_id} - Delete session")
    print("  GET  /api/health       - Health check")
    print("  GET  /api/stats        - System statistics")
    print("\n📌 To ingest PDFs, run: python ingest.py --parallel --workers 2")
    print("="*60 + "\n")
    logger.info("STARTUP_COMPLETE")
    
    yield
    
    # SHUTDOWN
    print("\nShutting down API...")
    logger.info("SHUTDOWN: Kenyan Legal Assistant API stopping")
    if data_manager:
        data_manager.close()
    print("✓ Clean shutdown complete")


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="Kenyan Legal Assistant API",
    description="AI-powered legal information system for Kenyan law",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# API ENDPOINTS
# ============================================

@app.post("/api/debug/remove-sources")
async def debug_remove_sources(sources: List[str]):
    """Remove every chunk whose 'source' metadata exactly matches one of
    the given values, regardless of category. Built specifically to clean
    up the 4 case-law judgments that were ingested through the statute
    pipeline (_ingest_pdf) and got tagged category='kenyan_law_pdf' by
    mistake — see /api/debug/collection-stats' possibly_misfiled_case_law_sources.

    POST body: a JSON list of exact source strings, e.g.
        ["James Kariuki Wagana V Republic 2018Kehc6157(Klr)", ...]
    (copy these verbatim from collection-stats' output to guarantee an
    exact match — this does NOT do fuzzy/partial matching on purpose, to
    avoid accidentally deleting real statute chunks that happen to share
    a substring).

    Debugging aid, not authenticated/rate-limited — same caveat as
    collection-stats above."""
    if not rag_engine or not rag_engine.is_ready() or not rag_engine.collection:
        return {"error": "rag_engine not ready"}

    all_docs = rag_engine.collection.get(include=["metadatas"])
    metadatas = all_docs.get("metadatas", [])
    ids = all_docs.get("ids", [])

    ids_to_delete = []
    for i, m in enumerate(metadatas):
        if m and m.get("source") in sources:
            ids_to_delete.append(ids[i])

    if ids_to_delete:
        rag_engine.collection.delete(ids=ids_to_delete)

    logger.info(f"DEBUG_REMOVE_SOURCES: sources={sources} removed={len(ids_to_delete)}")
    return {"removed_count": len(ids_to_delete), "sources_requested": sources}


@app.post("/api/debug/ingest-case-law")
async def debug_ingest_case_law(folder: str):
    """Runs ingest_case_law_folder() against the live in-process
    rag_engine — use this after moving the misfiled case-law PDFs into
    their own folder and cleaning up their old mistagged chunks via
    /api/debug/remove-sources, to properly re-ingest them with correct
    category='case_law_pdf' tagging and case_name/citation metadata
    extraction. Runs in-process specifically to avoid the concurrent-DB-
    access problem standalone re-ingestion scripts hit while this server
    is running (same reasoning as collection-stats above).

    POST body: {"folder": "data/case_law_pdfs"} (path relative to the
    backend directory, or an absolute path)."""
    if not rag_engine or not rag_engine.is_ready():
        return {"error": "rag_engine not ready"}

    before = rag_engine.collection.count()
    ingestor = PDFIngestor(rag_engine=rag_engine)

    try:
        ingestor.ingest_case_law_folder(folder)
    except Exception as e:
        logger.error(f"DEBUG_INGEST_CASE_LAW_FAILED: folder={folder} err={e}", exc_info=True)
        return {"error": str(e)}

    after = rag_engine.collection.count()
    logger.info(f"DEBUG_INGEST_CASE_LAW: folder={folder} before={before} after={after}")
    return {"before_count": before, "after_count": after, "folder": folder}


@app.get("/api/debug/collection-stats")
async def debug_collection_stats():
    """Diagnostic endpoint: inspects the LIVE, already-loaded rag_engine
    instance's collection directly, in-process. Exists specifically to
    avoid the concurrent-access problems standalone diagnostic scripts hit
    when this server is already running — ChromaDB's persistent
    SQLite-backed client doesn't reliably support two separate processes
    opening the same DB at once (especially on Windows), so a standalone
    script's RAGEngine() often fails to load while this server is up.
    Querying through the server's own already-loaded instance sidesteps
    that entirely, and also guarantees we're looking at the exact
    instance actually serving real traffic — no "which version is
    running" ambiguity.

    Not authenticated/rate-limited — this is a debugging aid, not meant
    to stay exposed in a real production deployment long-term."""
    import re as _re

    if not rag_engine or not rag_engine.is_ready() or not rag_engine.collection:
        return {"error": "rag_engine not ready"}

    all_docs = rag_engine.collection.get(include=["metadatas"])
    metadatas = all_docs.get("metadatas", [])

    category_counts: Dict[str, int] = {}
    missing_category = 0
    for m in metadatas:
        if not m or "category" not in m:
            missing_category += 1
        else:
            cat = m.get("category")
            category_counts[cat] = category_counts.get(cat, 0) + 1

    case_like_pattern = _re.compile(r'\bv\.?\s|\bvs\.?\s|\bversus\b|\brepublic\b', _re.IGNORECASE)
    misfiled_sources = set()
    for m in metadatas:
        if not m or m.get("category") != "kenyan_law_pdf":
            continue
        source = m.get("source", "") or m.get("filename", "")
        if case_like_pattern.search(source):
            misfiled_sources.add(source)

    return {
        "total_documents": len(metadatas),
        "category_counts": category_counts,
        "documents_missing_category_field": missing_category,
        "doc_type_field_in_use": rag_engine.DOC_TYPE_FIELD,
        "case_law_type_in_use": rag_engine.CASE_LAW_TYPE,
        "possibly_misfiled_case_law_sources": sorted(misfiled_sources),
        "possibly_misfiled_count": len(misfiled_sources),
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint.

    FIX: this used to call _check_ollama() TWICE per request (once for
    `status`, once for `ollama_available`). _check_ollama() isn't a
    cheap single ping — it's a 3-attempt retry loop with a 1s sleep
    between attempts, meant for the one-time startup check where
    getting it right matters more than being fast. Called twice, on
    every single /api/health poll (the frontend polls this every 15s
    indefinitely — see Navbar.jsx), that meant up to 6 connection
    attempts and 6 logged warnings every 15 seconds for as long as the
    app stays open with Ollama down — real log noise and real added
    latency on a request that's supposed to be a cheap, fast check.
    Computing it once and reusing the result halves both."""
    ollama_ok = _check_ollama_fast()
    return {
        "status": "healthy" if ollama_ok else "degraded",
        "rag_loaded": rag_engine.is_ready() if rag_engine else False,
        "embeddings_ready": embeddings_service.is_ready() if hasattr(embeddings_service, 'is_ready') else False,
        "ollama_available": ollama_ok,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/new-session", response_model=NewSessionResponse)
async def new_session(request: NewSessionRequest = NewSessionRequest()):
    """Create a new chat session"""
    session_id = str(uuid.uuid4())[:8]

    # Create new chat manager, with user_type fixed at creation (mock
    # login — see NewSessionRequest). user_type is not changeable after
    # this point; ChatManager.user_type has no setter by design.
    chat_manager = ChatManager(session_id=session_id, user_type=request.user_type)
    
    # Store in active sessions
    active_sessions[session_id] = chat_manager

    # Welcome message, tailored slightly by user_type so 'normal' users
    # (Q&A only) aren't told about a document-drafting option they don't
    # have access to.
    if chat_manager.allowed_modes() == {'qa'}:
        welcome = """Jambo! Welcome to Kenyan Legal Assistant. 🇰🇪

Ask me any legal question about Kenyan law to get started."""
    else:
        welcome = """Jambo! Welcome to Kenyan Legal Assistant. 🇰🇪

Pick an option below to ask a legal question or draft an affidavit, then tell me what you need."""
    
    # Add welcome to history
    # FIX: chat_manager.messages returns a COPY (see ChatManager's property
    # getter — `return self._messages.copy()`), so calling .append() on it
    # mutated a throwaway list and never touched the real internal state.
    # add_message() correctly acquires the lock and appends to the real
    # self._messages list. This was the root cause of "nothing displayed" —
    # every message added this way was silently discarded.
    chat_manager.add_message("assistant", welcome)
    
    logger.info(f"NEW_SESSION: session_id={session_id} user_type={chat_manager.user_type}")
    
    return NewSessionResponse(
        session_id=session_id,
        welcome_message=welcome,
        user_type=chat_manager.user_type
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message"""
    session_id = request.session_id
    user_message = request.message.strip()
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Startup readiness checks now run in a background thread (see
    # _background_readiness_checks) instead of blocking the whole app
    # before it accepts any request — so unlike before, a chat message
    # CAN land here in the brief window before the RAG DB has finished
    # loading. rag_engine.search_database() already fails soft in that
    # case (SEARCH_SKIPPED, returns []), so this isn't required for
    # correctness, but an honest "still starting up" message is a better
    # experience than falling through to the generic no-answer fallback
    # for what is, this early, a totally different reason.
    if rag_engine is None or not rag_engine.is_ready():
        logger.info(f"CHAT_DURING_STARTUP: session_id={session_id} query='{user_message}'")
        return ChatResponse(
            messages=active_sessions[session_id].messages if session_id in active_sessions else [],
            stage="greeting",
            transient_failure=True,
            user_type=active_sessions[session_id].user_type if session_id in active_sessions else None,
        )
    
    # Get or create session
    if session_id not in active_sessions:
        chat_manager = ChatManager()
        active_sessions[session_id] = chat_manager
        logger.warning(f"CHAT_SESSION_NOT_FOUND: session_id={session_id} — created a new one")
    
    chat_manager = active_sessions[session_id]

    # Snapshot of state BEFORE this call, so we can roll back to it if a
    # retry fails again (see transient_failure handling below).
    messages_before = chat_manager.messages
    stage_before = chat_manager.stage
    doc_type_before = chat_manager.doc_type
    fields_before = chat_manager.collected_fields
    document_before = chat_manager.generated_document

    # On a normal send, the user's message is added to history as usual.
    # On a RETRY (not an edit), the original user message is already in
    # history from the first (failed) attempt — do not add a second copy.
    # On an EDIT, overwrite the message at edit_index and drop everything
    # after it (the old reply is stale once the question changes).
    if request.is_edit and request.edit_index is not None:
        chat_manager.truncate_and_replace(request.edit_index, "user", user_message)
    elif not request.is_retry:
        chat_manager.add_message("user", user_message)
    
    # Create message handler
    try:
        message_handler = MessageHandler(
            chat_manager=chat_manager,
            interview_engine=InterviewEngine(),
            data_manager=data_manager,
            rag_engine=rag_engine,
            search_tool=search_tool
        )
        
        # Generate response
        response_text = message_handler.process(user_message)
        
        if not response_text:
            response_text = "I'm having trouble processing that. Could you rephrase your question or describe your situation differently?"
            logger.warning(f"EMPTY_RESPONSE_TEXT: session_id={session_id} query='{user_message}'")

        is_unavailable = (response_text.strip() == UNAVAILABLE_MESSAGE_TEXT)
        is_mode_notice = response_text.strip() in MessageHandler.TRANSIENT_MODE_RESPONSES

        if is_mode_notice:
            # Mode-switch confirmations/denials/nudges are transient UI
            # feedback, not real conversation content — neither the
            # user's literal 'question'/'document' message nor this
            # response gets persisted.
            if request.is_edit and request.edit_index is not None:
                # truncate_and_replace already added the edited text as
                # a real message before process() ran — remove it now
                # that we know this edit turned into a mode switch.
                chat_manager.remove_last_message(role="user")
            elif not request.is_retry:
                # Normal send: the user message was added above before
                # we knew what kind of turn this was. On retry, nothing
                # new was added this call, so there's nothing to remove.
                chat_manager.remove_last_message(role="user")
            return ChatResponse(
                messages=chat_manager.messages,
                stage=chat_manager.stage,
                doc_type=chat_manager.doc_type,
                collected_fields=chat_manager.collected_fields,
                document_text=chat_manager.generated_document,
                has_document=bool(chat_manager.generated_document),
                user_type=chat_manager.user_type,
                mode_notice=response_text,
            )

        if is_unavailable:
            # The "unavailable" response is never persisted to chat
            # history, whether this is the first attempt or a retry — the
            # frontend shows a toast instead. This also means the user's
            # message they just sent stays as the LAST message in
            # `messages_before` (since it wasn't a retry, it WAS just
            # added above) — that's correct: the user's question stays
            # visible with its retry/edit icons available, and a second
            # attempt re-sends the same text via the retry icon.
            logger.info(f"CHAT_UNAVAILABLE_RESPONSE: session_id={session_id} query='{user_message}'")
            return ChatResponse(
                messages=chat_manager.messages,
                stage=stage_before,
                doc_type=doc_type_before,
                collected_fields=fields_before,
                document_text=document_before,
                has_document=bool(document_before),
                transient_failure=True,
                user_type=chat_manager.user_type,
            )

        # Normal path: persist the assistant's reply as usual.
        chat_manager.add_message("assistant", response_text)
        
        # Save to persistent storage
        _save_session(session_id, chat_manager)
        
        return ChatResponse(
            messages=chat_manager.messages,
            stage=chat_manager.stage,
            doc_type=chat_manager.doc_type,
            collected_fields=chat_manager.collected_fields,
            document_text=chat_manager.generated_document,
            has_document=bool(chat_manager.generated_document),
            user_type=chat_manager.user_type,
        )
    
    except Exception as e:
        # FIX: was print(f"Error processing chat: {e}") + traceback.print_exc()
        # — both go to stdout/stderr only, never to the log file, so the
        # exact same class of failure that logging was supposed to make
        # debuggable was invisible for the one endpoint that matters most.
        # logger.error(..., exc_info=True) writes the full traceback to
        # both the rotating file and the console in one call.
        logger.error(
            f"CHAT_PROCESSING_FAILED: session_id={session_id} query='{user_message}' err={e}",
            exc_info=True,
        )

        if request.is_retry:
            # Same treatment as an unavailable retry: don't persist, show
            # a transient toast instead.
            return ChatResponse(
                messages=messages_before,
                stage=stage_before,
                doc_type=doc_type_before,
                collected_fields=fields_before,
                document_text=document_before,
                has_document=bool(document_before),
                transient_failure=True,
                user_type=chat_manager.user_type,
            )
        
        error_response = "I encountered an error processing your request. Please try again or rephrase your question."
        # FIX: same .append()-on-a-copy bug — use add_message().
        chat_manager.add_message("assistant", error_response)
        
        return ChatResponse(
            messages=chat_manager.messages,
            stage=chat_manager.stage,
            doc_type=chat_manager.doc_type,
            collected_fields=chat_manager.collected_fields,
            document_text=chat_manager.generated_document,
            has_document=False,
            user_type=chat_manager.user_type,
        )


@app.post("/api/load-chat")
async def load_chat(request: LoadChatRequest):
    """Load a saved chat into a session"""
    session_id = request.session_id
    chat_id = request.chat_id
    
    chats = data_manager.load_chats()
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    chat_data = chats[chat_id]
    
    # Get or create session
    if session_id not in active_sessions:
        chat_manager = ChatManager()
        active_sessions[session_id] = chat_manager
    else:
        chat_manager = active_sessions[session_id]
    
    # Load the saved data
    chat_manager.messages = chat_data.get('messages', [])
    chat_manager.collected_fields = chat_data.get('collected_fields', {})
    chat_manager.doc_type = chat_data.get('doc_type')
    chat_manager.generated_document = None
    chat_manager.stage = chat_data.get('stage', 'greeting')
    
    _save_session(session_id, chat_manager)
    
    logger.info(f"LOAD_CHAT: session_id={session_id} chat_id={chat_id}")
    
    return {"status": "loaded", "session_id": session_id}


@app.get("/api/history")
async def get_history():
    """Get all chat sessions"""
    try:
        chats = data_manager.load_chats()
        
        sessions = []
        for session_id, data in chats.items():
            sessions.append({
                'session_id': session_id,
                'title': data.get('title', f"Chat {session_id[:8]}"),
                'date': data.get('date', ''),
                'doc_type': data.get('doc_type'),
                'has_document': data.get('has_document', False),
                'message_count': len(data.get('messages', []))
            })
        
        sessions.sort(key=lambda x: x['date'], reverse=True)
        
        return {'chats': {s['session_id']: s for s in sessions}}
    except Exception as e:
        # FIX: was print() — see rationale above; now persisted with
        # a traceback so a broken history load is actually debuggable.
        logger.error(f"GET_HISTORY_FAILED: err={e}", exc_info=True)
        return {'chats': {}}


@app.get("/api/history/{session_id}")
async def get_session(session_id: str):
    """Get a specific chat session"""
    try:
        chats = data_manager.load_chats()
        
        if session_id not in chats:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = chats[session_id]
        
        return SessionData(
            messages=session_data.get('messages', []),
            collected_fields=session_data.get('collected_fields', {}),
            doc_type=session_data.get('doc_type'),
            stage=session_data.get('stage', 'greeting'),
            has_document=session_data.get('has_document', False)
        )
    except HTTPException:
        raise
    except Exception as e:
        # FIX: was print() — now logged with a traceback before the 500
        # is raised, so the underlying cause is recoverable from logs.
        logger.error(f"GET_SESSION_FAILED: session_id={session_id} err={e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/history/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    try:
        chats = data_manager.load_chats()
        
        if session_id in chats:
            del chats[session_id]
            data_manager.save_chats(chats)
            
            # Remove from active sessions if present
            if session_id in active_sessions:
                del active_sessions[session_id]
            
            logger.info(f"SESSION_DELETED: session_id={session_id}")
            return {"status": "deleted", "session_id": session_id}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        # FIX: was print() — now logged with a traceback.
        logger.error(f"DELETE_SESSION_FAILED: session_id={session_id} err={e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    stats = {
        "rag": {
            "loaded": rag_engine.is_ready() if rag_engine else False,
            "documents": rag_engine.collection.count() if rag_engine and rag_engine.collection else 0
        },
        "embeddings": {
            "loaded": embeddings_service.is_ready() if hasattr(embeddings_service, 'is_ready') else False,
        },
        "ollama": {
            "available": _check_ollama_fast(),
            "model": rag_engine.ollama_model if rag_engine and hasattr(rag_engine, 'ollama_model') else "unknown"
        },
        "sessions": {
            "active": len(active_sessions),
            "total_stored": len(data_manager.load_chats()) if data_manager else 0
        }
    }
    
    # Add search tool stats if available
    if search_tool and hasattr(search_tool, 'get_stats'):
        try:
            tool_stats = search_tool.get_stats()
            if isinstance(tool_stats, tuple):
                stats["search_tool"] = {"topics": tool_stats[0].get('known_topics', 0) if tool_stats else 0}
            else:
                stats["search_tool"] = {"topics": tool_stats.get('known_topics', 0)}
        except Exception as e:
            # FIX: was a bare `except: stats["search_tool"] = {"topics": 0}`
            # — now the underlying failure is at least logged before
            # falling back to the same default.
            logger.warning(f"SEARCH_TOOL_STATS_FAILED: err={e}")
            stats["search_tool"] = {"topics": 0}
    
    # Add citation index size if available
    if rag_engine and hasattr(rag_engine, 'citation_index'):
        stats["citation_index_size"] = len(rag_engine.citation_index)
    
    return stats


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("KENYAN LEGAL ASSISTANT - BACKEND SERVER")
    print("="*60)
    
    print("\n⚠ IMPORTANT: Make sure Ollama is running in another terminal!")
    print("   Terminal 1: ollama serve")
    print("   Terminal 2: ollama pull llama3.2")
    print("")
    
    # Check for required directories
    if not PDF_DIR.exists():
        PDF_DIR.mkdir(parents=True)
        print(f"📁 Created PDF directory: {PDF_DIR}")
        print("   Place Kenyan legal PDFs here")
    
    print("\n📌 To ingest PDFs, run in another terminal:")
    print("   python ingest.py --parallel --workers 2")
    
    print("\n🚀 Starting server on http://0.0.0.0:8000")
    print("   Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
