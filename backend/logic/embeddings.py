"""
Shared embeddings service - Production ready
- Loads model from local storage only
- Caches embeddings for performance
- Health checks and monitoring
- Thread-safe singleton
- Graceful degradation
"""

import hashlib
import pickle
import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from threading import Lock, Thread
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model information and health status"""
    loaded: bool
    path: str
    dimension: int = 384
    last_used: Optional[datetime] = None
    total_encodes: int = 0
    cache_size: int = 0
    cache_hits: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class EmbeddingsService:
    """
    Production-ready singleton embeddings service.
    
    Features:
    - Loads model once, shared across all components
    - Local-only loading (no internet required)
    - Embedding cache for performance
    - Health monitoring
    - Thread-safe operations
    - Graceful error handling
    - Automatic cache management
    """
    
    _instance = None
    _lock = Lock()
    
    # Cache configuration
    MAX_CACHE_AGE_DAYS = 30
    MAX_CACHE_SIZE_MB = 500
    CACHE_CLEANUP_THRESHOLD = 0.8  # Clean when 80% full
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        # Core components
        self._model = None
        self._model_lock = Lock()
        self._cache_lock = Lock()
        
        # Paths
        self._model_dir = Path(__file__).parent.parent / "data" / "models"
        self._cache_dir = Path(__file__).parent.parent / "data" / "embeddings_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self._loading = False
        self._loaded = False
        self._load_error = None
        
        # Health & monitoring
        self._model_info = ModelInfo(
            loaded=False,
            path="",
            errors=[]
        )
        self._health_check_interval = 300  # 5 minutes
        self._last_health_check = None
        
        # Start background tasks
        self._start_cache_cleanup()
        
        logger.info("EmbeddingsService initialized")
    
    # ============ PUBLIC API ============
    
    def load(self, force: bool = False) -> bool:
        """
        Load the embedding model from local storage.
        
        Args:
            force: Force reload even if already loaded
            
        Returns:
            bool: True if model loaded successfully
        """
        if self._loaded and not force:
            return True
        
        if self._loading:
            logger.debug("Model already loading, waiting...")
            timeout = 30
            start = time.time()
            while self._loading and (time.time() - start) < timeout:
                time.sleep(0.5)
            return self._loaded
        
        with self._model_lock:
            self._loading = True
            try:
                from sentence_transformers import SentenceTransformer
                
                # Force offline mode - critical for production
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                
                # Find model path
                model_path = self._find_model_path()
                
                if not model_path:
                    self._load_error = (
                        "Model not found. Please run: python setup_embeddings.py"
                    )
                    logger.error(f"❌ {self._load_error}")
                    self._model_info.errors.append(self._load_error)
                    return False
                
                logger.info(f"Loading model from: {model_path}")
                start_time = time.time()
                
                # Load model with production settings
                self._model = SentenceTransformer(
                    str(model_path),
                    local_files_only=True,
                    device="cpu",  # CPU is more stable for production
                    cache_folder=str(self._model_dir)
                )
                
                load_time = time.time() - start_time
                
                # Verify model works
                test_embedding = self._model.encode(
                    ["test"],
                    show_progress_bar=False
                )
                
                self._loaded = True
                
                # Update model info
                self._model_info = ModelInfo(
                    loaded=True,
                    path=str(model_path),
                    dimension=len(test_embedding[0]),
                    last_used=datetime.now(),
                    total_encodes=0,
                    cache_size=self._get_cache_size_mb()
                )
                
                logger.info(
                    f"✓ Model loaded successfully in {load_time:.1f}s\n"
                    f"  Path: {model_path}\n"
                    f"  Dimension: {self._model_info.dimension}\n"
                    f"  Cache: {self._model_info.cache_size}MB"
                )
                
                return True
                
            except ImportError as e:
                self._load_error = (
                    "sentence_transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
                logger.error(f"❌ {self._load_error}")
                self._model_info.errors.append(str(e))
                return False
                
            except Exception as e:
                self._load_error = str(e)
                logger.error(f"❌ Failed to load model: {e}", exc_info=True)
                self._model_info.errors.append(str(e))
                return False
                
            finally:
                self._loading = False
    
    def is_ready(self) -> bool:
        """Check if model is loaded and healthy"""
        if not self._loaded or self._model is None:
            return False
        
        # Perform periodic health check
        if self._should_health_check():
            return self._perform_health_check()
        
        return True
    
    def encode(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            use_cache: Whether to use embedding cache
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If model not loaded
        """
        if not self.is_ready():
            raise RuntimeError(
                self._load_error or "Embeddings service not loaded"
            )
        
        start_time = time.time()
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        cache_hits = 0
        
        # Check cache first
        if use_cache:
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    results[i] = [0.0] * self._model_info.dimension
                    continue
                
                cache_key = hashlib.md5(text.encode()).hexdigest()
                cache_path = self._cache_dir / f"{cache_key}.pkl"
                
                if cache_path.exists():
                    try:
                        with self._cache_lock:
                            with open(cache_path, 'rb') as f:
                                results[i] = pickle.load(f)
                        cache_hits += 1
                    except (pickle.UnpicklingError, EOFError, IOError):
                        # Corrupted cache, remove it
                        cache_path.unlink(missing_ok=True)
                        uncached_indices.append(i)
                        uncached_texts.append(text)
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    results[i] = [0.0] * self._model_info.dimension
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        
        # Encode uncached texts
        if uncached_texts:
            try:
                embeddings = self._model.encode(
                    uncached_texts,
                    show_progress_bar=False,
                    batch_size=32,
                    normalize_embeddings=True
                ).tolist()
                
                # Cache results
                if use_cache:
                    for idx, text, embedding in zip(
                        uncached_indices, uncached_texts, embeddings
                    ):
                        try:
                            cache_key = hashlib.md5(text.encode()).hexdigest()
                            cache_path = self._cache_dir / f"{cache_key}.pkl"
                            
                            with self._cache_lock:
                                with open(cache_path, 'wb') as f:
                                    pickle.dump(embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
                        except IOError as e:
                            logger.warning(f"Failed to cache embedding: {e}")
                        
                        results[idx] = embedding
                else:
                    for idx, embedding in zip(uncached_indices, embeddings):
                        results[idx] = embedding
                        
            except Exception as e:
                logger.error(f"Failed to encode texts: {e}", exc_info=True)
                self._model_info.errors.append(str(e))
                raise
        
        # Update stats
        self._model_info.total_encodes += len(texts)
        self._model_info.cache_hits += cache_hits
        self._model_info.last_used = datetime.now()
        
        encode_time = time.time() - start_time
        if encode_time > 1.0:  # Log slow encodes
            logger.debug(
                f"Encoded {len(texts)} texts in {encode_time:.2f}s "
                f"(cache hits: {cache_hits})"
            )
        
        return results
    
    def encode_single(self, text: str) -> List[float]:
        """Encode a single text"""
        return self.encode([text])[0]
    
    def get_health(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "loaded": self._loaded,
            "model_path": self._model_info.path,
            "dimension": self._model_info.dimension,
            "last_used": (
                self._model_info.last_used.isoformat()
                if self._model_info.last_used else None
            ),
            "total_encodes": self._model_info.total_encodes,
            "cache_size_mb": self._get_cache_size_mb(),
            "cache_hits": self._model_info.cache_hits,
            "cache_hit_rate": (
                self._model_info.cache_hits / max(self._model_info.total_encodes, 1)
            ),
            "errors": self._model_info.errors[-10:],  # Last 10 errors
            "uptime": (
                str(datetime.now() - self._model_info.last_used)
                if self._model_info.last_used else "N/A"
            )
        }
    
    def clear_cache(self, max_age_days: int = None):
        """
        Clear embedding cache.
        
        Args:
            max_age_days: Only clear entries older than this (None = clear all)
        """
        if max_age_days is None:
            max_age_days = self.MAX_CACHE_AGE_DAYS
        
        cutoff = datetime.now() - timedelta(days=max_age_days)
        count = 0
        freed_bytes = 0
        
        with self._cache_lock:
            for cache_file in self._cache_dir.glob("*.pkl"):
                try:
                    mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if mtime < cutoff:
                        freed_bytes += cache_file.stat().st_size
                        cache_file.unlink()
                        count += 1
                except OSError:
                    continue
        
        if count > 0:
            logger.info(
                f"Cache cleanup: removed {count} entries, "
                f"freed {freed_bytes / 1024 / 1024:.1f}MB"
            )
    
    # ============ PRIVATE METHODS ============
    
    def _find_model_path(self) -> Optional[Path]:
        """Find model in local storage"""
        paths_to_check = [
            self._model_dir / "all-MiniLM-L6-v2",
            Path.home() / ".cache" / "huggingface" / "hub" / 
            "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots",
        ]
        
        for base_path in paths_to_check:
            if not base_path.exists():
                continue
            
            # If it's a snapshots directory, get latest
            if base_path.name == "snapshots":
                snapshots = sorted(base_path.iterdir())
                if snapshots:
                    return snapshots[-1]
            else:
                # Check for required files
                required = ['config.json', 'model.safetensors']
                if all((base_path / f).exists() for f in required):
                    return base_path
        
        return None
    
    def _should_health_check(self) -> bool:
        """Determine if health check is needed"""
        if self._last_health_check is None:
            return True
        
        elapsed = (datetime.now() - self._last_health_check).total_seconds()
        return elapsed > self._health_check_interval
    
    def _perform_health_check(self) -> bool:
        """Perform model health check"""
        try:
            # Quick encode test
            self._model.encode(["health_check"], show_progress_bar=False)
            self._last_health_check = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._loaded = False
            self._model_info.errors.append(f"Health check: {str(e)}")
            return False
    
    def _get_cache_size_mb(self) -> float:
        """Get cache size in MB"""
        if not self._cache_dir.exists():
            return 0.0
        
        total = sum(
            f.stat().st_size 
            for f in self._cache_dir.glob("*.pkl")
        )
        return total / (1024 * 1024)
    
    def _start_cache_cleanup(self):
        """Start background cache cleanup thread"""
        def cleanup_worker():
            while True:
                time.sleep(3600)  # Run every hour
                try:
                    cache_size = self._get_cache_size_mb()
                    if cache_size > self.MAX_CACHE_SIZE_MB:
                        logger.info(
                            f"Cache size {cache_size:.0f}MB exceeds limit, cleaning..."
                        )
                        self.clear_cache(max_age_days=7)  # Keep last week
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        thread = Thread(target=cleanup_worker, daemon=True, name="cache-cleanup")
        thread.start()


# Global singleton instance
embeddings_service = EmbeddingsService()