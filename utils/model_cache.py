"""
Shared model cache utility to prevent threading issues with SentenceTransformer.
This module ensures only one model instance is created and reused across the application.
"""
import os
import threading
import warnings

# Set environment variables early to prevent threading issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Suppress warnings that might cause issues
warnings.filterwarnings("ignore")

# Thread-safe singleton pattern
_model_instance = None
_model_lock = threading.Lock()
_model_loading = False

def get_embedding_model():
    """
    Get or create the SentenceTransformer model instance.
    Uses thread-safe singleton pattern to ensure only one instance exists.
    Lazy loading to prevent import-time crashes.
    """
    global _model_instance, _model_loading
    
    if _model_instance is None and not _model_loading:
        with _model_lock:
            # Double-check pattern
            if _model_instance is None and not _model_loading:
                try:
                    _model_loading = True
                    # Import here to avoid import-time crashes
                    from sentence_transformers import SentenceTransformer
                    
                    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                    _model_instance = SentenceTransformer(
                        model_name,
                        device='cpu'  # Force CPU to avoid GPU threading issues
                    )
                except Exception as e:
                    _model_loading = False
                    raise RuntimeError(f"Failed to load embedding model: {e}") from e
                finally:
                    _model_loading = False
    
    if _model_instance is None:
        raise RuntimeError("Model instance is None after loading attempt")
    
    return _model_instance

