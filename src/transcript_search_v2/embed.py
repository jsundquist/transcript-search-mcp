"""Local embedding generation via sentence-transformers.

Lazily loaded (the model is ~80MB and takes a moment to load) so importing
this module -- e.g. for tests that never call embed() -- stays cheap.
"""

from __future__ import annotations

import threading

from transcript_search_v2 import config

_model_instance = None
_model_lock = threading.Lock()


def _model():
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:  # re-check: another thread may have won the race
                from sentence_transformers import SentenceTransformer

                # device="cpu": this model is embedded from a background
                # asyncio.to_thread worker, and sentence-transformers'
                # default device auto-selection picks MPS on Apple Silicon --
                # initializing MPS/Metal from a non-main thread hangs
                # indefinitely. CPU is plenty fast for this model/corpus size
                # and sidesteps the whole class of GPU/thread-affinity issues.
                try:
                    # Skip Hub re-validation HEAD requests once the model is
                    # already cached locally -- otherwise every process
                    # startup pays a slow network round-trip for no reason.
                    _model_instance = SentenceTransformer(
                        config.EMBEDDING_MODEL_NAME, device="cpu", local_files_only=True
                    )
                except Exception:
                    _model_instance = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device="cpu")
    return _model_instance


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts. Returns one 384-dim vector per input."""
    if not texts:
        return []
    vectors = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
