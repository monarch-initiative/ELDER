"""
ELDER: A framework to evaluate the use case of LLM embeddings for differential diagnosis of rare-diseases.
"""

import importlib_metadata

try:
    __version__ = importlib_metadata.version(__name__)
except importlib_metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"  # pragma: no cover
