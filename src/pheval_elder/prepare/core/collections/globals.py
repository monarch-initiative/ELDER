"""
Global variables for collections and other shared resources.

This module contains global variables that are used for sharing resources
between different processes. These globals are necessary because ChromaDB
collections are not pickleable and cannot be passed directly to worker
processes in multiprocessing.
"""

from typing import Any

# Global variables for multiprocessing
# These should be initialized before use
global_avg_disease_emb_collection: Any = None
global_wgt_avg_disease_embd_collection: Any = None