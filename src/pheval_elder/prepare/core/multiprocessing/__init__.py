"""
Multiprocessing module for Elder.

This module contains code for parallel processing of phenotype sets
using different strategies.
"""

from pheval_elder.prepare.core.multiprocessing.avg_multiprocessing import (
    process_avg_analysis_parallel,
    OptimizedAverageDiseaseEmbedAnalysis,
)
from pheval_elder.prepare.core.multiprocessing.wgt_avg_multiprocessing import (
    process_wgt_avg_analysis_parallel,
    OptimizedWeightedAverageDiseaseEmbedAnalysis,
)
from pheval_elder.prepare.core.multiprocessing.best_match_multiprocessing import (
    process_phenotype_sets_parallel,
    OptimizedTermSetPairwiseComparison,
)

__all__ = [
    "process_avg_analysis_parallel",
    "OptimizedAverageDiseaseEmbedAnalysis",
    "process_wgt_avg_analysis_parallel",
    "OptimizedWeightedAverageDiseaseEmbedAnalysis",
    "process_phenotype_sets_parallel",
    "OptimizedTermSetPairwiseComparison",
]