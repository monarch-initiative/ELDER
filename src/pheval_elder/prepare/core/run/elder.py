"""
Elder runner module for running analysis on phenotype sets.

This module contains the ElderRunner class, which is responsible for running
analysis on phenotype sets using different strategies.
"""

import time
from dataclasses import dataclass
from typing import Iterator, List, Any, Dict, Optional, Union, Type

from pheval.post_processing.post_processing import PhEvalDiseaseResult

from pheval_elder.prepare.core.collections.disease_avg_embedding_service import DiseaseAvgEmbeddingService
from pheval_elder.prepare.core.collections.disease_weighted_avg_embedding_service import DiseaseWeightedAvgEmbeddingService
from pheval_elder.prepare.core.data_processing.data_processor import DataProcessor
from pheval_elder.prepare.core.query.multi_process_best_match import OptimizedMultiprocessing
from pheval_elder.prepare.core.query.termsetpairwise import TermSetPairWiseComparisonQuery
from pheval_elder.prepare.core.store.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures

# Import multiprocessing modules
from pheval_elder.prepare.core.multiprocessing.avg_multiprocessing import (
    OptimizedAverageDiseaseEmbedAnalysis,
    process_avg_analysis_parallel,
)
from pheval_elder.prepare.core.multiprocessing.wgt_avg_multiprocessing import (
    OptimizedWeightedAverageDiseaseEmbedAnalysis,
    process_wgt_avg_analysis_parallel,
)
from pheval_elder.prepare.core.multiprocessing.best_match_multiprocessing import (
    OptimizedTermSetPairwiseComparison,
    process_phenotype_sets_parallel,
)


@dataclass
class ElderRunner:
    """
    Runner for Elder analysis strategies.
    
    This class is responsible for running analysis on phenotype sets using
    different strategies, such as average embeddings, weighted average embeddings,
    and term-set pairwise comparison.
    """
    collection_name: str
    embedding_model: str
    strategy: str
    nr_of_phenopackets: str
    db_collection_path: str
    nr_of_results: int
    similarity_measure: SimilarityMeasures = SimilarityMeasures.COSINE
    results_dir_name: Optional[str] = None
    results_sub_dir: Optional[str] = None
    
    # Services for different strategies
    data_processor: Optional[DataProcessor] = None
    disease_service: Optional[DiseaseAvgEmbeddingService] = None
    disease_weighted_service: Optional[DiseaseWeightedAvgEmbeddingService] = None
    db_manager: Optional[ChromaDBManager] = None

    def __post_init__(self):
        """Initialize the runner after dataclass initialization."""
        # Normalize embedding model name
        self.embedding_model = self.embedding_model.lower()
        
        # Set result directory names if not provided
        if not self.results_dir_name:
            self.results_dir_name = f"{self.embedding_model}_{self.strategy}_{self.nr_of_phenopackets}pp_top{self.nr_of_results}"
        if not self.results_sub_dir:
            self.results_sub_dir = f"{self.embedding_model}_{self.strategy}_{self.collection_name}_{self.nr_of_phenopackets}pp_top{self.nr_of_results}"
        
        # Initialize database manager
        self.db_manager = ChromaDBManager(
            similarity=self.similarity_measure,
            collection_name=self.collection_name,
            model_shorthand=self.embedding_model,
            strategy=self.strategy,
            nr_of_phenopackets=self.nr_of_phenopackets,
            path=self.db_collection_path,
            nr_of_results=self.nr_of_results
        )
        
        # Initialize data processor
        self.data_processor = DataProcessor(db_manager=self.db_manager)
        
        # Initialize services based on strategy
        if self.strategy == "avg":
            self.disease_service = DiseaseAvgEmbeddingService(data_processor=self.data_processor)
        elif self.strategy == "wgt_avg":
            self.disease_weighted_service = DiseaseWeightedAvgEmbeddingService(data_processor=self.data_processor)

    def initialize_data(self):
        """Initialize data for processing."""
        print(f"Initializing data for strategy: {self.strategy}")
        # Load HP embeddings and disease-to-HP mappings
        _ = self.data_processor.hp_embeddings
        _ = self.data_processor.disease_to_hps
        if self.strategy == "wgt_avg":
            _ = self.data_processor.disease_to_hps_with_frequencies

    def setup_collections(self):
        """Set up collections for processing."""
        print(f"Setting up collections for strategy: {self.strategy}")
        if self.strategy == "avg" and self.disease_service:
            self.disease_service.process_data()
        elif self.strategy == "wgt_avg" and self.disease_weighted_service:
            self.disease_weighted_service.process_data()

    def optimized_avg_analysis(self, phenotype_sets, nr_of_results):
        """Run optimized average embedding analysis on phenotype sets."""
        print(f"Running optimized average analysis on {len(phenotype_sets)} phenotype sets")
        oadea = OptimizedAverageDiseaseEmbedAnalysis(
            data_processor=self.data_processor
        )
        return process_avg_analysis_parallel(phenotype_sets, oadea, nr_of_results)

    def avg_analysis(self, input_hpos, nr_of_results):
        """Run average embedding analysis on phenotype sets."""
        print(f"Running average analysis on {len(input_hpos)} phenotype sets")
        from pheval_elder.prepare.core.query.query_service import QueryService
        query_service = QueryService(
            data_processor=self.data_processor,
            db_manager=self.db_manager,
            average_llm_embedding_service=self.disease_service,
        )
        return query_service.query_for_average_llm_embeddings_collection_top10_only(
            hpo_ids=input_hpos,
            n_results=nr_of_results
        )

    def optimized_wgt_avg_analysis(self, phenotype_sets, nr_of_results):
        """Run optimized weighted average embedding analysis on phenotype sets."""
        print(f"Running optimized weighted average analysis on {len(phenotype_sets)} phenotype sets")
        owadea = OptimizedWeightedAverageDiseaseEmbedAnalysis(
            data_processor=self.data_processor,
        )
        return process_wgt_avg_analysis_parallel(phenotype_sets, owadea, nr_of_results)

    def wgt_avg_analysis(self, input_hpos, nr_of_results):
        """Run weighted average embedding analysis on phenotype sets."""
        print(f"Running weighted average analysis on {len(input_hpos)} phenotype sets")
        from pheval_elder.prepare.core.query.query_service import QueryService
        query_service = QueryService(
            data_processor=self.data_processor,
            db_manager=self.db_manager,
            weighted_average_llm_embedding_service=self.disease_weighted_service,
        )
        return query_service.query_for_weighted_average_llm_embeddings_collection_top10_only(
            hps=input_hpos,
            n_results=nr_of_results
        )

    def tcp_analysis_multi(self, input_hpos, nr_of_results):
        """Run multi-process term-set pairwise comparison analysis on phenotype sets."""
        print(f"Running multi-process TCP analysis on {len(input_hpos)} phenotype sets")
        tcp = OptimizedMultiprocessing(data_processor=self.data_processor)
        return tcp.process_all_sets(input_hpos, nr_of_results)

    def tcp_analysis_normal(self, input_hpos, nr_of_results) -> Iterator[PhEvalDiseaseResult]:
        """Run normal term-set pairwise comparison analysis on phenotype sets."""
        print(f"Running normal TCP analysis on {len(input_hpos)} phenotype sets")
        tcp = TermSetPairWiseComparisonQuery(data_processor=self.data_processor)
        return tcp.termset_pairwise_comparison_disease_embeddings(input_hpos, nr_of_results)

    def tcp_analysis_optimized(self, phenotype_sets: List[List[str]], nr_of_results: int) -> List[
        List[PhEvalDiseaseResult]]:
        """Run optimized term-set pairwise comparison analysis on phenotype sets."""
        print(f"Running optimized TCP analysis on {len(phenotype_sets)} phenotype sets")
        start_time = time.time()
        tcp = OptimizedTermSetPairwiseComparison(data_processor=self.data_processor)
        tcp.precompute_similarities(phenotype_sets)
        results = process_phenotype_sets_parallel(phenotype_sets, tcp, nr_of_results)
        end_time = time.time()
        print(f"TCP analysis completed in {end_time - start_time:.2f} seconds")
        return results