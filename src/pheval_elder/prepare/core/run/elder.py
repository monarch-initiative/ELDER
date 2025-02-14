from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Set, List, Iterable

from pheval.post_processing.post_processing import PhEvalDiseaseResult

from pheval_elder.prepare.core.query.multi_process_best_match import OptimizedMultiprocessing
from pheval_elder.prepare.core.query.optimized_parallel_best_match import OptimizedTermSetPairwiseComparison, \
    process_phenotype_sets_parallel
from pheval_elder.prepare.core.query.termsetpairwise import TermSetPairWiseComparisonQuery
from pheval_elder.prepare.core.store.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processing.data_processor import DataProcessor
from pheval_elder.prepare.core.collections.disease_avg_embedding_service import DiseaseAvgEmbeddingService
from pheval_elder.prepare.core.collections.disease_weighted_avg_embedding_service import DiseaseWeightedAvgEmbeddingService
from pheval_elder.prepare.core.query.query_service import (QueryService)
from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures
from enum import Enum, auto


@dataclass
class ElderRunner:
    collection_name: str
    embedding_model: str
    strategy: str
    nr_of_phenopackets: str
    db_collection_path: str
    nr_of_results: int
    similarity_measure: SimilarityMeasures = SimilarityMeasures.COSINE
    results_dir_name: str = None
    results_sub_dir: str = None

    def __post_init__(self):
        """
        Data Resources
        """
        # self.strategy = self.strategy.lower()
        self.embedding_model = self.embedding_model.lower()
        self.results_dir_name = self.embedding_model + "_" + self.strategy + "_" + self.nr_of_phenopackets + "pp" + f"_top{self.nr_of_results}"
        self.results_sub_dir = self.embedding_model + "_" + self.strategy + "_" + self.collection_name + "_" + self.nr_of_phenopackets + "pp" + f"_top{self.nr_of_results}"
        self.db_manager = ChromaDBManager(
            similarity=self.similarity_measure,
            collection_name=self.collection_name,
            model_shorthand=self.embedding_model,
            strategy=self.strategy,
            nr_of_phenopackets=self.nr_of_phenopackets,
            path = self.db_collection_path,
            nr_of_results = self.nr_of_results
        )

        self.data_processor = DataProcessor(db_manager=self.db_manager)
        # self.disease_service = DiseaseAvgEmbeddingService(data_processor=self.data_processor)
        # self.disease_weighted_service = DiseaseWeightedAvgEmbeddingService(data_processor=self.data_processor)

    def initialize_data(self):
        _ = self.data_processor.hp_embeddings
        _ = self.data_processor.disease_to_hps
        _ = self.data_processor.disease_to_hps_with_frequencies
        print(len(self.data_processor.hp_embeddings))
        print(len(self.data_processor.disease_to_hps))
        print(len(self.data_processor.disease_to_hps_with_frequencies))

    def setup_collections(self):
        # self.disease_service.process_data()
        pass

    def avg_analysis(self, input_hpos, nr_of_results):
        query_service = QueryService(
            data_processor=self.data_processor,
            db_manager=self.db_manager,
            average_llm_embedding_service=self.disease_service,
        )
        avg_strategy_result = query_service.query_for_average_llm_embeddings_collection_top10_only(
            hpo_ids=input_hpos,
            n_results=nr_of_results
        )
        return avg_strategy_result

    def wgt_avg_analysis(self, input_hpos, nr_of_results):
        query_service = QueryService(
            data_processor=self.data_processor,
            db_manager=self.db_manager,
            weighted_average_llm_embedding_service=self.disease_weighted_service,
        )
        weighted_avg_strategy_result = query_service.query_for_weighted_average_llm_embeddings_collection_top10_only(
            hps= input_hpos,
            n_results=nr_of_results)
        return weighted_avg_strategy_result

    def tcp_analysis_multi(self, input_hpos, nr_of_results):
        tcp = OptimizedMultiprocessing(data_processor=self.data_processor)
        return tcp.process_all_sets(input_hpos, nr_of_results)

    def tcp_analysis_normal(self, input_hpos, nr_of_results) -> Iterator[PhEvalDiseaseResult]:
        tcp = TermSetPairWiseComparisonQuery(data_processor=self.data_processor)
        return tcp.termset_pairwise_comparison_disease_embeddings(input_hpos, nr_of_results)

    def tcp_analysis_optimized(self, phenotype_sets: List[List[str]], nr_of_results: int) -> List[
        List[PhEvalDiseaseResult]]:
        tcp = OptimizedTermSetPairwiseComparison(data_processor=self.data_processor)
        tcp.precompute_similarities(phenotype_sets)
        return process_phenotype_sets_parallel(phenotype_sets, tcp, nr_of_results)
# if __name__ == "__main__":
#     elder = ElderRunner(collection_name="lrd_hpo", strategy="tpc")
#     elder.initialize_data()


