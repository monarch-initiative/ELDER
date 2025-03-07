import gc
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Set, List, Iterable, Tuple, Any
from chromadb.types import Collection
from torch.backends.opt_einsum import strategy

import pheval_elder.prepare.core.collections.globals as g

import numpy as np
from pheval.post_processing.post_processing import PhEvalDiseaseResult
from tqdm import tqdm

from pheval_elder.prepare.core.collections.globals import global_avg_disease_emb_collection
from pheval_elder.prepare.core.query.multi_process_best_match import OptimizedMultiprocessing
from pheval_elder.prepare.core.query.optimized_parallel_best_match import OptimizedTermSetPairwiseComparison, \
    process_phenotype_sets_parallel, distribute_sets_evenly
from pheval_elder.prepare.core.query.termsetpairwise import TermSetPairWiseComparisonQuery
from pheval_elder.prepare.core.store.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processing.data_processor import DataProcessor
from pheval_elder.prepare.core.collections.disease_avg_embedding_service import DiseaseAvgEmbeddingService
from pheval_elder.prepare.core.collections.disease_weighted_avg_embedding_service import DiseaseWeightedAvgEmbeddingService
from pheval_elder.prepare.core.query.query_service import (QueryService)
from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures
from enum import Enum, auto
import multiprocessing as mp


class OptimizedWeightedAverageDiseaseEmbedAnalysis:
    def __init__(self, data_processor:DataProcessor):
        self.data_processor = data_processor
        self.hp_embeddings = data_processor.hp_embeddings
        # self.disease_to_hps_with_frequencies = data_processor.disease_to_hps_with_frequencies

    def get_parallel_processing_data(self):
        return self.hp_embeddings

def calculate_average_embedding(query, map):
    return np.mean([map[i]['embeddings'] for i in query if i in map], axis=0)

def sort_and_create_pheval_disease_results(query) -> list[PhEvalDiseaseResult]:
    disease_ids = query['ids'][0] if 'ids' in query and query['ids'] else []
    distances = query['distances'][0] if 'distances' in query and query['distances'] else []
    disease_names = (
        [metadata['disease_name'] for metadata in query['metadatas'][0]]
        if 'metadatas' in query and query['metadatas']
        else []
    )
    sorted_results = sorted(zip(disease_ids, disease_names, distances), key=lambda x: x[2])
    pheval_disease_result_list = [PhEvalDiseaseResult(
        disease_identifier=a,
        disease_name=b,
        score=c
    ) for a, b, c in sorted_results]
    return pheval_disease_result_list # ->  make Phevaldiseaseresult

def query_disease_weighted_avg_collection(pheno_set, hp_embeddings, n_results):
    avg_embedding = calculate_average_embedding(pheno_set, hp_embeddings)
    if avg_embedding is None:
        raise ValueError("No valid embeddings found for provided HPO terms.")

    query_params = {
        "query_embeddings": [avg_embedding.tolist()],
        "include": ["metadatas", "embeddings", "distances"],
        "n_results": n_results
    }

    query_results = g.global_wgt_avg_disease_embd_collection.query(**query_params)
    sorted_results = sort_and_create_pheval_disease_results(query=query_results)
    return sorted_results

def process_wgt_tasks(args) -> List[Tuple[Any, List[PhEvalDiseaseResult]]]:
    batch, nr_of_results, processing_data = args
    hp_embeddings = processing_data
    results = []
    for orig_idx, phenotype_set in batch:
        result = query_disease_weighted_avg_collection(
            pheno_set=phenotype_set,
            hp_embeddings=hp_embeddings,
            n_results=nr_of_results,
        )
        results.append((orig_idx, result))
    return results

def process_wgt_avg_analysis_parallel(
        phenotype_sets: List[List[str]],
        owadea_analyzer,
        nr_of_results: int
) -> List[List[PhEvalDiseaseResult]]:
    processing_data = owadea_analyzer.get_parallel_processing_data()

    num_cores = mp.cpu_count()
    num_workers = min(num_cores, len(phenotype_sets))
    distributed_sets = distribute_sets_evenly(phenotype_sets, num_workers)

    process_args = [
            (worker_sets, nr_of_results, processing_data)
            for worker_sets in distributed_sets
            if worker_sets
        ]

    with mp.Pool(num_cores) as pool:
        batch_results = list(tqdm(pool.imap(process_wgt_tasks, process_args),
                                  total=len(process_args),
                                  desc="Processing phenotype sets in parallel for avg analysis"
                                  ))
        pool.close()
        pool.join()

    print(f"""
    \n           ----------               \n
    -------Finished processing {len(phenotype_sets)} phenotype sets in parallel-------
    \n           ----------               \n""")
    gc.collect()
    final_results = [[] for _ in range(len(phenotype_sets))]

    for worker_results in batch_results:
        for orig_idx, result_list in worker_results:
            final_results[orig_idx] = result_list

    return final_results

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
        if self.strategy == "avg":
            self.disease_service = DiseaseAvgEmbeddingService(data_processor=self.data_processor)
        if self.strategy == "wgt_avg":
            self.disease_weighted_service = DiseaseWeightedAvgEmbeddingService(data_processor=self.data_processor)

    def initialize_data(self):
        _ = self.data_processor.hp_embeddings
        _ = self.data_processor.disease_to_hps
        _ = self.data_processor.disease_to_hps_with_frequencies

    def setup_collections(self):
        if self.strategy == "avg":
            self.disease_service.process_data()
        if self.strategy == "wgt_avg":
            self.disease_weighted_service.process_data()
        pass

    def optimized_avg_analysis(self, phenotype_sets, nr_of_results):
        oadea = OptimizedAverageDiseaseEmbedAnalysis(
            data_processor=self.data_processor
        )
        return process_avg_analysis_parallel(phenotype_sets, oadea, nr_of_results)

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

    def optimized_wgt_avg_analysis(self, phenotype_sets, nr_of_results):
        owadea = OptimizedWeightedAverageDiseaseEmbedAnalysis(
            data_processor=self.data_processor,
        )
        return process_wgt_avg_analysis_parallel(phenotype_sets, owadea, nr_of_results)

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





class OptimizedAverageDiseaseEmbedAnalysis:
    def __init__(self, data_processor:DataProcessor):
        self.data_processor = data_processor
        self.hp_embeddings = data_processor.hp_embeddings

        import pickle

        # Test pickling hp_embeddings for multi
        try:
            pickle.dumps(self.data_processor.hp_embeddings)
            print("hp_embeddings is picklable")
        except Exception as e:
            print("hp_embeddings is not picklable:", e)

    def get_parallel_processing_data(self):
        """Return the minimal data needed for parallel processing."""
        return self.hp_embeddings

def calculate_average_embedding(query, map):
    # give dataprocessor maybe calculate_average a Callable
    return np.mean([map[i]['embeddings'] for i in query if i in map], axis=0)

def sort_and_create_pheval_disease_results(query) -> list[PhEvalDiseaseResult]:
    disease_ids = query['ids'][0] if 'ids' in query and query['ids'] else []
    distances = query['distances'][0] if 'distances' in query and query['distances'] else []
    disease_names = (
        [metadata['disease_name'] for metadata in query['metadatas'][0]]
        if 'metadatas' in query and query['metadatas']
        else []
    )
    sorted_results = sorted(zip(disease_ids, disease_names, distances), key=lambda x: x[2])
    pheval_disease_result_list = [PhEvalDiseaseResult(
        disease_identifier=a,
        disease_name=b,
        score=c
    ) for a, b, c in sorted_results]
    return pheval_disease_result_list # ->  make Phevaldiseaseresult

def query_disease_avg_collection(pheno_set, hp_embeddings, n_results):
    avg_embedding = calculate_average_embedding(pheno_set, hp_embeddings)
    if avg_embedding is None:
        raise ValueError("No valid embeddings found for provided HPO terms.")

    query_params = {
        "query_embeddings": [avg_embedding.tolist()],
        "include": ["metadatas", "embeddings", "distances"],
        "n_results": n_results
    }

    query_results = g.global_avg_disease_emb_collection.query(**query_params)
    sorted_results = sort_and_create_pheval_disease_results(query=query_results)
    return sorted_results

def process_avg_tasks(args) -> List[Tuple[Any, List[PhEvalDiseaseResult]]]:
    batch, nr_of_results, processing_data = args
    hp_embeddings = processing_data
    results = []
    for orig_idx, phenotype_set in batch:
        result = query_disease_avg_collection(
            pheno_set=phenotype_set,
            hp_embeddings=hp_embeddings,
            n_results=nr_of_results,
        )
        results.append((orig_idx, result))
    return results

def process_avg_analysis_parallel(
        phenotype_sets: List[List[str]],
        oadea_analyzer,
        nr_of_results: int
) -> List[List[PhEvalDiseaseResult]]:
    processing_data = oadea_analyzer.get_parallel_processing_data()

    num_cores = mp.cpu_count()
    num_workers = min(num_cores, len(phenotype_sets))
    distributed_sets = distribute_sets_evenly(phenotype_sets, num_workers)

    process_args = [
            (worker_sets, nr_of_results, processing_data)
            for worker_sets in distributed_sets
            if worker_sets
        ]

    with mp.Pool(num_cores) as pool:
        batch_results = list(tqdm(pool.imap(process_avg_tasks, process_args),
                                  total=len(process_args),
                                  desc="Processing phenotype sets in parallel for avg analysis"
                                  ))
        pool.close()
        pool.join()

    print(f"""
    \n           ----------               \n
    -------Finished processing {len(phenotype_sets)} phenotype sets in parallel-------
    \n           ----------               \n""")
    gc.collect()
    final_results = [[] for _ in range(len(phenotype_sets))]

    for worker_results in batch_results:
        for orig_idx, result_list in worker_results:
            final_results[orig_idx] = result_list

    return final_results