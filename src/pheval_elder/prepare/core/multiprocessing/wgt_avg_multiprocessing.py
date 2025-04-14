"""
Multiprocessing module for weighted average embedding analysis.

This module contains code for parallel processing of phenotype sets
using the weighted average embedding strategy.
"""

import gc
from typing import List, Tuple, Any
import multiprocessing as mp
import numpy as np
from pheval.post_processing.post_processing import PhEvalDiseaseResult
from tqdm import tqdm
import chromadb
from pheval_elder.prepare.core.data_processing.data_processor import DataProcessor
from pheval_elder.prepare.core.query.optimized_parallel_best_match import distribute_sets_evenly


class OptimizedWeightedAverageDiseaseEmbedAnalysis:
    """Analyzer for weighted average disease embedding analysis."""
    
    def __init__(self, data_processor: DataProcessor):
        """Initialize the analyzer with a data processor."""
        self.data_processor = data_processor
        self.hp_embeddings = data_processor.hp_embeddings
        
    def get_parallel_processing_data(self):
        """Return the data needed for parallel processing."""
        return self.hp_embeddings


def calculate_average_embedding(query, embedding_map):
    """Calculate the average embedding for a set of query terms."""
    try:
        embeddings = [embedding_map[i]['embeddings'] for i in query if i in embedding_map]
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)
    except Exception as e:
        print(f"Error calculating average embedding: {e}")
        return None


def sort_and_create_pheval_disease_results(query) -> List[PhEvalDiseaseResult]:
    """Create PhEvalDiseaseResult objects from query results."""
    disease_ids = query['ids'][0] if 'ids' in query and query['ids'] else []
    distances = query['distances'][0] if 'distances' in query and query['distances'] else []
    disease_names = (
        [metadata['disease_name'] for metadata in query['metadatas'][0]]
        if 'metadatas' in query and query['metadatas']
        else []
    )
    sorted_results = sorted(zip(disease_ids, disease_names, distances), key=lambda x: x[2])
    return [
        PhEvalDiseaseResult(
            disease_identifier=a,
            disease_name=b,
            score=c
        ) for a, b, c in sorted_results
    ]


def query_disease_weighted_avg_collection(pheno_set, hp_embeddings, n_results, collection):
    """Query the disease weighted average collection for a phenotype set."""
    avg_embedding = calculate_average_embedding(pheno_set, hp_embeddings)
    if avg_embedding is None:
        raise ValueError("No valid embeddings found for provided HPO terms.")

    query_params = {
        "query_embeddings": [avg_embedding.tolist()],
        "include": ["metadatas", "embeddings", "distances"],
        "n_results": n_results
    }

    query_results = collection.query(**query_params)
    return sort_and_create_pheval_disease_results(query=query_results)


def process_wgt_tasks(args) -> List[Tuple[Any, List[PhEvalDiseaseResult]]]:
    """Process a batch of phenotype sets for weighted average embedding analysis."""
    batch, nr_of_results, processing_data, db_path, collection_name = args
    hp_embeddings = processing_data
    results = []
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    for orig_idx, phenotype_set in batch:
        result = query_disease_weighted_avg_collection(
            pheno_set=phenotype_set,
            hp_embeddings=hp_embeddings,
            n_results=nr_of_results,
            collection=collection
        )
        results.append((orig_idx, result))
    return results


def process_wgt_avg_analysis_parallel(
        phenotype_sets: List[List[str]],
        owadea_analyzer,
        nr_of_results: int
) -> List[List[PhEvalDiseaseResult]]:
    """
    Process phenotype sets in parallel for weighted average embedding analysis.
    
    Args:
        phenotype_sets: List of phenotype sets (lists of HPO IDs)
        owadea_analyzer: Analyzer for weighted average disease embedding analysis
        nr_of_results: Number of results to return
        
    Returns:
        List of lists of PhEvalDiseaseResult objects, one list per phenotype set
    """
    processing_data = owadea_analyzer.get_parallel_processing_data()

    num_cores = mp.cpu_count()
    num_workers = min(num_cores, len(phenotype_sets))
    
    distributed_sets = distribute_sets_evenly(phenotype_sets, num_workers)

    db_path = owadea_analyzer.data_processor.db_manager.path
    collection_name = owadea_analyzer.data_processor.db_manager.disease_weighted_avg_embeddings_collection.name
    process_args = [
            (worker_sets, nr_of_results, processing_data, db_path, collection_name)
            for worker_sets in distributed_sets
            if worker_sets
        ]

    with mp.Pool(num_cores) as pool:
        batch_results = list(tqdm(
            pool.imap(process_wgt_tasks, process_args),
            total=len(process_args),
            desc="Processing phenotype sets in parallel for wgt avg analysis"
        ))
        pool.close()
        pool.join()

    print(f"\n----------\nFinished processing {len(phenotype_sets)} phenotype sets in parallel\n----------\n")
    gc.collect()
    
    final_results = [[] for _ in range(len(phenotype_sets))]
    for worker_results in batch_results:
        for orig_idx, result_list in worker_results:
            final_results[orig_idx] = result_list

    return final_results