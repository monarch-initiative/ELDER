import gc
import os
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import psutil
from pheval.post_processing.post_processing import PhEvalDiseaseResult
from tqdm import tqdm

worker_task_count = defaultdict(int)  # Track number of tasks per worker

def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1e9  # to GB
    print(f"[{stage}] Memory Usage: {mem:.3f} GB (Process {os.getpid()})")


class OptimizedTermSetPairwiseComparison:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        log_memory_usage("Before loading embeddings")
        self.hp_embeddings = data_processor.hp_embeddings
        log_memory_usage("After loading embeddings")
        self.disease_to_hps_from_omim = data_processor.disease_to_hps
        self.input_hp_index_map = {}
        self.disease_metadata = {}
        self.all_similarities = None
        self.disease_phenotype_indices = {}


    def precompute_similarities(self, all_phenotype_sets: List[List[str]]):
        """Precompute all possible similarity scores between any input phenotype and disease phenotype."""
        print("Starting similarity precomputation...")

        # Get all disease phenotypes and build index maps
        all_disease_phenotypes = set()
        for disease_id, disease_data in self.disease_to_hps_from_omim.items():
            all_disease_phenotypes.update(disease_data["phenotypes"])
            self.disease_metadata[disease_id] = {
                "name": disease_data["disease_name"],
                "phenotypes": set(disease_data["phenotypes"])
            }

        valid_disease_phenotypes = [hp for hp in all_disease_phenotypes if hp in self.hp_embeddings]
        disease_embeddings_matrix = np.array([
            self.hp_embeddings[hp]['embeddings']
            for hp in valid_disease_phenotypes
        ])

        # Get all unique input phenotypes and build index map
        unique_input_hps = set()
        for hp_set in all_phenotype_sets:
            unique_input_hps.update(hp_set)
        valid_input_hps = [hp for hp in unique_input_hps if hp in self.hp_embeddings]

        # Build input phenotype index map
        for idx, hp in enumerate(valid_input_hps):
            self.input_hp_index_map[hp] = idx

        input_embeddings_matrix = np.array([
            self.hp_embeddings[hp]['embeddings']
            for hp in valid_input_hps
        ])

        print(
            f"Computing similarities between {len(valid_input_hps)} \
            input phenotypes and {len(valid_disease_phenotypes)} "
            f"disease phenotypes..."
        )

        # Compute all pairwise similarities at once
        norm_disease = np.linalg.norm(disease_embeddings_matrix, axis=1).reshape(1, -1)
        norm_input = np.linalg.norm(input_embeddings_matrix, axis=1).reshape(-1, 1)
        log_memory_usage("Before computing similarity matrix")
        self.all_similarities = np.dot(input_embeddings_matrix, disease_embeddings_matrix.T) / (
                    norm_input @ norm_disease)
        log_memory_usage("After computing similarity matrix")

        # Pre-calculate disease-phenotype indices
        for disease_id, disease_data in self.disease_to_hps_from_omim.items():
            phenotype_indices = [i for i, hp in enumerate(valid_disease_phenotypes)
                                 if hp in disease_data["phenotypes"]]
            if phenotype_indices:
                self.disease_phenotype_indices[disease_id] = np.array(phenotype_indices)

        print("Precomputation complete!")

    def get_parallel_processing_data(self):
        """Return the minimal data needed for parallel processing."""
        return (self.input_hp_index_map,
                self.disease_metadata,
                self.all_similarities,
                self.disease_phenotype_indices)


def process_batch_of_sets(args):
    """Process a batch of phenotype sets with minimal data."""
    pid = os.getpid()
    worker_task_count[pid] += 1

    indexed_phenotype_sets_batch, nr_of_results, processing_data = args
    input_hp_index_map, disease_metadata, all_similarities, disease_phenotype_indices = processing_data

    batch_results = []
    batch_start_time = time.time()

    for orig_idx, phenotype_set in indexed_phenotype_sets_batch:
        set_start_time = time.time()
        log_memory_usage(f"Before processing phenotype set in worker {pid}")

        # Get valid phenotypes and their indices
        valid_phenotypes = [hp for hp in phenotype_set if hp in input_hp_index_map]
        if not valid_phenotypes:
            batch_results.append((orig_idx, []))  # Empty result for this set
            continue

        phenotype_indices = np.array([input_hp_index_map[hp] for hp in valid_phenotypes])
        disease_scores = []

        # Calculate scores for all diseases
        for disease_id, indices in disease_phenotype_indices.items():
            if indices.size == 0:
                continue

            relevant_similarities = all_similarities[phenotype_indices][:, indices]
            max_similarities = np.max(relevant_similarities, axis=1)
            avg_score = float(np.mean(max_similarities))

            disease_scores.append((
                disease_id,
                disease_metadata[disease_id]["name"],
                avg_score
            ))

        set_end_time = time.time()
        print(f"Worker {pid} processed set of size {len(phenotype_set)} in {set_end_time - set_start_time:.2f} seconds")

        log_memory_usage(f"After processing phenotype set in worker {pid}")

        # Sort and get top results for this set
        sorted_scores = sorted(disease_scores, key=lambda x: x[2], reverse=True)[:nr_of_results]
        batch_results.append((
            orig_idx,
            [
                PhEvalDiseaseResult(
                    disease_identifier=disease_id,
                    disease_name=disease_name,
                    score=score
                )
                for disease_id, disease_name, score in sorted_scores
            ]
        ))

    batch_end_time = time.time()
    print(
        f"Worker {pid} completed batch of {len(indexed_phenotype_sets_batch)} sets in {batch_end_time - batch_start_time:.2f} seconds")
    print(f"Worker {pid} has processed {worker_task_count[pid]} total batches")

    return batch_results


def process_phenotype_sets_parallel(
        phenotype_sets: List[List[str]],
        tcp_analyzer,
        nr_of_results: int
) -> List[List[PhEvalDiseaseResult]]:
    """Process multiple phenotype sets in parallel."""
    import multiprocessing as mp
    processing_data = tcp_analyzer.get_parallel_processing_data()

    num_cores = mp.cpu_count()
    num_workers = min(num_cores, len(phenotype_sets))
    print(f"Using {num_workers} CPU cores for parallel processing")

    distributed_sets = distribute_sets_evenly(phenotype_sets, num_workers)

    process_args = [
        (worker_sets, nr_of_results, processing_data)
        for worker_sets in distributed_sets
        if worker_sets
    ]

    with mp.Pool(num_workers) as pool:
        batch_results = list(tqdm(
            pool.imap(process_batch_of_sets, process_args),
            total=num_workers,
            desc="Processing phenotype sets in parallel"
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


def distribute_sets_evenly(phenotype_sets: List[List[str]], num_workers: int) -> List[List[Tuple[int, List[str]]]]:
    """
    Distribute phenotype sets across workers to balance total workload.
    Returns a list where each inner list contains tuples of (original_index, phenotype_set).
    """

    indexed_sets = [(i, s, len(s)) for i, s in enumerate(phenotype_sets)]
    sorted_sets = sorted(indexed_sets, key=lambda x: x[2], reverse=True)

    workers = [[] for _ in range(num_workers)]
    worker_loads = [0] * num_workers

    for orig_idx, phenotype_set, size in sorted_sets:
        min_load_worker = min(range(num_workers), key=lambda i: worker_loads[i])
        workers[min_load_worker].append((orig_idx, phenotype_set))
        worker_loads[min_load_worker] += size
    # debugging
    print("\nWorkload distribution across workers:")
    for i, worker_sets in enumerate(workers):
        total_size = sum(len(s) for _, s in worker_sets)
        sizes = [len(s) for _, s in worker_sets]
        print(f"Worker {i}: {len(worker_sets)} sets, total size: {total_size}, "
              f"sizes: {sizes}")

    return workers



# """
# Multiprocessing module for best match (term-set pairwise comparison) analysis.
#
# This module contains code for parallel processing of phenotype sets
# using the best match strategy.
# """
#
# import multiprocessing as mp
# import time
# from typing import List, Dict, Any, Tuple, Set, Optional
# import numpy as np
# from pheval.post_processing.post_processing import PhEvalDiseaseResult
# from tqdm import tqdm
#
# from pheval_elder.prepare.core.data_processing.data_processor import DataProcessor
#
#
# def distribute_sets_evenly(phenotype_sets, num_workers):
#     """
#     Distribute phenotype sets evenly among workers.
#
#     Args:
#         phenotype_sets: List of phenotype sets to distribute
#         num_workers: Number of workers to distribute to
#
#     Returns:
#         List of lists of (index, phenotype_set) pairs
#     """
#     result = [[] for _ in range(num_workers)]
#     for i, phenotype_set in enumerate(phenotype_sets):
#         worker_index = i % num_workers
#         result[worker_index].append((i, phenotype_set))
#     return result
#
#
# class OptimizedTermSetPairwiseComparison:
#     """
#     Optimizer for term-set pairwise comparison analysis.
#
#     This class precomputes similarities and runs term-set pairwise comparison
#     analysis in parallel.
#     """
#
#     def __init__(self, data_processor: DataProcessor):
#         """Initialize the optimizer with a data processor."""
#         self.data_processor = data_processor
#         self.hp_embeddings = data_processor.hp_embeddings
#         self.disease_to_hps = data_processor.disease_to_hps
#         self.similarities = {}
#         self.disease_metadata = {}
#         for disease_id, disease_data in self.disease_to_hps.items():
#             self.disease_metadata[disease_id] = disease_data.get("disease_name", disease_id)
#
#     def precompute_similarities(self, phenotype_sets: List[List[str]]) -> None:
#         """
#         Precompute similarities between phenotype terms and diseases.
#
#         Args:
#             phenotype_sets: List of phenotype sets (lists of HPO IDs)
#         """
#         start_time = time.time()
#         print("Precomputing similarities...")
#
#         all_terms = set()
#         for phenotype_set in phenotype_sets:
#             all_terms.update(phenotype_set)
#
#         all_diseases = set(self.disease_to_hps.keys())
#
#         for term in tqdm(all_terms, desc="Precomputing term similarities"):
#             if term not in self.hp_embeddings:
#                 continue
#
#             term_embedding = self.hp_embeddings[term]['embeddings']
#             self.similarities[term] = {}
#
#             for disease_id in all_diseases:
#                 hps = self.disease_to_hps.get(disease_id, [])
#                 best_similarity = 0.0
#
#                 for hp in hps:
#                     if hp not in self.hp_embeddings:
#                         continue
#
#                     hp_embedding = self.hp_embeddings[hp]['embeddings']
#                     similarity = self.cosine_similarity(term_embedding, hp_embedding)
#                     best_similarity = max(best_similarity, similarity)
#
#                 self.similarities[term][disease_id] = best_similarity
#
#         end_time = time.time()
#         print(f"Precomputing similarities took {end_time - start_time:.2f} seconds")
#
#     def get_parallel_processing_data(self) -> Dict:
#         """Return the data needed for parallel processing."""
#         return {
#             'similarities': self.similarities,
#             'disease_to_hps': self.disease_to_hps,
#             'disease_metadata': self.disease_metadata
#         }
#
#     @staticmethod
#     def cosine_similarity(embedding1, embedding2):
#         """
#         Calculate cosine similarity between two embeddings.
#
#         Args:
#             embedding1: First embedding
#             embedding2: Second embedding
#
#         Returns:
#             Cosine similarity value
#         """
#         dot_product = np.dot(embedding1, embedding2)
#         norm1 = np.linalg.norm(embedding1)
#         norm2 = np.linalg.norm(embedding2)
#
#         if norm1 == 0 or norm2 == 0:
#             return 0.0
#
#         return dot_product / (norm1 * norm2)
#
#
# def process_phenotype_set(phenotype_set, processing_data, n_results):
#     """
#     Process a single phenotype set.
#
#     Args:
#         phenotype_set: List of HPO IDs
#         processing_data: Data needed for processing
#         n_results: Number of results to return
#
#     Returns:
#         List of PhEvalDiseaseResult objects
#     """
#     similarities = processing_data['similarities']
#     disease_to_hps = processing_data['disease_to_hps']
#     disease_metadata = processing_data['disease_metadata']
#
#     disease_scores = {}
#     for disease_id in disease_to_hps.keys():
#         total_score = 0.0
#         count = 0
#
#         for term in phenotype_set:
#             if term in similarities and disease_id in similarities[term]:
#                 total_score += similarities[term][disease_id]
#                 count += 1
#
#         if count > 0:
#             disease_scores[disease_id] = total_score / count
#
#     # Sort diseases by score
#     sorted_diseases = sorted(
#         disease_scores.items(),
#         key=lambda x: x[1],
#         reverse=True
#     )[:n_results]
#
#     # Create PhEvalDiseaseResult
#     results = []
#     for disease_id, score in sorted_diseases:
#         disease_name = disease_metadata.get(disease_id, disease_id)
#         results.append(PhEvalDiseaseResult(
#             disease_identifier=disease_id,
#             disease_name=disease_name,  # We don't have disease names here
#             score=score
#         ))
#
#     return results
#
#
# def process_batch(args):
#     """
#     Process a batch of phenotype sets.
#
#     Args:
#         args: Tuple of (batch, n_results, processing_data)
#
#     Returns:
#         List of (index, results) pairs
#     """
#     batch, n_results, processing_data = args
#     results = []
#
#     for orig_idx, phenotype_set in batch:
#         result = process_phenotype_set(phenotype_set, processing_data, n_results)
#         results.append((orig_idx, result))
#
#     return results
#
#
# def process_phenotype_sets_parallel(
#     phenotype_sets: List[List[str]],
#     optimizer,
#     n_results: int
# ) -> List[List[PhEvalDiseaseResult]]:
#     """
#     Process phenotype sets in parallel.
#
#     Args:
#         phenotype_sets: List of phenotype sets (lists of HPO IDs)
#         optimizer: OptimizedTermSetPairwiseComparison instance
#         n_results: Number of results to return
#
#     Returns:
#         List of lists of PhEvalDiseaseResult objects
#     """
#     processing_data = optimizer.get_parallel_processing_data()
#
#     # Determine the number of workers
#     num_cores = mp.cpu_count()
#     num_workers = min(num_cores, len(phenotype_sets))
#
#     # Distribute phenotype sets among workers
#     distributed_sets = distribute_sets_evenly(phenotype_sets, num_workers)
#
#     # Create arguments for each worker
#     process_args = [
#         (worker_sets, n_results, processing_data)
#         for worker_sets in distributed_sets
#         if worker_sets
#     ]
#
#     # Process in parallel
#     with mp.Pool(num_cores) as pool:
#         batch_results = list(tqdm(
#             pool.imap(process_batch, process_args),
#             total=len(process_args),
#             desc="Processing phenotype sets in parallel for TCP analysis"
#         ))
#         pool.close()
#         pool.join()
#
#     # Collect results
#     final_results = [[] for _ in range(len(phenotype_sets))]
#     for worker_results in batch_results:
#         for orig_idx, result_list in worker_results:
#             final_results[orig_idx] = result_list
#
#     return final_results