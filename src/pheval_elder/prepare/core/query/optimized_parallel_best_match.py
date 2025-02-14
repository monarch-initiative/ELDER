import gc
import os
import time
from collections import defaultdict
from itertools import islice
from multiprocessing import shared_memory
from typing import List, Iterator, TypeVar, Tuple, Dict

import numpy as np
import psutil
from flatbuffers.packer import float32
from pheval.post_processing.post_processing import PhEvalDiseaseResult
from tqdm import tqdm

worker_task_count = defaultdict(int)  # Track number of tasks per worker

def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1e9  # Convert to GB
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

    # def precompute_similarities(self, all_phenotype_sets: List[List[str]]):
    #     """Precompute all possible similarity scores between any input phenotype and disease phenotype."""
    #     print("Starting similarity precomputation...")
    #
    #     # Prepare disease phenotype embeddings  <--- check with origin below
    #     valid_disease_phenotypes = [hp for hp in all_disease_phenotypes if hp in self.hp_embeddings]
    #     disease_embeddings_matrix = np.array([
    #         self.hp_embeddings[hp]['embeddings']
    #         for hp in valid_disease_phenotypes
    #     ], dtype=np.float32)
    #
    #     # Get unique input phenotypes
    #     unique_input_hps = set()
    #     for phenotype_set in all_phenotype_sets:
    #         unique_input_hps.update(phenotype_set)
    #
    #     # Prepare input phenotype embeddings
    #     valid_input_hps = [hp for hp in unique_input_hps if hp in self.hp_embeddings]
    #     input_embeddings_matrix = np.array([
    #         self.hp_embeddings[hp]['embeddings']
    #         for hp in valid_input_hps
    #     ], dtype=np.float32)
    #
    #     print(f"Computing similarities between {len(valid_input_hps)} input phenotypes and "
    #           f"{len(valid_disease_phenotypes)} disease phenotypes...")
    #
    #     # Compute similarities
    #     norm_disease = np.linalg.norm(disease_embeddings_matrix, axis=1).reshape(1, -1)
    #     norm_input = np.linalg.norm(input_embeddings_matrix, axis=1).reshape(-1, 1)
    #     log_memory_usage("Before computing similarity matrix")
    #     similarities = np.dot(input_embeddings_matrix, disease_embeddings_matrix.T) / (norm_input @ norm_disease)
    #
    #     # Convert to shared memory
    #     shm = shared_memory.SharedMemory(create=True, size=similarities.nbytes)
    #     shared_similarities = np.ndarray(similarities.shape, dtype=np.float32, buffer=shm.buf)
    #     shared_similarities[:] = similarities[:]
    #
    #     self.shared_mem_name = shm.name  # Store shared memory name
    #     self.similarity_shape = similarities.shape
    #     self.similarity_dtype = np.float32  # Save dtype
    #     log_memory_usage("After computing similarity matrix")
    #
    #     # Create and store index mappings
    #     self.input_hp_index_map = {hp: idx for idx, hp in enumerate(valid_input_hps)}
    #     self.disease_phenotype_indices = {
    #         disease_id: np.array([
    #             idx for idx, hp in enumerate(valid_disease_phenotypes)
    #             if hp in disease_phenotypes
    #         ])
    #         for disease_id, disease_phenotypes in self.disease_phenotypes.items()
    #     }

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
            f"Computing similarities between {len(valid_input_hps)} input phenotypes and {len(valid_disease_phenotypes)} disease phenotypes...")

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
    # def get_parallel_processing_data(self):
    #     """Return the minimal data needed for parallel processing and clear metadata safely."""
    #     data = (self.input_hp_index_map, self.disease_metadata.copy(), self.disease_phenotype_indices)
    #     self.disease_metadata.clear()  # Safely clear metadata
    #     gc.collect()
    #     return data


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

def process_batch_of_sets_2_but_good_before_index_in_sets(args):
    """Process a batch of phenotype sets with minimal data."""
    pid = os.getpid()
    worker_task_count[pid] += 1

    phenotype_sets_batch, nr_of_results, processing_data = args
    input_hp_index_map, disease_metadata, all_similarities, disease_phenotype_indices = processing_data

    batch_results = []
    batch_start_time = time.time()

    for phenotype_set in phenotype_sets_batch:
        set_start_time = time.time()
        log_memory_usage(f"Before processing phenotype set in worker {pid}")

        # Get valid phenotypes and their indices
        valid_phenotypes = [hp for hp in phenotype_set if hp in input_hp_index_map]
        if not valid_phenotypes:
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
        print(
            f"Worker {pid} processed set of size {len(phenotype_set)} in {set_end_time - set_start_time:.2f} seconds")

        log_memory_usage(f"After processing phenotype set in worker {pid}")

        # Sort and get top results for this set
        sorted_scores = sorted(disease_scores, key=lambda x: x[2], reverse=True)[:nr_of_results]
        batch_results.append([
            PhEvalDiseaseResult(
                disease_identifier=disease_id,
                disease_name=disease_name,
                score=score
            )
            for disease_id, disease_name, score in sorted_scores
        ])

    batch_end_time = time.time()
    print(
        f"Worker {pid} completed batch of {len(phenotype_sets_batch)} sets in {batch_end_time - batch_start_time:.2f} seconds")
    print(f"Worker {pid} has processed {worker_task_count[pid]} total batches")

    return batch_results


def process_single_set(args):
    """Process a single phenotype set with minimal data."""

    pid = os.getpid()
    worker_task_count[pid] += 1  # Count how many times this worker runs

    log_memory_usage("Before processing phenotype set")
    phenotype_set, nr_of_results, processing_data = args
    input_hp_index_map, disease_metadata, all_similarities, disease_phenotype_indices = processing_data
    start_time = time.time()  # Track time
    # Get valid phenotypes and their indices
    valid_phenotypes = [hp for hp in phenotype_set if hp in input_hp_index_map]
    if not valid_phenotypes:
        return []

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

    end_time = time.time()
    print(f"Worker {os.getpid()} processed set in {end_time - start_time:.2f} seconds")

    log_memory_usage("After processing phenotype set")

    # Sort and return top results
    sorted_scores = sorted(disease_scores, key=lambda x: x[2], reverse=True)[:nr_of_results]
    print(f"Worker {pid} processed {worker_task_count[pid]} tasks")
    return [
        PhEvalDiseaseResult(
            disease_identifier=disease_id,
            disease_name=disease_name,
            score=score
        )
        for disease_id, disease_name, score in sorted_scores
    ]


def process_phenotype_sets_parallel(
        phenotype_sets: List[List[str]],
        tcp_analyzer,
        nr_of_results: int
) -> List[List[PhEvalDiseaseResult]]:
    """Process multiple phenotype sets in parallel."""
    import multiprocessing as mp
    # Get the minimal data needed for processing
    processing_data = tcp_analyzer.get_parallel_processing_data()

    # Determine number of workers
    num_cores = mp.cpu_count()
    num_workers = min(num_cores, len(phenotype_sets))
    print(f"Using {num_workers} CPU cores for parallel processing")

    # Distribute sets evenly across workers
    distributed_sets = distribute_sets_evenly(phenotype_sets, num_workers)

    # Prepare arguments for each worker
    process_args = [
        (worker_sets, nr_of_results, processing_data)
        for worker_sets in distributed_sets
        if worker_sets  # Only include non-empty worker sets
    ]

    with mp.Pool(num_workers) as pool:
        batch_results = list(tqdm(
            pool.imap(process_batch_of_sets, process_args),
            total=num_workers,
            desc="Processing phenotype sets in parallel"
        ))
        pool.close()
        pool.join()

    print(f"""\n           ----------               \n
-------Finished processing {len(phenotype_sets)} phenotype sets in parallel-------
\n           ----------               \n""")
    gc.collect()

    # Initialize results list with empty lists
    final_results = [[] for _ in range(len(phenotype_sets))]

    # Place results in their original positions
    for worker_results in batch_results:
        for orig_idx, result_list in worker_results:
            final_results[orig_idx] = result_list

    return final_results

def process_phenotype_sets_parallel_2_but_good_before_index_in_set(phenotype_sets: List[List[str]], tcp_analyzer, nr_of_results: int, batch_size: int = 1) -> List[
    List[PhEvalDiseaseResult]]:
    """Process multiple phenotype sets in parallel."""
    import multiprocessing as mp
    from functools import partial

    # Get the minimal data needed for processing
    processing_data = tcp_analyzer.get_parallel_processing_data()

    # -------_---------
    # Determine number of workers
    num_cores = mp.cpu_count()
    # num_workers = min(num_cores, (len(phenotype_sets) + batch_size - 1) // batch_size)
    print(len(phenotype_sets))
    num_workers = min(num_cores, len(phenotype_sets))
    print(f"Using {num_workers} CPU cores for parallel processing")

    # Distribute sets evenly across workers
    distributed_sets = distribute_sets_evenly(phenotype_sets, num_workers)

    # Prepare arguments for each worker
    process_args = [
        (worker_sets, nr_of_results, processing_data)
        for worker_sets in distributed_sets
        if worker_sets  # Only include non-empty worker sets
    ]
    # -------_---------

    # Prepare arguments for parallel processing
    # process_args = [(phenotype_set, nr_of_results, processing_data) for phenotype_set in phenotype_sets]

    # num_cores = mp.cpu_count()
    # print(f"Using {num_cores} CPU cores for parallel processing")

    with mp.Pool(num_cores) as pool:
        batch_results = list(tqdm(
            pool.imap(process_batch_of_sets, process_args),
            total=num_cores,
            desc="Processing phenotype sets in parallel"
        ))
        pool.close()
        pool.join()
    # del results  # Free up memory
    print(f"""\n           ----------               \n
-------Del results, Finished processing {len(phenotype_sets)} phenotype sets in parallel-------
\n           ----------               \n""")
    gc.collect()
    flattened_results = []
    for worker_results in batch_results:
        for result_list in worker_results:
            # Each result_list should be a list containing a single PhEvalDiseaseResult
            if result_list:  # Only add non-empty results
                flattened_results.extend(result_list)

    return flattened_results
    # return results

# def process_phenotype_sets_parallel(
#         self,
#         phenotype_sets: List[List[str]],
#         nr_of_results: int,
#         batch_size: int = 28  # Number of sets per worker
# ) -> List[List[PhEvalDiseaseResult]]:
#     """
#     Process multiple phenotype sets in parallel with balanced workload distribution
#     using shared memory for similarity matrix.
#     """
#     import multiprocessing as mp
#
#     # Determine number of workers
#     num_cores = mp.cpu_count()
#     num_workers = min(num_cores, (len(phenotype_sets) + batch_size - 1) // batch_size)
#     print(f"Using {num_workers} CPU cores for parallel processing")
#
#     # Distribute sets evenly across workers
#     distributed_sets = distribute_sets_evenly(phenotype_sets, num_workers)
#
#     # Prepare processing data (minimal data needed for processing)
#     processing_data = {
#         'input_hp_index_map': self.input_hp_index_map,
#         'disease_metadata': self.disease_metadata,
#         'disease_phenotype_indices': self.disease_phenotype_indices
#     }
#
#     # Prepare arguments for each worker
#     process_args = [
#         (worker_sets, nr_of_results, processing_data, self.shared_mem_name, self.similarity_shape)
#         for worker_sets in distributed_sets
#         if worker_sets  # Only include non-empty worker sets
#     ]
#
#     # Process in parallel
#     with mp.Pool(num_workers) as pool:
#         results = list(tqdm(
#             pool.imap(process_single_set, process_args),
#             total=len(process_args),
#             desc="Processing phenotype sets in parallel"
#         ))
#         pool.close()
#         pool.join()
#
#     gc.collect()
#
#     # Flatten results
#     return [result for sublist in results for result in sublist]

T = TypeVar('T')

def batch(iterable: Iterator[T], n: int = 5) -> Iterator[List[T]]:
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def distribute_sets_evenly(phenotype_sets: List[List[str]], num_workers: int) -> List[List[Tuple[int, List[str]]]]:
    """
    Distribute phenotype sets across workers to balance total workload.
    Returns a list where each inner list contains tuples of (original_index, phenotype_set).
    """
    # Create list of (index, set, size) tuples
    indexed_sets = [(i, s, len(s)) for i, s in enumerate(phenotype_sets)]

    # Sort by size in descending order, keeping original indices
    sorted_sets = sorted(indexed_sets, key=lambda x: x[2], reverse=True)

    # Initialize workers with empty lists
    workers = [[] for _ in range(num_workers)]
    worker_loads = [0] * num_workers

    # Distribute sets using a greedy approach
    for orig_idx, phenotype_set, size in sorted_sets:
        # Find worker with minimum current load
        min_load_worker = min(range(num_workers), key=lambda i: worker_loads[i])

        # Assign set to that worker (store only index and set, not size)
        workers[min_load_worker].append((orig_idx, phenotype_set))
        worker_loads[min_load_worker] += size

    # Print distribution statistics
    print("\nWorkload distribution across workers:")
    for i, worker_sets in enumerate(workers):
        total_size = sum(len(s) for _, s in worker_sets)
        sizes = [len(s) for _, s in worker_sets]
        print(f"Worker {i}: {len(worker_sets)} sets, total size: {total_size}, "
              f"sizes: {sizes}")

    return workers


def distribute_sets_evenly_2_good_before_index_in_set(phenotype_sets: List[List[str]], num_workers: int) -> List[List[List[str]]]:
    """
    Distribute phenotype sets across workers to ensure each worker gets roughly the same number of sets.
    If the division isn't even, some workers will get one extra set.

    Args:
        phenotype_sets: List of phenotype sets to distribute
        num_workers: Number of available workers

    Returns:
        List where each inner list contains the sets for one worker
    """
    total_sets = len(phenotype_sets)
    base_sets_per_worker = total_sets // num_workers  # Integer division
    remaining_sets = total_sets % num_workers

    # Sort sets by size in descending order to try to balance workload
    sorted_sets = sorted(phenotype_sets, key=len, reverse=True)

    # Initialize empty lists for each worker
    workers = [[] for _ in range(num_workers)]
    current_set_idx = 0

    # Distribute base sets to all workers
    for worker_idx in range(num_workers):
        for _ in range(base_sets_per_worker):
            workers[worker_idx].append(sorted_sets[current_set_idx])
            current_set_idx += 1

    # Distribute remaining sets (if any) to the first 'remaining_sets' workers
    for worker_idx in range(remaining_sets):
        workers[worker_idx].append(sorted_sets[current_set_idx])
        current_set_idx += 1

    # Print distribution statistics
    print("\nWorkload distribution across workers:")
    for i, worker_sets in enumerate(workers):
        total_size = sum(len(s) for s in worker_sets)
        print(f"Worker {i}: {len(worker_sets)} sets, total size: {total_size}, "
              f"sizes: {[len(s) for s in worker_sets]}")

    return workers


def distribute_sets_evenly_former(phenotype_sets: List[List[str]], num_workers: int) -> List[List[List[str]]]:
    """
    Distribute phenotype sets across workers to balance total workload.
    Returns a list where each inner list represents the sets assigned to one worker.
    """
    # Sort sets by size in descending order
    sorted_sets = sorted(phenotype_sets, key=len, reverse=True)

    # Initialize workers with empty lists and their current total workload
    workers = [[] for _ in range(num_workers)]
    worker_loads = [0] * num_workers

    # Distribute sets using a greedy approach
    for phenotype_set in sorted_sets:
        # Find worker with minimum current load
        min_load_worker = min(range(num_workers), key=lambda i: worker_loads[i])

        # Assign set to that worker
        workers[min_load_worker].append(phenotype_set)
        worker_loads[min_load_worker] += len(phenotype_set)

    # Print distribution statistics
    print("\nWorkload distribution across workers:")
    for i, (worker_sets, total_load) in enumerate(zip(workers, worker_loads)):
        print(f"Worker {i}: {len(worker_sets)} sets, total size: {total_load}, "
              f"sizes: {[len(s) for s in worker_sets]}")

    return workers


# def process_batch_of_sets(args: Tuple[List[List[str]], int, Dict, str, Tuple[int, int]]) -> List[
#     List[PhEvalDiseaseResult]]:
#     """
#     Process a batch of phenotype sets with shared similarity matrix.
#
#     Args:
#         args: Tuple containing:
#             - phenotype_set_batch: List of phenotype sets to process
#             - nr_of_results: Number of top results to return
#             - processing_data: Dict containing input_hp_index_map, disease_metadata, disease_phenotype_indices
#             - shared_mem_name: Name of shared memory block containing similarity matrix
#             - shape: Shape of the similarity matrix
#     """
#     phenotype_set_batch, nr_of_results, processing_data, shared_mem_name, shape = args
#     input_hp_index_map, disease_metadata, disease_phenotype_indices = processing_data
#
#     # Attach to shared memory
#     existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
#     all_similarities = np.ndarray(shape, dtype=np.float32, buffer=existing_shm.buf)
#
#     results = []
#     for phenotype_set in phenotype_set_batch:
#         valid_phenotypes = [hp for hp in phenotype_set if hp in input_hp_index_map]
#         if not valid_phenotypes:
#             continue
#
#         phenotype_indices = np.array([input_hp_index_map[hp] for hp in valid_phenotypes])
#         disease_scores = []
#
#         for disease_id, indices in disease_phenotype_indices.items():
#             if indices.size == 0:
#                 continue
#
#             relevant_similarities = all_similarities[phenotype_indices][:, indices]
#             max_similarities = np.max(relevant_similarities, axis=1)
#             avg_score = float(np.mean(max_similarities))
#
#             disease_scores.append((
#                 disease_id,
#                 disease_metadata[disease_id]["name"],
#                 avg_score
#             ))
#
#         sorted_scores = sorted(disease_scores, key=lambda x: x[2], reverse=True)[:nr_of_results]
#         results.append([
#             PhEvalDiseaseResult(
#                 disease_identifier=disease_id,
#                 disease_name=disease_name,
#                 score=score
#             )
#             for disease_id, disease_name, score in sorted_scores
#         ])
#
#     # Clean up shared memory attachment
#     existing_shm.close()
#     return results