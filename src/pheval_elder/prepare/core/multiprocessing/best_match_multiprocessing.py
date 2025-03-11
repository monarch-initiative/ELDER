"""
Multiprocessing module for best match (term-set pairwise comparison) analysis.

This module contains code for parallel processing of phenotype sets
using the best match strategy.
"""

import multiprocessing as mp
import time
from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np
from pheval.post_processing.post_processing import PhEvalDiseaseResult
from tqdm import tqdm

from pheval_elder.prepare.core.data_processing.data_processor import DataProcessor


def distribute_sets_evenly(phenotype_sets, num_workers):
    """
    Distribute phenotype sets evenly among workers.
    
    Args:
        phenotype_sets: List of phenotype sets to distribute
        num_workers: Number of workers to distribute to
        
    Returns:
        List of lists of (index, phenotype_set) pairs
    """
    result = [[] for _ in range(num_workers)]
    for i, phenotype_set in enumerate(phenotype_sets):
        worker_index = i % num_workers
        result[worker_index].append((i, phenotype_set))
    return result


class OptimizedTermSetPairwiseComparison:
    """
    Optimizer for term-set pairwise comparison analysis.
    
    This class precomputes similarities and runs term-set pairwise comparison
    analysis in parallel.
    """
    
    def __init__(self, data_processor: DataProcessor):
        """Initialize the optimizer with a data processor."""
        self.data_processor = data_processor
        self.hp_embeddings = data_processor.hp_embeddings
        self.disease_to_hps = data_processor.disease_to_hps
        self.similarities = {}
        
    def precompute_similarities(self, phenotype_sets: List[List[str]]) -> None:
        """
        Precompute similarities between phenotype terms and diseases.
        
        Args:
            phenotype_sets: List of phenotype sets (lists of HPO IDs)
        """
        start_time = time.time()
        print("Precomputing similarities...")
        
        # Collect all unique phenotype terms
        all_terms = set()
        for phenotype_set in phenotype_sets:
            all_terms.update(phenotype_set)
            
        # Collect all unique disease terms
        all_diseases = set(self.disease_to_hps.keys())
        
        # Precompute similarities
        for term in tqdm(all_terms, desc="Precomputing term similarities"):
            if term not in self.hp_embeddings:
                continue
                
            term_embedding = self.hp_embeddings[term]['embeddings']
            self.similarities[term] = {}
            
            for disease_id in all_diseases:
                hps = self.disease_to_hps.get(disease_id, [])
                best_similarity = 0.0
                
                for hp in hps:
                    if hp not in self.hp_embeddings:
                        continue
                        
                    hp_embedding = self.hp_embeddings[hp]['embeddings']
                    similarity = self.cosine_similarity(term_embedding, hp_embedding)
                    best_similarity = max(best_similarity, similarity)
                    
                self.similarities[term][disease_id] = best_similarity
                
        end_time = time.time()
        print(f"Precomputing similarities took {end_time - start_time:.2f} seconds")
        
    def get_parallel_processing_data(self) -> Dict:
        """Return the data needed for parallel processing."""
        return {
            'similarities': self.similarities,
            'disease_to_hps': self.disease_to_hps
        }
        
    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity value
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        

def process_phenotype_set(phenotype_set, processing_data, n_results):
    """
    Process a single phenotype set.
    
    Args:
        phenotype_set: List of HPO IDs
        processing_data: Data needed for processing
        n_results: Number of results to return
        
    Returns:
        List of PhEvalDiseaseResult objects
    """
    similarities = processing_data['similarities']
    disease_to_hps = processing_data['disease_to_hps']
    
    # Calculate score for each disease
    disease_scores = {}
    for disease_id in disease_to_hps.keys():
        total_score = 0.0
        count = 0
        
        for term in phenotype_set:
            if term in similarities and disease_id in similarities[term]:
                total_score += similarities[term][disease_id]
                count += 1
                
        if count > 0:
            disease_scores[disease_id] = total_score / count
    
    # Sort diseases by score
    sorted_diseases = sorted(
        disease_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:n_results]
    
    # Create PhEvalDiseaseResult objects
    results = []
    for disease_id, score in sorted_diseases:
        results.append(PhEvalDiseaseResult(
            disease_identifier=disease_id,
            disease_name=disease_id,  # We don't have disease names here
            score=score
        ))
        
    return results


def process_batch(args):
    """
    Process a batch of phenotype sets.
    
    Args:
        args: Tuple of (batch, n_results, processing_data)
        
    Returns:
        List of (index, results) pairs
    """
    batch, n_results, processing_data = args
    results = []
    
    for orig_idx, phenotype_set in batch:
        result = process_phenotype_set(phenotype_set, processing_data, n_results)
        results.append((orig_idx, result))
        
    return results


def process_phenotype_sets_parallel(
    phenotype_sets: List[List[str]],
    optimizer,
    n_results: int
) -> List[List[PhEvalDiseaseResult]]:
    """
    Process phenotype sets in parallel.
    
    Args:
        phenotype_sets: List of phenotype sets (lists of HPO IDs)
        optimizer: OptimizedTermSetPairwiseComparison instance
        n_results: Number of results to return
        
    Returns:
        List of lists of PhEvalDiseaseResult objects
    """
    processing_data = optimizer.get_parallel_processing_data()
    
    # Determine the number of workers
    num_cores = mp.cpu_count()
    num_workers = min(num_cores, len(phenotype_sets))
    
    # Distribute phenotype sets among workers
    distributed_sets = distribute_sets_evenly(phenotype_sets, num_workers)
    
    # Create arguments for each worker
    process_args = [
        (worker_sets, n_results, processing_data)
        for worker_sets in distributed_sets
        if worker_sets
    ]
    
    # Process in parallel
    with mp.Pool(num_cores) as pool:
        batch_results = list(tqdm(
            pool.imap(process_batch, process_args),
            total=len(process_args),
            desc="Processing phenotype sets in parallel for TCP analysis"
        ))
        pool.close()
        pool.join()
    
    # Collect results
    final_results = [[] for _ in range(len(phenotype_sets))]
    for worker_results in batch_results:
        for orig_idx, result_list in worker_results:
            final_results[orig_idx] = result_list
            
    return final_results