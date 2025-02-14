
from pheval.post_processing.post_processing import PhEvalDiseaseResult

import multiprocessing as mp
from typing import List, Dict, Set
import numpy as np
from tqdm import tqdm
from functools import partial
from collections import defaultdict


class OptimizedMultiprocessing:
    def __init__(self, data_processor, n_processes=None):
        self.hp_embeddings = data_processor.hp_embeddings
        self.disease_to_hps = data_processor.disease_to_hps
        self.n_processes = n_processes or mp.cpu_count()
        # Precompute disease phenotypes that have embeddings
        self.valid_disease_phenotypes = {
            disease_id: [p for p in data["phenotypes"] if p in self.hp_embeddings]
            for disease_id, data in self.disease_to_hps.items()
        }

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def precompute_all_needed_similarities(self, phenotype_sets: List[List[str]]) -> Dict:
        """Precompute all cosine similarities that will be needed"""
        # Get unique phenotypes from input sets
        all_input_phenotypes: Set[str] = set()
        for pheno_set in phenotype_sets:
            all_input_phenotypes.update(pheno_set)

        # Get unique disease phenotypes
        all_disease_phenotypes: Set[str] = set()
        for disease_data in self.disease_to_hps.values():
            all_disease_phenotypes.update(disease_data["phenotypes"])

        # Filter for valid phenotypes
        valid_input_phenotypes = {p for p in all_input_phenotypes if p in self.hp_embeddings}
        valid_disease_phenotypes = {p for p in all_disease_phenotypes if p in self.hp_embeddings}

        # Create pairs for computation
        pairs_to_compute = [(i, d) for i in valid_input_phenotypes for d in valid_disease_phenotypes]

        # Prepare embeddings dictionary
        embeddings_dict = {
            hp: np.array(self.hp_embeddings[hp]['embeddings'])
            for hp in valid_input_phenotypes.union(valid_disease_phenotypes)
        }

        # Split for parallel processing
        chunk_size = len(pairs_to_compute) // self.n_processes + 1
        pair_chunks = [pairs_to_compute[i:i + chunk_size]
                       for i in range(0, len(pairs_to_compute), chunk_size)]

        # Process chunks in parallel
        with mp.Pool(processes=self.n_processes) as pool:
            chunk_results = list(tqdm(
                pool.imap(
                    partial(self._compute_similarities_chunk, embeddings_dict=embeddings_dict),
                    pair_chunks
                ),
                total=len(pair_chunks),
                desc="Computing cosine similarities"
            ))

        # Merge results
        similarity_cache = {}
        for chunk_result in chunk_results:
            similarity_cache.update(chunk_result)

        return similarity_cache

    @staticmethod
    def _compute_similarities_chunk(pairs: List[tuple], embeddings_dict: Dict) -> Dict:
        chunk_results = {}
        for hp1, hp2 in pairs:
            vec1 = embeddings_dict[hp1]
            vec2 = embeddings_dict[hp2]
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            chunk_results[(hp1, hp2)] = similarity
        return chunk_results

    def _process_phenotype_set(self, args):
        phenotype_set, similarity_cache = args

        best_match_scores = defaultdict(lambda: {"best_scores": {}})
        valid_phenotypes = [hp for hp in phenotype_set if hp in self.hp_embeddings]

        if not valid_phenotypes:
            print(f"No valid phenotypes found in set: {phenotype_set}")
            return []

        # Process each disease
        for disease_id, disease_phenotypes in self.valid_disease_phenotypes.items():
            if not disease_phenotypes:
                continue

            disease_name = self.disease_to_hps[disease_id]["disease_name"]

            # For each input phenotype, find best matching disease phenotype
            for input_hp in valid_phenotypes:
                best_match = max(
                    similarity_cache.get((input_hp, disease_hp), float('-inf'))
                    for disease_hp in disease_phenotypes
                )

                if best_match > float('-inf'):
                    best_match_scores[disease_id]["disease_name"] = disease_name
                    best_match_scores[disease_id]["best_scores"][input_hp] = best_match

        # Sort and create results
        results = []
        sorted_scores = sorted(
            best_match_scores.items(),
            key=lambda item: np.mean(list(item[1]["best_scores"].values())),
            reverse=True
        )

        for disease_id, scores in sorted_scores[:self.nr_of_results]:
            avg_score = float(np.mean(list(scores["best_scores"].values())))
            results.append(PhEvalDiseaseResult(
                disease_identifier=disease_id,
                disease_name=scores["disease_name"],
                score=avg_score
            ))

        return results

    def process_all_sets(self, phenotype_sets: List[List[str]], nr_of_results: int):
        """Process all phenotype sets with precomputed similarities"""
        self.nr_of_results = nr_of_results  # Store for use in _process_phenotype_set

        # First precompute all similarities
        print("Precomputing cosine similarities...")
        similarity_cache = self.precompute_all_needed_similarities(phenotype_sets)

        # Process each phenotype set with the precomputed similarities
        print("Processing phenotype sets...")
        process_args = [(pheno_set, similarity_cache) for pheno_set in phenotype_sets]

        with mp.Pool(processes=self.n_processes) as pool:
            results = list(tqdm(
                pool.imap(self._process_phenotype_set, process_args),
                total=len(phenotype_sets),
                desc="Processing phenotype sets"
            ))

        return results