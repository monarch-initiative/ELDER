from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np

from pheval_elder.prepare.core.store.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processing.data_processor import DataProcessor
from pheval_elder.prepare.core.collections.disease_avg_embedding_service import DiseaseAvgEmbeddingService
from pheval_elder.prepare.core.collections.disease_weighted_avg_embedding_service import DiseaseWeightedAvgEmbeddingService


class TermSetPairWiseComparisonQuery:
    data_processor: DataProcessor
    db_manager: ChromaDBManager
    average_llm_embedding_service: DiseaseAvgEmbeddingService
    weighted_average_llm_embedding_service: DiseaseWeightedAvgEmbeddingService
    similarity_strategy = None,

    def __post_init__(self):
        self.db_manager = self.db_manager
        self.data_processor = self.data_processor
        self.similarity_strategy = self.similarity_strategy
        self.hp_embeddings = self.data_processor.hp_embeddings
        self.disease_to_hps_from_omim = self.data_processor.disease_to_hps_with_frequencies
        self.disease_service = self.average_llm_embedding_service
        self.disease_weighted_service = self.weighted_average_llm_embedding_service

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def termset_pairwise_comparison_disease_embeddings(
        self,
        hps: List[str],
    ) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]], dict[str, float]]:

        final_disease_scores: Dict[str, float] = defaultdict() # final average cosine similarity for each disease from best matches
        best_match_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        pairwise_scores: Dict[str, Dict[str, Dict[str, float]]] = {}

        for phenotype in hps:
            phenotype_embedding = self.hp_embeddings[phenotype]
            disease_cosines = [] # store cosim for this hp with all diseases

            for disease, disease_phenotypes in self.disease_to_hps_from_omim.items():
                disease_phenotype_embeddings = [self.hp_embeddings[d_hp] for d_hp in disease_phenotypes]
                all_cosine_similarities = [self.cosine_similarity(phenotype_embedding, embedding)
                                       for embedding in disease_phenotype_embeddings]

                disease_cosines.append(all_cosine_similarities)
                if disease not in pairwise_scores:
                    pairwise_scores[disease] = {}
                # pairwise each hp to each hp in disease
                pairwise_scores[disease][phenotype] = dict(
                    zip(disease_phenotypes, all_cosine_similarities)
                )

                # pairwise_scores = {
                #     "DiseaseA": {
                #         "HP:0001250": {"HP:0001250": 0.98, "HP:0001263": 0.76, "HP:0001274": 0.85},
                #         "HP:0004322": {"HP:0001250": 0.45, "HP:0001263": 0.60, "HP:0001274": 0.40},
                #     },
                #     "DiseaseB": {
                #         "HP:0001250": {"HP:0004322": 0.32, "HP:0002110": 0.48, "HP:0008123": 0.52},
                #         "HP:0004322": {"HP:0004322": 0.95, "HP:0002110": 0.87, "HP:0008123": 0.78},
                #     },
                # }

                best_match = max(all_cosine_similarities)
                best_match_scores[disease][phenotype] = best_match

                # best_match_scores = {
                #     "DiseaseA": {"HP:0001250": 0.98, "HP:0004322": 0.60},
                #     "DiseaseB": {"HP:0001250": 0.52, "HP:0004322": 0.95},
                # }

                final_disease_scores: Dict[str, float] = {
                    disease: float(np.mean(list(scores.values()))) for disease, scores in best_match_scores.items()
                }

                # final_disease_scores = {
                #     "DiseaseA": 0.79,  # mean([0.98, 0.60])
                #     "DiseaseB": 0.735,  # mean([0.52, 0.95])
                # }

        return dict(pairwise_scores), dict(best_match_scores), final_disease_scores





        # weighting the best match similarities ?
        # weighting coourccrences of phenotypes/diseases ?

        # thing that doing post processing with an llm after i get the list to check relevant options and rerank like for example if there is a disease it cannot be with because
    # a phenotype has a way higher weighting there and it would not fit
    # or even there is another phenotype missing that from weighting must be in