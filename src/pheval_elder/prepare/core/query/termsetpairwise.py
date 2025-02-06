from collections import defaultdict
from typing import List, Dict, Union

import numpy as np
from pydantic import BaseModel

AVG_BEST_MATCH_SCORE = Dict[str, Union[int, float]]
BEST_MATCH_SCORES = Dict[str, Dict[str, float]]
PAIRWISE_SCORES = Dict[str, Dict[str, Dict[str, float]]]
AVG_PAIRWISE_SCORE = Dict[str, Union[int, float]]


class TPCResult(BaseModel):
    pairwise_scores: PAIRWISE_SCORES
    best_match_scores: BEST_MATCH_SCORES
    avg_best_match_score: AVG_BEST_MATCH_SCORE
    avg_pairwise_score: AVG_PAIRWISE_SCORE
    sorted_avg_pairwise_score: list
    sorted_avg_best_match_score: list

class TermSetPairWiseComparisonQuery:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.similarity_strategy = None
        self.hp_embeddings = self.data_processor.hp_embeddings
        self.disease_to_hps_from_omim = self.data_processor.disease_to_hps

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def termset_pairwise_comparison_disease_embeddings(
        self,
        hps: List[str],
        nr_of_results: int
    ) -> TPCResult:

        pairwise_scores: PAIRWISE_SCORES = defaultdict(dict)
        best_match_scores: BEST_MATCH_SCORES = defaultdict(dict)


        # loop over hps from input list & get embedding for input_hp
        for input_hp in hps:
            if input_hp not in self.hp_embeddings:
                raise ValueError(f"Embedding for phenotype {input_hp} not found in hp_embeddings.")
            input_hp_embedding = self.hp_embeddings[input_hp]['embeddings']

            # loop over each rare disease in omim
            for disease, disease_phenotypes in self.disease_to_hps_from_omim.items():
                # create a new list for each disease to store cosims of iHP to dHP.
                ihp_to_dhps_cosine_similarities = []
                # get all phenotypes from the disease and get the embedding of it
                for disease_hp in disease_phenotypes:
                    if disease_hp not in self.hp_embeddings:
                        raise ValueError(f"Embedding for disease phenotype {disease_hp} not found in hp_embeddings.")
                    disease_hp_embedding = self.hp_embeddings[disease_hp]['embeddings']

                    # calculate cosim for the input_hp & EACH disease_hp ( x1-y1; x1-y2; x1-y3; ... )
                    cosine_ = self.cosine_similarity(input_hp_embedding, disease_hp_embedding)
                    ihp_to_dhps_cosine_similarities.append(cosine_)


                # pairwise each hp to each hp in disease
                pairwise_scores[disease][input_hp] = dict(
                    zip(disease_phenotypes, ihp_to_dhps_cosine_similarities)
                )

                """
                 pairwise_scores = {
                     "DiseaseA": {
                         "HP:0001250": {"HP:0001250": 0.98, "HP:0001263": 0.76, "HP:0001274": 0.85}, # row avg = 0.863
                         "HP:0004322": {"HP:0001250": 0.45, "HP:0001263": 0.60, "HP:0001274": 0.40}, row_avg = 0.483
                     },
                     "DiseaseB": {
                         "HP:0001250": {"HP:0004322": 0.32, "HP:0002110": 0.48, "HP:0008123": 0.52},
                         "HP:0004322": {"HP:0004322": 0.95, "HP:0002110": 0.87, "HP:0008123": 0.78},
                     },
                 }
                """

                best_match = max(ihp_to_dhps_cosine_similarities)
                best_match_scores[disease][input_hp] = best_match

                """
                 best_match_scores = {
                     "DiseaseA": {"HP:0001250": 0.98, "HP:0004322": 0.60},
                     "DiseaseB": {"HP:0001250": 0.52, "HP:0004322": 0.95},
                 }
                """

        avg_pairwise_score: AVG_PAIRWISE_SCORE = {
            disease: np.mean([np.mean(list(iHP.values())) for iHP in iHPs.values()])
            for disease, iHPs in pairwise_scores.items()
        }

        """
                avg_pairwise_score = {
                     "DiseaseA": 0.79,  # mean([0.98, 0.76, 0.85])
                     "DiseaseB": 0.735,  # mean([0.45, 0.60, 0.40])
                }
                """


        avg_best_match_score: AVG_BEST_MATCH_SCORE  = {
            disease: float(np.mean(list(scores.values()))) for disease, scores in best_match_scores.items()
        }

        """
        # problem is when same HP = 1.0 and other is also inside its 1.0 but it does not take care of the other 10 hps inside just shows both are inside
        # feel like with embeddings this is doable that avg_pairwise scores make more sense, as it captures all from all - we'll see (tbd)
        avg_best_match_score = {
             "DiseaseA": 0.79,  # mean([0.98, 0.60])
             "DiseaseB": 0.735,  # mean([0.52, 0.95])
        }
        """

        tpc_result = TPCResult(
            pairwise_scores=pairwise_scores,
            best_match_scores=best_match_scores,
            avg_best_match_score=avg_best_match_score,
            avg_pairwise_score=avg_pairwise_score,
            sorted_avg_best_match_score=sorted(avg_best_match_score.items(), key=lambda x: x[1], reverse=True)[:nr_of_results],
            sorted_avg_pairwise_score=sorted(avg_pairwise_score.items(), key=lambda x: x[1], reverse=True)[:nr_of_results],
        )

        return tpc_result.sorted_avg_pairwise_score


        # weighting the best match similarities ?
        # weighting coourccrences of phenotypes/diseases ?

        # thing that doing post processing with an llm after i get the list to check relevant options and rerank like for example if there is a disease it cannot be with because
    # a phenotype has a way higher weighting there and it would not fit
    # or even there is another phenotype missing that from weighting must be


