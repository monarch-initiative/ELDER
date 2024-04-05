from typing import Dict

import numpy as np

from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures
from pheval_elder.prepare.core.data_processor import DataProcessor
from pheval_elder.prepare.core.hpo_clustering import HPOClustering


class EmbeddingsAggregator:
    def __init__(self):
        self.manager = ChromaDBManager(SimilarityMeasures.COSINE)
        self.ont_hp = self.manager.get_collection("ont_hp")
        self.data_processor = DataProcessor(db_manager=self.manager)
        self.disease_to_hps = self.data_processor.disease_to_hps_with_frequencies
        self.hp_to_embedding = self.data_processor.hp_embeddings
        self.hpo_clustering = HPOClustering()

    def get_disease_dict(self) -> Dict:
        return self.disease_to_hps

    def get_hp_embeddings(self) -> Dict:
        return self.hp_to_embedding

    def aggregate_embeddings(self) -> Dict:
        # go over each diseas in disease dict
        disease_average_embeddings = {}
        for disease, hp_terms in self.disease_to_hps.items():
            embeddings = [self.hp_to_embedding[hp]['embeddings'] for hp in hp_terms if hp in self.hp_to_embedding]
            if embeddings:
                disease_average_embeddings[disease] = np.mean(embeddings, axis=0)
        return disease_average_embeddings
