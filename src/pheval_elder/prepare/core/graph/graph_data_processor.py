import json
from typing import Dict

import numpy as np
from chromadb.types import Collection


from pheval_elder.prepare.core.data_processing.OMIMHPOExtractor import OMIMHPOExtractor
from pheval_elder.prepare.core.data_processing.graph_data_extractor import GraphEmbeddingExtractor
from pheval_elder.prepare.core.store.chromadb_manager import ChromaDBManager


class GraphDataProcessor:

    def __init__(self, extractor: GraphEmbeddingExtractor, manager: ChromaDBManager):
        self.extractor = extractor
        self.manager = manager
        self._line_graph_embeddings = None # make it property
        self._deepwalk_graph_embeddings = None
        self._disease_to_hps = None
        self._disease_to_hps_with_frequencies = None

    def line_graph_embeddings(self) -> dict:
        if self._line_graph_embeddings is None:
            self._line_graph_embeddings = self.extractor.parse_line_embeddings()
        return self._line_graph_embeddings

    def deepwalk_graph_embeddings(self) -> dict:
        if self._deepwalk_graph_embeddings is None:
            self._deepwalk_graph_embeddings = self.extractor.parse_deep_embeddings()
        return self._deepwalk_graph_embeddings

    def disease_to_hp_default(self) -> dict:
        if self._disease_to_hps is None:
            file_path = "/Users/carlo/PycharmProjects/chroma_db_playground/phenotype.hpoa"
            data = OMIMHPOExtractor.read_data_from_file(file_path)
            self._disease_to_hps = OMIMHPOExtractor.extract_omim_hpo_mappings_default(data)
        return self._disease_to_hps

    def disease_to_hp_with_frequencies(self) -> dict:
        if self._disease_to_hps_with_frequencies is None:
            file_path = "/Users/carlo/PycharmProjects/chroma_db_playground/phenotype.hpoa"
            data = OMIMHPOExtractor.read_data_from_file(file_path)
            self._disease_to_hps_with_frequencies = OMIMHPOExtractor.extract_omim_hpo_mappings_with_frequencies_1(data)
        return self._disease_to_hps_with_frequencies

    def calculate_average_line_graph_embeddings(self, hps: list) -> np.ndarray:
        embeddings = [self._line_graph_embeddings[hp_id] for hp_id in hps if hp_id in self._line_graph_embeddings]
        return np.mean(embeddings, axis=0) if embeddings else []
        ## TODO: typical error new embedding length and need make class instance to pull from for new data processor
    def calculate_average_deepwalk_graph_embeddings(self, hps: list) -> np.ndarray:
        embeddings = [self._deepwalk_graph_embeddings[hp_id] for hp_id in hps if hp_id in self._deepwalk_graph_embeddings]
        return np.mean(embeddings, axis=0) if embeddings else []

    '''
    Weighted Average Calculation:
    For each disease:
        1. iterate through the embeddings associated with it.
        2. Multiply each embedding by its corresponding frequency (weight).
        3. Accumulate these weighted embeddings.
        4. Sum the frequencies (weights).
    Finally, dividing the sum of weighted embeddings by the sum of weights to get the average embedding for the disease.
    This approach effectively gives more "importance" to embeddings with higher frequencies in the final average embedding calculation.
    '''

    def calculate_weighted_line_graph_embeddings(self, disease: str) -> np.ndarray:
        weighted_embeddings = np.zeros_like(next(iter(self._line_graph_embeddings.values())))
        total_weight = 0
        hps_with_frequencies = self._disease_to_hps_with_frequencies[disease]

        for hp_id, proportion in hps_with_frequencies.items():
            embedding = self._line_graph_embeddings.get(hp_id, {})
            if embedding is not None:
                weighted_embeddings += proportion * embedding
                total_weight += proportion

        return weighted_embeddings / total_weight if total_weight > 0 else np.array([])

    def calculate_weighted_deepwalk_graph_embeddings(self, disease: str) -> np.ndarray:
        weighted_embeddings = np.zeros_like(next(iter(self._deepwalk_graph_embeddings.values())))
        total_weight = 0
        hps_with_frequencies = self._disease_to_hps_with_frequencies[disease]

        for hpo_id, frequency in hps_with_frequencies.items():
            embedding = self._deepwalk_graph_embeddings.get(hpo_id, {})
            if embedding is not None:
                weighted_embeddings += frequency * embedding
                total_weight += frequency

        return weighted_embeddings / total_weight if total_weight > 0 else np.array([])