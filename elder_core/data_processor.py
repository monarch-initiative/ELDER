import json
from typing import Dict

from chromadb.types import Collection
from elder_core.chromadb_manager import ChromaDBManager
import numpy as np

from elder_core.OMIMHPOExtractor import OMIMHPOExtractor

"""
    This class main function is to create cached dictionaries from the ont_hp and hpoa collection given by the 
    ChromaDBManager. It also should calculate the 
"""


class DataProcessor:
    def __init__(self, db_manager: ChromaDBManager):
        self.db_manager = db_manager
        self._hp_embeddings = None
        self._disease_to_hps = None

    @property
    def hp_embeddings(self) -> Dict:
        if self._hp_embeddings is None:
            self._hp_embeddings = self.create_hpo_id_to_embedding(self.db_manager.ont_hp)
        return self._hp_embeddings

    # @property
    # def disease_to_hps(self) -> Dict:
    #     if self._disease_to_hps is None:
    #         self._disease_to_hps = self.create_disease_to_hps_dict(self.db_manager.hpoa)
    #     return self._disease_to_hps

    @property
    def disease_to_hps(self) -> Dict:
        if self._disease_to_hps is None:
            # traceback.print_stack()  # This will print where the call is coming from
            file_path = "/Users/carlo/PycharmProjects/chroma_db_playground/phenotype.hpoa"
            data = OMIMHPOExtractor.read_data_from_file(file_path)
            self._disease_to_hps = OMIMHPOExtractor.extract_omim_hpo_mappings_default(data)
        return self._disease_to_hps

    @staticmethod
    def create_hpo_id_to_embedding(collection: Collection) -> Dict:
        """
        Create a dictionary mapping HPO IDs to embeddings.

        :param collection: The collection to process
        :return: A dictionary mapping HPO IDs to a dictionary of their label and embeddings.
        """
        hpo_id_to_data = {}
        results = collection.get(include=["metadatas", "embeddings"])
        for metadata, embedding in zip(results.get("metadatas", []), results.get("embeddings", [])):
            metadata_json = json.loads(metadata['_json'])
            hpo_id = metadata_json.get("original_id")
            if hpo_id:
                hpo_id_to_data[hpo_id] = {"embeddings": embedding}  # #{'HP:0005872': [1,2,3, ...]}
        return hpo_id_to_data

    # to be faster
    # results = collection.get(include=["metadatas", "embeddings"])
    # metadatas = results.get("metadatas", [])
    # embeddings = results.get("embeddings", [])
    # hpo_id_to_data = {
    #     json.loads(metadata['_json']).get("original_id"): {"embeddings": embedding}
    #     for metadata, embedding in zip(metadatas, embeddings)
    #     if json.loads(metadata['_json']).get("original_id")
    # }

    @staticmethod
    def create_disease_to_hps_dict(collection: Collection) -> Dict:
        """
        Creates a dictionary mapping diseases (OMIM IDs) to their associated HPO IDs.

        :param collection: The collection to process
        :return: Dictionary with diseases as keys and lists of corresponding HPO IDs as values.
        """
        disease_to_hps_dict = {}
        results = collection.get(include=["metadatas"])
        for item in results.get("metadatas"):
            metadata_json = json.loads(item["_json"])
            disease = metadata_json.get("disease")
            phenotype = metadata_json.get('phenotype')
            if disease and phenotype:
                if disease not in disease_to_hps_dict:
                    disease_to_hps_dict[disease] = [phenotype]
                else:
                    disease_to_hps_dict[disease].append(phenotype)
        return disease_to_hps_dict

    @staticmethod
    def calculate_average_embedding(hps: list, embeddings_dict: Dict) -> np.ndarray:
        """
        Calculates the average embedding for a given set of HPO IDs.

        :param hps: List of HPO IDs.
        :param embeddings_dict: Dictionary mapping HPO IDs to their embeddings.
        :return: A numpy array representing the average embedding for the HPO IDs.
        """
        embeddings = [embeddings_dict[hp_id]['embeddings'] for hp_id in hps if hp_id in embeddings_dict]
        return np.mean(embeddings, axis=0) if embeddings else []

    # deprecated cause using hpoa collection instead of .hpoa file now
    @staticmethod
    def extract_and_use_omim_hpo_mappings(file_path):
        with open(file_path, 'r') as file:
            data = file.read()
        return OMIMHPOExtractor.extract_omim_hpo_mappings(data)

