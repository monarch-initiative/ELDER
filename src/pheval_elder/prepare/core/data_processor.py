import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from typing import Dict, List, Any

import numpy as np
from chromadb.types import Collection

from src.pheval_elder.prepare.core.OMIMHPOExtractor import OMIMHPOExtractor
from src.pheval_elder.prepare.core.chromadb_manager import ChromaDBManager

# from pheval_elder.prepare.elder_core.chromadb_manager import ChromaDBManager
# from pheval_elder.prepare.elder_core.OMIMHPOExtractor import OMIMHPOExtractor

"""
    This class main function is to create cached dictionaries from the ont_hp and hpoa collection given by the
    ChromaDBManager. 
    Dictionaries are used for the setup of the collections
"""


class DataProcessor:
    def __init__(self, db_manager: ChromaDBManager):
        self.db_manager = db_manager
        self._hp_embeddings = None
        self._disease_to_hps = None
        self._disease_to_hps_with_frequencies = None
    @property
    def hp_embeddings(self) -> Dict:
        if self._hp_embeddings is None:
            self._hp_embeddings = self.create_hpo_id_to_embedding(self.db_manager.ont_hp)
        return self._hp_embeddings

    @property
    def disease_to_hps_with_frequencies(self) -> Dict:
        if self._disease_to_hps_with_frequencies is None:
            file_path = "/Users/carlo/PycharmProjects/chroma_db_playground/phenotype.hpoa"
            data = OMIMHPOExtractor.read_data_from_file(file_path)
            self._disease_to_hps_with_frequencies = OMIMHPOExtractor.extract_omim_hpo_mappings_with_frequencies_1(data)
        return self._disease_to_hps_with_frequencies

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
        for metadata, embedding in zip(results.get("metadatas", []), results.get("embeddings", []), strict=False):
            metadata_json = json.loads(metadata["_json"])
            hpo_id = metadata_json.get("original_id")
            if hpo_id:
                hpo_id_to_data[hpo_id] = {"embeddings": embedding}  # #{'HP:0005872': [1,2,3, ...]}
        return {k: {'embeddings': np.array(v['embeddings'])} for k, v in hpo_id_to_data.items()}

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
            phenotype = metadata_json.get("phenotype")
            # label = metadata_json.get("disease_label")
            if disease and phenotype:
                if disease not in disease_to_hps_dict:
                    disease_to_hps_dict[disease] = [phenotype]  # put the label
                else:
                    disease_to_hps_dict[disease].append(phenotype)  # put the label
        # print(len(disease_to_hps_dict))
        # print(disease_to_hps_dict)
        return disease_to_hps_dict

    @staticmethod
    def calculate_average_embedding(hps: list, embeddings_dict: Dict) -> np.ndarray:
        """
        Calculates the average embedding for a given set of HPO IDs.

        :param hps: List of HPO IDs.
        :param embeddings_dict: Dictionary mapping HPO IDs to their embeddings.
        :return: A numpy array representing the average embedding for the HPO IDs.
        """
        embeddings = [embeddings_dict[hp_id]["embeddings"] for hp_id in hps if hp_id in embeddings_dict]
        return np.mean(embeddings, axis=0) if embeddings else []

    @staticmethod
    def load_embeddings_from_json(path):
        with open(path, 'r') as f:
            data = f.read()
            json_data = json.loads(data)
        embeddings = DataProcessor.convert_embeddings_to_numpy(json_data)
        return embeddings

    # Creating a function to convert lists to numpy arrays
    @staticmethod
    def convert_embeddings_to_numpy(embeddings_dict):
        return {k: {'embeddings': np.array(v['embeddings'])} for k, v in embeddings_dict.items()}

    # Adjusting the calculate_average_embedding_weighted function to include conversion
    @staticmethod
    def calculate_average_embedding_weighted(hps_with_frequencies: dict, embeddings_dict: dict) -> np.ndarray:
        # Convert embeddings to numpy arrays if they are lists
        embeddings_dict = DataProcessor.convert_embeddings_to_numpy(embeddings_dict)

        weighted_embeddings = []
        total_weight = 0
        for hp_id, proportion in hps_with_frequencies.items():
            embedding = embeddings_dict.get(hp_id, {}).get('embeddings')
            if embedding is not None:
                weighted_embeddings.append(proportion * embedding)
                total_weight += proportion

        if total_weight > 0:
            return np.sum(weighted_embeddings, axis=0) / total_weight
        return np.array([])

    def calculate_weighted_average_for_hp_inputlist(
        self,
        disease: list[str],
        hpo_ids: list[str]
    ) -> np.ndarray:

        weighted_embeddings = np.zeros_like(next(iter(self.hp_embeddings.values()))['embeddings'])
        total_weight = 0

        hps_with_frequencies = self.disease_to_hps_with_frequencies[disease]

        for hpo_id in hpo_ids:
            if hpo_id in hps_with_frequencies.keys():

                for hp_id, proportion in hps_with_frequencies.items():
                    embedding = self.hp_embeddings.get(hp_id, {}).get('embeddings')
                    if embedding is not None:
                        weighted_embeddings += proportion * embedding
                        total_weight += proportion

        return weighted_embeddings / total_weight if total_weight > 0 else np.array([])

        pass

    # TODO: include in integration test to see how it works on full dict as this is more useful in query_service
    def calculate_weighted_llm_embeddings(self, disease: str) -> np.ndarray:

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

        weighted_embeddings = np.zeros_like(next(iter(self.hp_embeddings.values()))['embeddings'])
        # TODO: iitialize HP embedding when starting otherwise this gets called 12k times ..
        # TODO: Explanation - >went back to this as we need 1D array since we do it each disease one by one
        ## TODO: so shape is (1536,) not (12468,1536)
        total_weight = 0
        # gets called from a loop & the disease will be updated -> more efficient than transporting the whole dict
        hps_with_frequencies = self.disease_to_hps_with_frequencies[disease]

        for hp_id, proportion in hps_with_frequencies.items():
            embedding = self.hp_embeddings.get(hp_id, {}).get('embeddings')
            if embedding is not None:
                weighted_embeddings += proportion * embedding
                total_weight += proportion

        return weighted_embeddings / total_weight if total_weight > 0 else np.array([])

    def process_disease_data(self, disease, disease_to_hps_from_omim, embeddings_dict):
        # Function that will be executed in parallele for each disease
        average_embedding = self.calculate_weighted_llm_embeddings(
            disease_to_hps_from_omim=disease_to_hps_from_omim[disease],
            embeddings_dict=embeddings_dict
        )
        return disease, average_embedding

    def process_data_parallel(self, disease_to_hps_from_omim, embeddings_dict):
        all_data = []
        with ProcessPoolExecutor() as executor:
            # Create a list of futures
            futures = [executor.submit(self.process_disease_data, disease, disease_to_hps_from_omim, embeddings_dict)
                       for disease in disease_to_hps_from_omim]

            for future in as_completed(futures):
                disease, average_embedding = future.result()
                all_data.append((disease, average_embedding))

        return all_data


    @staticmethod
    def calculate_average_embedding_weighted_old(hps_with_frequencies: dict, embeddings_dict: dict) -> np.ndarray:
        """
        Calculates the weighted average embedding for a given set of HPO IDs and their frequencies.

        :param hps_with_frequencies: Dictionary of HPO IDs and their frequencies.
        :param embeddings_dict: Dictionary mapping HPO IDs to their embeddings.
        :return: A numpy array representing the weighted average embedding for the HPO IDs.
        """
        weighted_embeddings = []
        total_weight = 0

        for hp_id, proportion in hps_with_frequencies.items():
            embedding = embeddings_dict.get(hp_id, {}).get('embeddings')
            if embedding is not None:
                if isinstance(embedding, np.ndarray):
                    weighted_embeddings.append(proportion * np.array(embedding))
                    total_weight += proportion

        if total_weight > 0:
            return np.sum(weighted_embeddings, axis=0) / total_weight
        return np.array([])



    # Example usage
    # hps_with_frequencies = {"HP:0001": 75.0, "HP:0002": 50.0}
    # embeddings_dict = {"HP:0001": {"embeddings": [1, 2, 3]}, "HP:0002": {"embeddings": [4, 5, 6]}}
    # result = calculate_average_embedding_weighted(hps_with_frequencies, embeddings_dict)

    @staticmethod
    def extract_and_use_omim_hpo_mappings(file_path):
        with open(file_path, "r") as file:
            data = file.read()
        return OMIMHPOExtractor.extract_omim_hpo_mappings_with_frequencies_1(data)
