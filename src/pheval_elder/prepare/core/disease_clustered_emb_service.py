import time
from typing import List

import numpy as np
from chromadb.types import Collection
from tqdm import tqdm

from src.pheval_elder.prepare.core.base_service import BaseService
from src.pheval_elder.prepare.core.data_processor import DataProcessor
from src.pheval_elder.prepare.core.graph_data_processor import GraphDataProcessor
from src.pheval_elder.prepare.core.hpo_clustering import HPOClustering
from src.pheval_elder.prepare.core.organ_systems import OrganSystems


# from pheval_elder.prepare.elder_core.base_service import BaseService
# from pheval_elder.prepare.elder_core.data_processor import DataProcessor
# from pheval_elder.prepare.elder_core.graph_data_processor import GraphDataProcessor
# from pheval_elder.prepare.elder_core.hpo_clustering import HPOClustering
# from pheval_elder.prepare.elder_core.organ_systems import OrganSystems


class DiseaseClusteredEmbeddingService(BaseService):
    """
    Handles the computation and upserting of clustered embeddings for diseases. These embeddings are created
    by clustering HPO terms associated with each disease and then concatenating the average embeddings of each cluster.
    """

    def __init__(self, data_processor: DataProcessor, hpo_clustering: HPOClustering,
                 graph_data_processor: GraphDataProcessor):
        super().__init__(
            data_processor=data_processor,
            graph_data_processor=graph_data_processor)

        self.hpo_clustering = hpo_clustering
        self.organs = len(self.get_all_clusters())
        self.embedding_size = 1536
        self.concatenated_embedding_size = self.embedding_size * self.organs
        self.organ_system_embeddings = {
            organ_hp_term_enum.value: self.hp_embeddings.get(organ_hp_term_enum.value).get('embeddings') for
            organ_hp_term_enum in OrganSystems if organ_hp_term_enum is not None
        }

    def get_all_clusters(self) -> list[str]:
        return self.hpo_clustering.all_organs

    def process_data(self) -> Collection:
        if not self.disease_to_hps:
            raise ValueError("Disease to HPO data is not initialized")
        if not self.clustered_new_embeddings_collection:
            raise ValueError("Clustered embeddings collection is not initialized")
        # if self.clustered_new_embeddings_collection:
        #     print("Clustered Embeddings collection early return, cause already initialized!")
        #     return self.clustered_new_embeddings_collection

        batch_size = 100
        num_diseases = len(self.disease_to_hps_with_frequencies_dp)
        all_embeddings = np.zeros((batch_size, self.concatenated_embedding_size))
        all_diseases = np.empty(batch_size, dtype=object)  #

        current_index = 0
        embedding_calc_time = 0
        upsert_time = 0

        for disease, hpo_terms in tqdm(self.disease_to_hps.items(), total=num_diseases):
            start = time.time()
            concatenated_organ_embeddings = self.compute_organ_embeddings_more_efficient(hpo_terms)
            embedding_calc_time += time.time() - start

            all_diseases[current_index] = disease
            all_embeddings[current_index] = concatenated_organ_embeddings
            current_index += 1

            if current_index % batch_size == 0:
                start = time.time()
                self.upsert_batch(all_diseases, all_embeddings)
                upsert_time += time.time() - start
                current_index = 0

        # Handling last batch
        if current_index > 0:
            start = time.time()
            self.upsert_batch(all_diseases[:current_index], all_embeddings[:current_index])
            upsert_time += time.time() - start

        print(f"Total time for embedding calculations (clustered): {embedding_calc_time}s")
        print(f"Total time for upsert operations (clustered): {upsert_time}s")

        return self.clustered_new_embeddings_collection

    def upsert_batch(self, disease_ids, embeddings):
        valid_indices = [i for i, diseases in enumerate(disease_ids) if diseases is not None]
        filtered_ids = [disease_ids[i] for i in valid_indices]
        metadatas = [{"type": "disease"}] * len(disease_ids)
        self.clustered_new_embeddings_collection.upsert(ids=filtered_ids, embeddings=embeddings, metadatas=metadatas)

    def compute_organ_embeddings_more_efficient(self, hp_input_list: List[str] = None) -> np.ndarray:
        # TODO: when getting embeddings for organ system get the one from all HP terms under the term and average!
        all_organ_systems = sorted(self.hpo_clustering.all_organs)

        # init arrays and counters outside the loop
        organ_embeddings = {organ: np.zeros((len(hp_input_list), self.embedding_size)) for organ in all_organ_systems}
        # tracks how many embeddings have been added per organ so HPs per organ
        organ_count = {organ: 0 for organ in all_organ_systems}
        terms_without_cluster = []

        # Populate organ_embeddings for the current disease
        for hpo_term in hp_input_list:
            organ_system = self.hpo_clustering.get_organ_system(hpo_term)
            if organ_system is None:
                terms_without_cluster.append(hpo_term)
                continue
            if organ_system:
                embedding = self.hp_embeddings.get(hpo_term, {}).get('embeddings')
                organ_embeddings[organ_system][organ_count[organ_system]] = embedding
                organ_count[organ_system] += 1

        concatenated_embeddings = np.concatenate([
            np.mean(organ_embeddings[organ][:organ_count[organ]], axis=0) if organ_count[organ] > 0
            else -1 * self.organ_system_embeddings.get(organ) for organ in all_organ_systems
        ])

        expected_size = self.concatenated_embedding_size
        if concatenated_embeddings.size != expected_size:
            raise ValueError(
                f"The final embedding size is incorrect: {concatenated_embeddings.size}, expected: {expected_size}")

        return concatenated_embeddings
