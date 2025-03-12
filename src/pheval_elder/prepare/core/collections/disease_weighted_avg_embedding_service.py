import time
from dataclasses import dataclass

import numpy as np
from chromadb.types import Collection
from tqdm import tqdm

from pheval_elder.prepare.core.collections.base_service import BaseService
from pheval_elder.prepare.core.data_processing.data_processor import DataProcessor


@dataclass
class DiseaseWeightedAvgEmbeddingService(BaseService):
    """
    upsert averaged embeddings from hp_embeddings (cached dict from ont_hp collection) that are connected to the
    relevant disease from disease_to_hps (cached dict from hpoa) into the disease_avg_embeddings_collection that
    contains disease and the average embeddings of the correlating hp terms
    """
    data_processor: DataProcessor

    def process_data(self) -> Collection:
        if not self.disease_to_hps_with_frequencies_dp:
            raise ValueError("disease to hps data is not initialized")
        if not self.disease_weighted_avg_embeddings_collection:
            raise ValueError("disease_new_avg_embeddings collection is not initialized")
        # if self.disease_weighted_avg_embeddings_collection:
        #     print("Clustered Embeddings collection early return, cause already initialized!")
        #     return self.disease_weighted_avg_embeddings_collection

        batch_size = 100
        num_diseases = len(self.disease_to_hps_with_frequencies_dp)

        all_embeddings = np.zeros((batch_size, 3072))
        all_diseases = np.empty(batch_size, dtype=object)

        current_index = 0

        for disease, disease_and_phenotype_data in tqdm(self.disease_to_hps_with_frequencies_dp.items(), total=num_diseases):
            start = time.time()
            disease_name = disease_and_phenotype_data.get("disease_name")
            average_weighted_embedding = self.data_processor.calculate_weighted_llm_embeddings(
                disease=disease)

            all_diseases[current_index] = (disease,disease_name)
            all_embeddings[current_index] = average_weighted_embedding
            current_index += 1

            if current_index % batch_size == 0:
                start = time.time()
                self.upsert_batch(all_diseases, all_embeddings)
                current_index = 0

        if current_index > 0:
            self.upsert_batch(all_diseases[:current_index], all_embeddings[:current_index])

        return self.disease_weighted_avg_embeddings_collection

    def upsert_batch(self, disease_data, embeddings):
        valid_indices = [i for i, disease in enumerate(disease_data) if disease is not None]
        filtered_entries = [disease_data[i] for i in valid_indices]
        filtered_embeddings = embeddings[valid_indices]
        ids = [x[0] for x in filtered_entries]
        disease_name = [{"disease_name": y[0]} for y in filtered_entries]
        self.disease_weighted_avg_embeddings_collection.upsert(ids=ids, embeddings=filtered_embeddings,
                                                               metadatas=disease_name)
