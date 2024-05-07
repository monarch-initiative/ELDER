import time

import numpy as np
from chromadb.types import Collection

# from pheval_elder.prepare.elder_core.base_service import BaseService
# from pheval_elder.prepare.elder_core.data_processor import DataProcessor
from tqdm import tqdm

from pheval_elder.prepare.core.base_service import BaseService
from pheval_elder.prepare.core.data_processor import DataProcessor
from pheval_elder.prepare.core.graph_data_processor import GraphDataProcessor


# from pheval_elder.prepare.elder_core.graph_data_processor import GraphDataProcessor


class DiseaseAvgEmbeddingService(BaseService):
    """
    upsert averaged embeddings from hp_embeddings (cached dict from ont_hp collection) that are connected to the
    relevant disease from disease_to_hps (cached dict from hpoa) into the disease_avg_embeddings_collection that
    contains disease and the average embeddings of the correlating hp terms
    """

    def __init__(self, data_processor: DataProcessor, graph_data_processor: GraphDataProcessor):
        super().__init__(
            data_processor=data_processor,
            graph_data_processor=graph_data_processor
        )

    def process_data(self) -> Collection:
        if not self.disease_to_hps:
            raise ValueError("disease to hps data is not initialized")
        if not self.disease_new_avg_embeddings_collection:
            raise ValueError("disease_new_avg_embeddings collection is not initialized")

        # if self.disease_new_avg_embeddings_collection:
        #     print("DiseaseAvgEmbeddings collection already initialized!")
        #     return self.disease_new_avg_embeddings_collection

        batch_size = 100
        num_diseases = len(self.disease_to_hps)
        batch_embeddings = np.zeros((batch_size, 1536))
        # Use dtype=object for flexibility with string lengths and special characters, avoiding truncation issues.
        # special numpy stuff, otherwise if length of str changes in future numpy would with later insertions truncate it
        batch_diseases = np.empty(batch_size, dtype=object)
        current_index = 0
        embedding_calc_time = 0
        upsert_time = 0

        for disease, hps in tqdm(self.disease_to_hps.items(), total=num_diseases):
            start = time.time()
            average_embedding = self.data_processor.calculate_average_embedding(hps, self.hp_embeddings)
            embedding_calc_time += time.time() - start

            batch_diseases[current_index] = disease
            batch_embeddings[current_index] = average_embedding
            current_index += 1

            if current_index % batch_size == 0:
                self.upsert_batch(batch_diseases, batch_embeddings)
                current_index = 0

        if current_index > 0:
            start = time.time()
            self.upsert_batch(batch_diseases[:current_index], batch_embeddings[:current_index])
            upsert_time += time.time() - start

        print(f"Total time for embedding calculations (avg): {embedding_calc_time}s")
        print(f"Total time for upsert operations (avg): {upsert_time}s")

        return self.disease_new_avg_embeddings_collection

    def upsert_batch(self, disease_ids, embeddings):
        valid_indices = [i for i, disease in enumerate(disease_ids) if disease is not None]
        filtered_ids = [disease_ids[i] for i in valid_indices]
        metadatas = [{"type": "disease"}] * len(disease_ids)
        self.disease_new_avg_embeddings_collection.upsert(ids=filtered_ids, embeddings=embeddings, metadatas=metadatas)

