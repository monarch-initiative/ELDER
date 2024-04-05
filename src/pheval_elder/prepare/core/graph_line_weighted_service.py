import logging
import time

import numpy as np
from chromadb.types import Collection
from tqdm import tqdm

from src.pheval_elder.prepare.core.base_service import BaseService
from src.pheval_elder.prepare.core.data_processor import DataProcessor
from src.pheval_elder.prepare.core.graph_data_processor import GraphDataProcessor

# from pheval_elder.prepare.elder_core.base_service import BaseService
# from pheval_elder.prepare.elder_core.data_processor import DataProcessor
# from pheval_elder.prepare.elder_core.graph_data_processor import GraphDataProcessor

logger = logging.getLogger(__name__)
class GraphLineWeightedEmbeddingService(BaseService):
    def __int__(self, data_processor: DataProcessor, graph_data_processor: GraphDataProcessor):
        super().__init__(
            data_processor=data_processor,
            graph_data_processor=graph_data_processor
        )

    def process_data(self) -> Collection:
        if not self.graph_data_processor.line_graph_embeddings:
            raise ValueError("Line Embeddings are not initialized!")
        if not self.data_processor.disease_to_hps_with_frequencies:
            raise ValueError("DiseaseToHPs with frequencies Dictionary is not initialized!")
        # if self.disease_weighted_avg_line_graph_embeddings_collection:
        #     print("Weighted Line Graph Embeddings collection already initialized!")
        #     return self.disease_weighted_avg_line_graph_embeddings_collection

        batch_size = 100
        # TODO: num_disease should be batch_size as we will make a new one every batch ?
        num_diseases = len(self.disease_to_hps_with_frequencies_dp)

        all_embeddings = np.zeros((batch_size, 100))  # Replace 1536 with better EmbeddingsSize?

        # Initialize an array to store disease IDs
        # Use dtype=object for flexibility with string lengths and special characters, avoiding truncation issues
        # special numpy stuff, otherwise if length of str changes in future numpy would with later insertions truncate it
        all_diseases = np.empty(batch_size, dtype=object)

        current_index = 0
        embedding_calc_time = 0
        upsert_time = 0

        for disease in tqdm(self.disease_to_hps_with_frequencies_dp.keys(), total=num_diseases):
            start = time.time()
            average_weighted_embedding = self.graph_data_processor.calculate_weighted_line_graph_embeddings(
                disease=disease)
            embedding_calc_time += time.time() - start

            all_diseases[current_index] = disease
            all_embeddings[current_index] = average_weighted_embedding
            current_index += 1

            # Check if the current batch is full, and if so, upsert it to the database
            if current_index % batch_size == 0:
                start = time.time()
                self.upsert_batch(all_diseases, all_embeddings)
                upsert_time += time.time() - start
                current_index = 0

        # Handling the last batch if it's not full
        if current_index > 0:
            start = time.time()
            self.upsert_batch(all_diseases[:current_index], all_embeddings[:current_index])
            upsert_time += time.time() - start

        logger.info(f"Total time for embedding calculations: {embedding_calc_time}s")
        logger.info(f"Total time for upsert operations: {upsert_time}s")

        return self.disease_weighted_avg_line_graph_embeddings_collection

    def upsert_batch(self, disease_ids, embeddings):
        valid_indices = [i for i, diseases in enumerate(disease_ids) if diseases is not None]
        filtered_ids = [disease_ids[i] for i in valid_indices]
        metadatas = [{"type": "disease"}] * len(disease_ids)
        self.disease_weighted_avg_line_graph_embeddings_collection.upsert(ids=filtered_ids, embeddings=embeddings, metadatas=metadatas)