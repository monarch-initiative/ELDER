import logging
import time

import numpy as np
from chromadb.types import Collection
from tqdm import tqdm

from pheval_elder.prepare.core.base_service import BaseService
from pheval_elder.prepare.core.data_processor import DataProcessor
from pheval_elder.prepare.core.graph_data_processor import GraphDataProcessor

# from pheval_elder.prepare.elder_core.base_service import BaseService
# from pheval_elder.prepare.elder_core.data_processor import DataProcessor
# from pheval_elder.prepare.elder_core.graph_data_processor import GraphDataProcessor

logger = logging.getLogger(__name__)


class GraphLineAverageEmbeddingService(BaseService):
    def __int__(self, data_processor: DataProcessor, graph_data_processor: GraphDataProcessor):
        super().__init__(
            data_processor=data_processor,
            graph_data_processor=graph_data_processor
        )

    def process_data(self) -> Collection:
        if not self.graph_data_processor.line_graph_embeddings:
            raise ValueError("Line Graph Embeddings Dictionary is not initialized.")
        if not self.disease_to_hps:
            raise ValueError("DiseaseToHPs Dictionary is not initialized.")
        # if self.disease_avg_line_graph_embeddings_collection:
        #     print("Average Line Graph Embeddings collection already initialized!")
        #     return self.disease_avg_line_graph_embeddings_collection

        num_diseases = len(self.disease_to_hps)
        batch_size = 100
        current_index = 0
        batch_embeddings = np.zeros((batch_size, 100))
        batch_diseases = np.empty(batch_size, dtype=object) # if 0 : ValueError: Expected ID to be a str, got 0


        embedding_calc_time = 0
        upsert_time = 0

        for disease, hps in tqdm(self.disease_to_hps.items(), total=num_diseases):
            start = time.time()
            average_embedding = self.graph_data_processor.calculate_average_line_graph_embeddings(hps=hps)
            embedding_calc_time += time.time() - start

            batch_embeddings[current_index] = average_embedding
            batch_diseases[current_index] = disease
            current_index += 1

            if current_index % batch_size == 0:
                self.upsert_batch(batch_diseases, batch_embeddings)
                current_index = 0

        # if there is a rest batch
        if current_index > 0:
            start = time.time()
            self.upsert_batch(batch_diseases[:current_index], batch_embeddings[:current_index])
            upsert_time += time.time() - start

        logger.info(f"Total time for embedding calculations (avg): {embedding_calc_time}s")
        logger.info(f"Total time for upsert operations (avg): {upsert_time}s")

        return self.disease_avg_line_graph_embeddings_collection

    def upsert_batch(self, disease_ids, embeddings):
        valid_indices = [i for i, diseases in enumerate(disease_ids) if diseases is not None]
        filtered_ids = [disease_ids[i] for i in valid_indices]
        metadatas = [{"type": "disease"}] * len(disease_ids)
        self.disease_avg_line_graph_embeddings_collection.upsert(ids=filtered_ids, embeddings=embeddings,
                                                                 metadatas=metadatas)
