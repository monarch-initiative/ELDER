from chromadb.types import Collection
import time
from elder_core.base_service import BaseService


class DiseaseAvgEmbeddingService(BaseService):
    """
        upsert averaged embeddings from hp_embeddings (cached dict from ont_hp collection) that are connected to the
        relevant disease from disease_to_hps (cached dict from hpoa) into the disease_avg_embeddings_collection that
        contains disease and the average embeddings of the correlating hp terms
    """

    def process_data(self) -> Collection:
        if not self.disease_to_hps:
            raise ValueError("disease to hps data is not initialized")
        if not self.disease_avg_embeddings_collection:
            raise ValueError("disease_avg_embeddings collection is not initialized")

        if self.disease_avg_embeddings_collection:
            return self.disease_avg_embeddings_collection

        batch_size = 25
        batch = []
        embedding_calc_time = 0
        upsert_time = 0

        for disease, hps in self.disease_to_hps.items():
            start = time.time()
            average_embedding = self.data_processor.calculate_average_embedding(hps, self.hp_embeddings)
            embedding_calc_time += time.time() - start
            batch.append((disease, average_embedding.tolist()))

            if len(batch) >= batch_size:
                start = time.time()
                self.upsert_batch(batch)
                upsert_time += time.time() - start
                batch = []

        if batch:
            start = time.time()
            self.upsert_batch(batch)
            upsert_time += time.time() - start

        print(f"Total time for embedding calculations: {embedding_calc_time}s")
        print(f"Total time for upsert operations: {upsert_time}s")

        return self.disease_avg_embeddings_collection

    def upsert_batch(self, batch):
        ids = [item[0] for item in batch]
        embeddings = [item[1] for item in batch]
        metadatas = [{"type": "disease"}] * len(batch)
        self.disease_avg_embeddings_collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
