from typing import Dict

from core.base_service import BaseService
from core.chromadb_manager import ChromaDBManager
from core.data_processor import DataProcessor
from chromadb.types import Collection
import time
import numpy as np

class HPEmbeddingService(BaseService):
    def process_data(self) -> Collection:
        """
            upsert hps and embeddings into hp_embeddings collection created by chromadbmanager
        """
        if not self.hp_embeddings:
            raise ValueError("HP embeddings data is not initialized")
        if not self.hp_embeddings_collection:
            raise ValueError("HP embeddings collection is not initialized")
        if self.hp_embeddings_collection:
            return self.hp_embeddings_collection

        # if self.hp_embeddings_collection is not None:
        batch_size = 25
        batch = []
        upsert_time = 0
        # make it a np array before and input this instead getting data from dict
        np_array = self.hp_embeddings.items()

        for hp_id, data in self.hp_embeddings.items():
            embedding_list = np.array(data['embeddings'])
            batch.append((hp_id, embedding_list.tolist()))

            if len(batch) >= batch_size:
                start = time.time()
                self.upsert_batch(batch)
                upsert_time += time.time() - start
                batch = []

        if batch:
            start = time.time()
            self.upsert_batch(batch)
            upsert_time += time.time() - start

        print(f"Total time for upsert operations: {upsert_time}s")

        return self.hp_embeddings_collection

    def upsert_batch(self, batch):
        ids = [item[0] for item in batch]
        embeddings = [item[1] for item in batch]
        metadatas = [{"type": "HP"}] * len(batch)
        self.hp_embeddings_collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)