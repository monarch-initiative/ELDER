import time

import numpy as np
from chromadb.types import Collection

from pheval_elder.prepare.core.base_service import BaseService


# from pheval_elder.prepare.elder_core.base_service import BaseService


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
            print("HP Embeddings collection early return, cause already initialized!")
            return self.hp_embeddings_collection

        batch_size = 100
        batch = []
        upsert_time = 0
        # make it a np array before and input this instead getting data from dict
        # np_array = self.hp_embeddings.items()

        for hp_id, data in self.hp_embeddings.items():
            embedding_list = np.array(data["embeddings"])
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
