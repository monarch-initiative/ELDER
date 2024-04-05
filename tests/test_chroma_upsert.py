import random
import time
import unittest
from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from chromadb.api.types import OneOrMany

from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures


class TestChromaUpsert(unittest.TestCase):
    def setUp(self):
        start = time.time()
        self.db_manager = ChromaDBManager(similarity=SimilarityMeasures.COSINE)
        end = time.time()
        end_time_chroma = end - start
        # print(f"CHROMA SETUP = {end_time_chroma}")

    def test_upsert_batch_as_list(self):
        batch_size = 1
        embedding = [random.random() for _ in range(1536)]
        ids = self.get_ids(batch_size)
        self.upsert(ids=ids, embedding=embedding, printing_string="UPSERT AS LIST")

    def test_upsert_batch_as_list_2(self):
        batch_size = 1
        embedding = [random.random() for _ in range(1536)]
        ids = self.get_ids(batch_size)
        self.upsert(ids=ids, embedding=embedding, printing_string="UPSERT AS LIST_2")

    def test_upsert_batch_as_np_empty(self):
        batch_size = 1
        batch = np.empty([0, 1536], float)
        ids = self.get_ids(batch_size)
        new_embedding = np.random.rand(1536)
        embedding = np.append(batch, [new_embedding], axis=0)
        self.upsert(ids=ids, embedding=embedding, printing_string="UPSERT AS NP_EMPTY ARRAY")

    def test_upsert_batch_as_np_zeros(self):
        batch_size = 1
        batch = np.zeros([0, 1536], float)
        ids = self.get_ids(batch_size)
        new_embedding = np.random.rand(1536)
        embedding = np.append(batch, [new_embedding], axis=0)
        self.upsert(ids=ids, embedding=embedding, printing_string="UPSERT AS NP_ZERO ARRAY")

    def test_upsert_multiple_batches_as_list(self):
        batch_size = 100
        batch = []
        ids = self.get_ids(batch_size)

        for i in range(batch_size):
            # batch.append(np.random.rand(1536))
            embedding_as_list = np.random.rand(1536).tolist()
            batch.append(embedding_as_list)

        self.upsert(ids=ids, embedding=batch, printing_string="UPSERT MULTIPLE AS LIST")

    def generate_embedding(self):
        return np.random.rand(1536).tolist()
    def test_upsert_multiple_batches_as_list_parallel(self):
        batch_size = 100
        batch = []
        ids = self.get_ids(batch_size)

        with ThreadPoolExecutor() as executor:
            for index, embedding in executor.map(self.assign_embedding, range(batch_size)):
                batch[index] = embedding

        self.upsert(ids=ids, embedding=batch, printing_string="UPSERT MULTIPLE AS LIST")

    # def test_upsert_multiple_batches_np_empty_without_appending(self):
    #     batch_size = 10000
    #     batch = np.empty((batch_size, 1536), dtype=float)
    #     ids = self.get_ids(batch_size)
    #
    #     for i in range(batch_size):
    #         batch[i] = np.random.rand(1536)
    #
    #     # new_embedding = [np.random.rand(1536) for i in range(batch_size)]
    #     # embedding = np.append(batch, [batch], axis=0)
    #     self.upsert(ids=ids, embedding=batch, printing_string="UPSERT MULTIPLE AS NP_EMPTY ARRAY WITHOUT APPEND")
    @staticmethod
    def assign_embedding(index):
        return index, np.random.rand(1536)

    ## TODO: the assign is in both methods and its wrong look gPT (LAST THING DONE ON FRIDAY EVE_)

    def test_upsert_multiple_batches_np_zeros_without_appending_parallel(self):
        batch_size = 100
        batch = np.zeros((batch_size, 1536), dtype=float)
        ids = self.get_ids(batch_size)

        with ThreadPoolExecutor() as executor:
            # Generate embeddings in parallel
            embeddings = list(executor.map(self.assign_embedding, range(batch_size)))

        # Assign the generated embeddings to the batch array
        for i, embedding in enumerate(embeddings):
            batch[i] = embedding

        self.upsert(ids=ids, embedding=batch, printing_string="UPSERT MULTIPLE AS NP_ZEROS ARRAY WITHOUT APPEND PARALLEL")

    def test_upsert_multiple_batches_np_zeros_without_appending(self):
        batch_size = 10000
        batch = np.zeros((batch_size, 1536), dtype=float)
        ids = self.get_ids(batch_size)

        for i in range(batch_size):
            batch[i] = np.random.rand(1536)

        # embedding = np.append(batch, [batch], axis=0)
        self.upsert(ids=ids, embedding=batch, printing_string="UPSERT MULTIPLE AS NP_ZEROS ARRAY WITHOUT APPEND")

    # def test_upsert_multiple_batches_np_empty_with_append(self):
    #     batch_size = 10000
    #     batch = np.empty((0, 1536), dtype=float)
    #
    #     for i in range(batch_size):
    #         # Append new random embeddings to the NumPy array
    #         new_embedding = np.random.rand(1536)
    #         batch = np.append(batch, [new_embedding], axis=0)
    #
    #     ids = self.get_ids(batch_size)
    #     self.upsert(ids=ids, embedding=batch, printing_string="UPSERT MULTIPLE AS NP_EMPTY WITH APPEND")

    # def test_upsert_multiple_batches_np_zeros_with_append(self):
    #     batch_size = 10000
    #     batch = np.zeros((0, 1536), dtype=float)
    #
    #     for i in range(batch_size):
    #         # Append new random embeddings to the NumPy array
    #         new_embedding = np.random.rand(1536)
    #         batch = np.append(batch, [new_embedding], axis=0)
    #
    #     ids = self.get_ids(batch_size)
    #     self.upsert(ids=ids, embedding=batch, printing_string="UPSERT MULTIPLE AS NP_ZERO WITH APPEND")

    def get_ids(self, batch_size: int):
        batch_ids = ["id_" + str(i) for i in range(0, batch_size)]
        return batch_ids

    def upsert(self, ids: list[str], embedding: Union[OneOrMany[list], OneOrMany[np.ndarray]], printing_string: str):
        start = time.time()
        # col = self.db_manager.create_collection(name="test_upsert_batch") or self.db_manager.get_collection(
        #     name="test_upsert_batch")
        # col = self.db_manager.create_collection(name="testing_more") or self.db_manager.get_collection(
        #     name="testing_more")
        col = self.db_manager.create_collection(name="tesing") or self.db_manager.get_collection(
            name="tesing")
        end = time.time()
        # end_time_colection_get = end - start
        # print(f"get or create = {end_time_colection_get}")
        start = time.time()
        col.upsert(ids=ids, embeddings=embedding)
        end = time.time()
        end_time_upsert = end - start
        print(f"{printing_string} = {end_time_upsert}")

    ## TODO: use np.zeros to allocate fixed space in CPU
    ## TODO: parallel processing

    ## TODO: compare the speed of 1 batch (embedding) to all with np.empty and np.zeros
