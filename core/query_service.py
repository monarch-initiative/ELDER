from typing import List, Any

from core.chromadb_manager import ChromaDBManager
from core.data_processor import DataProcessor
from core.disease_avg_embedding_service import DiseaseAvgEmbeddingService


class QueryService:
    def __init__(self, data_processor: DataProcessor, db_manager: ChromaDBManager, disease_service: DiseaseAvgEmbeddingService, similarity_strategy=None):
        self.db_manager = db_manager
        self.data_processor = data_processor
        self.similarity_strategy = similarity_strategy
        self.hp_embeddings = data_processor.hp_embeddings  # Dict
        self.disease_service = disease_service

    def query_diseases_by_hpo_terms_using_inbuild_distance_functions(self, hpo_ids: List[str], n_results: int) -> str | list[Any]: # str just for early return
        """
        Queries the 'DiseaseAvgEmbeddings' collection for diseases closest to the average embeddings of given HPO terms.

        :param n_results: number of results for query
        :param hpo_ids: List of HPO term IDs.
        :return: List of diseases sorted by closeness to the average HPO embeddings.
        """
        # need to check that self contains the collection needed here and the dicts !!!!
        avg_embedding = self.data_processor.calculate_average_embedding(hpo_ids, self.hp_embeddings) # self.data_processor
        if avg_embedding is None:
            return "No valid embeddings found for provided HPO terms."

        query_results = self.disease_service.disease_avg_embeddings_collection.query( # self.data_processor?
            query_embeddings=[avg_embedding.tolist()],
            n_results=n_results,  # optional, should be all for
            include=["embeddings", "distances"]  # just distances should also work
        )

        disease_ids = query_results['ids'][0] if 'ids' in query_results and query_results['ids'] else []
        distances = query_results['distances'][0] if 'distances' in query_results and query_results['distances'] else []
        sorted_results = sorted(zip(disease_ids, distances), key=lambda x: x[1])

        return sorted_results

    def query_with_custom_similarity_function(self, data1, data2):
        # Implementation using custom similarity measure
        if self.similarity_strategy:
            return self.similarity_strategy.calculate_similarity(data1, data2)
        else:
            raise ValueError("No similarity strategy provided")
