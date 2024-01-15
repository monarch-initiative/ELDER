from typing import List, Any

from core.chromadb_manager import ChromaDBManager
from core.data_processor import DataProcessor
from core.disease_avg_embedding_service import DiseaseAvgEmbeddingService


from typing import Any, List
from chromadb.types import Collection

from core import hpo_clustering
from core.chromadb_manager import ChromaDBManager
from core.data_processor import DataProcessor
from core.disease_avg_embedding_service import DiseaseAvgEmbeddingService
from core.disease_clustered_embedding_service import DiseaseClusteredEmbeddingService


class QueryService:
    def __init__(
            self,
            data_processor: DataProcessor,
            db_manager: ChromaDBManager,
            disease_service: DiseaseAvgEmbeddingService,
            disease_organ_service: DiseaseClusteredEmbeddingService,
            similarity_strategy=None,
    ):
        self.db_manager = db_manager
        self.data_processor = data_processor
        self.similarity_strategy = similarity_strategy
        self.hp_embeddings = data_processor.hp_embeddings  # Dict
        self.disease_service = disease_service
        self.hpo_clustering = hpo_clustering
        self.disease_organ_service = disease_organ_service

    def query_diseases_using_organ_syst_embeddings(self, hpo_ids: List[str], n_results: int = 10) -> list[Any]:
        """
        Queries the 'DiseaseClusteredEmbeddings' collection for diseases closest to the clustered embeddings of given HPO terms.

        :param hpo_ids: List of HPO term IDs.
        :param n_results: Optional number of results to return. Returns all if None.
        :return: List of diseases sorted by closeness to the clustered HPO embeddings.
        """
        patient_embedding = self.disease_organ_service.compute_organ_embeddings(hpo_terms=hpo_ids)

        query_params = {
            "query_embeddings": [patient_embedding.tolist()],
            "include": ["embeddings", "distances"],
            "n_results": n_results
        }
        # estimated_total_query_results = self.disease_organ_service.clustered_embeddings_collection.get(
        #         include=['embeddings'])
        # self.max_results(query_params=query_params,
        #                  n_results=n_results,
        #                  col=self.disease_organ_service.clustered_embeddings_collection,
        #                  estimated_total_query_results=estimated_total_query_results)
        query_results = self.disease_organ_service.clustered_embeddings_collection.query(**query_params)
        sorted_results = self.process_query_results(query_results=query_results)
        return sorted_results

    def query_diseases_by_hpo_terms_using_inbuild_distance_functions(self, hpo_ids: List[str], n_results: int = None) -> \
            list[Any]:
        """
        Queries the 'DiseaseAvgEmbeddings' collection for diseases closest to the average embeddings of given HPO terms.

        :param hpo_ids: List of HPO term IDs.
        :param n_results: Optional number of results to return. Returns all if None.
        :return: List of diseases sorted by closeness to the average HPO embeddings.
        """
        avg_embedding = self.data_processor.calculate_average_embedding(hpo_ids, self.hp_embeddings)
        if avg_embedding is None:
            raise ValueError("No valid embeddings found for provided HPO terms.")

        query_params = {
            "query_embeddings": [avg_embedding.tolist()],
            "include": ["embeddings", "distances"]
        }

        estimated_total_query_results = self.disease_service.disease_avg_embeddings_collection.get(
                include=['embeddings'])
        self.max_results(query_params=query_params,
                         n_results=n_results,
                         col=self.disease_service.disease_avg_embeddings_collection,
                         estimated_total_query_results=estimated_total_query_results)
        query_results = self.disease_service.disease_avg_embeddings_collection.query(**query_params)
        sorted_results = self.process_query_results(query_results=query_results)
        return sorted_results

    def max_results(self, query_params, n_results, col: Collection, estimated_total_query_results):
        if n_results is None:
            estimated_length = len(estimated_total_query_results["embeddings"])
            print(f"Estimated length (n_results) == {estimated_length}")
            max_n_results = self.binary_search_max_results(query_params, 11700, estimated_length, col=col)
            query_params["n_results"] = max_n_results
            print(f"Using max safe n_results: {max_n_results}")
        else:
            query_params["n_results"] = n_results

    def binary_search_max_results(self, query_params, lower_bound, upper_bound, col: Collection):
        max_safe_value = lower_bound

        while lower_bound < upper_bound - 1:
            mid_point = (lower_bound + upper_bound) // 2
            query_params['n_results'] = mid_point

            try:
                # query_results = self.disease_service.disease_avg_embeddings_collection.query(**query_params)
                query_results = col.query(**query_params)
                max_safe_value = mid_point
                lower_bound = mid_point
            except RuntimeError as e:
                upper_bound = mid_point

        # Verification step: test values around max_safe_value to ensure it's the highest safe value.
        for test_value in range(max_safe_value - 1, max_safe_value + 2):
            if test_value <= 0:
                continue  # Skip non-positive values
            query_params['n_results'] = test_value
            try:
                col.query(**query_params)
                # self.disease_service.disease_avg_embeddings_collection.query(**query_params)
                # self.disease_organ_service.clustered_embeddings_collection.query(**query_params)
                max_safe_value = test_value  # Update max_safe_value if this higher value is also safe
            except RuntimeError as e:
                break

        return max_safe_value

    @staticmethod
    def process_query_results(query_results):
        disease_ids = query_results['ids'][0] if 'ids' in query_results and query_results['ids'] else []
        distances = query_results['distances'][0] if 'distances' in query_results and query_results['distances'] else []
        # labels = query_results['labels'][0] if 'labels' in query_results and query_results[
        #     'labels'] else []  # Fetching labels
        sorted_results = sorted(zip(disease_ids, distances), key=lambda x: x[1])  # remember to add label if needed
        return sorted_results

    def query_with_custom_similarity_function(self, data1, data2):
        if self.similarity_strategy:
            return self.similarity_strategy.calculate_similarity(data1, data2)
        else:
            raise ValueError("No similarity strategy provided")
