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

    # def query_diseases_by_hpo_terms_using_inbuild_distance_functions(self, hpo_ids: List[str], n_results: int = None) -> str | list[Any]: # str just for early return
    #     """
    #     Queries the 'DiseaseAvgEmbeddings' collection for diseases closest to the average embeddings of given HPO terms.
    #
    #     :param n_results: number of results for query
    #     :param hpo_ids: List of HPO term IDs.
    #     :return: List of diseases sorted by closeness to the average HPO embeddings.
    #     """
    #     # need to check that self contains the collection needed here and the dicts !!!!
    #     avg_embedding = self.data_processor.calculate_average_embedding(hpo_ids, self.hp_embeddings) # self.data_processor
    #     if avg_embedding is None:
    #         return "No valid embeddings found for provided HPO terms."
    #
    #     # if self.disease_service.disease_avg_embeddings_collection:
    #     #     return self.disease_service.disease_avg_embeddings_collection
    #
    #     if n_results is None:
    #         n_results = len(self.disease_service.disease_avg_embeddings_collection.get(include=['embeddings']))
    #         l = len(n_results['emebddings'])
    #         print(f"angezeigte results == {l}")
    #
    #         query_params = {
    #             "query_embeddings": [avg_embedding.tolist()],
    #             "include": ["embeddings", "distances"],
    #             "n_results": n_results
    #         }
    #
    #         query_results = self.disease_service.disease_avg_embeddings_collection.query(**query_params)
    #
    #         disease_ids = query_results['ids'][0] if 'ids' in query_results and query_results['ids'] else []
    #         distances = query_results['distances'][0] if 'distances' in query_results and query_results['distances'] else []
    #         sorted_results = sorted(zip(disease_ids, distances), key=lambda x: x[1])
    #         print("done")
    #         c = 1
    #         for i in sorted_results:
    #             if c < 1:
    #                 print(i)
    #                 c += 1
    #
    #         return sorted_results

    def query_diseases_by_hpo_terms_using_inbuild_distance_functions(self, hpo_ids: List[str], n_results: int = None) -> \
    list[Any]:
        """
        Queries the 'DiseaseAvgEmbeddings' collection for diseases closest to the average embeddings of given HPO terms.

        :param hpo_ids: List of HPO term IDs.
        :param n_results: Optional number of results to return. Returns all if None.
        :return: List of diseases sorted by closeness to the average HPO embeddings.
        """
        print("n_results at default  = :" + f"{n_results}")
        avg_embedding = self.data_processor.calculate_average_embedding(hpo_ids, self.hp_embeddings)
        if avg_embedding is None:
            raise ValueError("No valid embeddings found for provided HPO terms.")

        query_params = {
            "query_embeddings": [avg_embedding.tolist()],
            "include": ["embeddings", "distances"]
        }

        if n_results is None:
            estimated_total = self.disease_service.disease_avg_embeddings_collection.get(include=['metadatas'])
            estimated_length = len(estimated_total["metadatas"]) #12468 - 1
            print(f"Estimated length (n_results) == {estimated_length}")
            max_n_results = self.binary_search_max_results(query_params, 11700, estimated_length)
            query_params["n_results"] = max_n_results
            print(f"Using max safe n_results: {max_n_results}")
        else:
            query_params["n_results"] = n_results

        print("1")
        query_results = self.disease_service.disease_avg_embeddings_collection.query(**query_params)
        print("2")
        disease_ids = query_results['ids'][0] if 'ids' in query_results and query_results['ids'] else []
        distances = query_results['distances'][0] if 'distances' in query_results and query_results['distances'] else []
        # labels = query_results['labels'][0] if 'labels' in query_results and query_results[
        #     'labels'] else []  # Fetching labels
        sorted_results = sorted(zip(disease_ids, distances), key=lambda x: x[1]) # remember to add label
        print("Returning result from query service")
        return sorted_results

    def query_with_custom_similarity_function(self, data1, data2):
        # Implementation using custom similarity measure
        if self.similarity_strategy:
            return self.similarity_strategy.calculate_similarity(data1, data2)
        else:
            raise ValueError("No similarity strategy provided")

    def binary_search_max_results(self, query_params, lower_bound, upper_bound):
        max_safe_value = lower_bound

        while lower_bound < upper_bound - 1:
            mid_point = (lower_bound + upper_bound) // 2
            query_params['n_results'] = mid_point

            try:
                query_results = self.disease_service.disease_avg_embeddings_collection.query(**query_params)
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
                self.disease_service.disease_avg_embeddings_collection.query(**query_params)
                max_safe_value = test_value  # Update max_safe_value if this higher value is also safe
            except RuntimeError as e:
                break

        return max_safe_value
