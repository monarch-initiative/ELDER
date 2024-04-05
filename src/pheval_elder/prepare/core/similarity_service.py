from abc import ABC, abstractmethod
from typing import List

import numpy as np


class SimilarityService(ABC):
    @abstractmethod
    def calculate_similarity(self, vector_a: List[float], vector_b: List[float]) -> float:
        pass


class CosineSimilarity(SimilarityService):
    def calculate_similarity(self, vector_a: List[float], vector_b: List[float]) -> float:
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        return dot_product / (norm_a * norm_b)


class L2Distance(SimilarityService):
    def calculate_similarity(self, vector_a: List[float], vector_b: List[float]) -> float:
        # Implement L2 distance calculation
        raise NotImplementedError
