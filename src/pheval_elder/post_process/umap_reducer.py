import numpy as np

from pheval_elder.post_process.embeddings_aggregator import EmbeddingsAggregator
import umap


class UMAPreducer:
    def __init__(self):
        self.EmbeddingsAggregator = EmbeddingsAggregator()
        self.embeddings = list(self.EmbeddingsAggregator.aggregate_embeddings().values())

    def reduce_dimensions(self):
        # play with init, min_dist and spread
        reducer = umap.UMAP(random_state=42)
        reduced_embeddings = reducer.fit_transform(np.array(self.embeddings))
        return reduced_embeddings
