from typing import Dict, List
import numpy as np
from umap import UMAP
import pandas as pd
import plotly.express as px

from elder_core.chromadb_manager import ChromaDBManager
from elder_core.data_processor import DataProcessor
from utils.similarity_measures import SimilarityMeasures


class DiseaseEmbeddingPlotter:
    def __init__(self):
        self.df_3d = None
        self.df_2d = None
        self.manager = ChromaDBManager(SimilarityMeasures.COSINE)
        self.data_processor = DataProcessor(db_manager=self.manager)
        self.disease_to_hps = self.data_processor.disease_to_hps_from_omim
        self.hp_to_embedding = self.data_processor.hp_embeddings

    def aggregate_embeddings(self) -> Dict:
        disease_average_embeddings = {}
        for disease, hp_terms in self.disease_to_hps.items():
            embeddings = [self.hp_to_embedding[hp]['embeddings'] for hp in hp_terms if hp in self.hp_to_embedding]
            if embeddings and all(isinstance(e, (list, np.ndarray)) for e in embeddings):
                disease_average_embeddings[disease] = np.mean(embeddings, axis=0)
            else:
                print(f"Invalid data structure for embeddings of disease {disease}")

        return disease_average_embeddings

    def reduce_dimensions_2d(self, embeddings) -> List:
        reducer = UMAP(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(np.array(embeddings))
        return reduced_embeddings

    def reduce_dimensions_3d(self, embeddings) -> List:
        if not embeddings:
            print("Empty embeddings provided to UMAP.")
            return None
        reducer = UMAP(n_components=3, random_state=42)  # Set n_components to 3 for 3D
        reduced_embeddings = reducer.fit_transform(np.array(embeddings))
        return reduced_embeddings

    def plot_embeddings_with_plotly_2d(self, embeddings, labels=None, title="UMAP Projection of Disease Embeddings"):
        df = pd.DataFrame(embeddings, columns=['UMAP Dimension 1', 'UMAP Dimension 2'])
        if labels is not None:
            df['Labels'] = labels
        fig = px.scatter(df, x='UMAP Dimension 1', y='UMAP Dimension 2', hover_data=['Labels'])
        fig.update_traces(textposition='top center')
        fig.update_layout(title=title, xaxis_title='UMAP Dimension 1', yaxis_title='UMAP Dimension 2')
        fig.show()

    def plot_embeddings_with_plotly_3d(self, embeddings, labels=None, title="3D UMAP Projection of Disease Embeddings"):
        if embeddings is None:
            print("No embeddings to plot.")
            return

        df = pd.DataFrame(embeddings, columns=['UMAP Dimension 1', 'UMAP Dimension 2', 'UMAP Dimension 3'])
        if labels is not None:
            df['Labels'] = labels

        fig = px.scatter_3d(df, x='UMAP Dimension 1', y='UMAP Dimension 2', z='UMAP Dimension 3', hover_data=['Labels'])
        fig.update_layout(title=title)
        fig.show()

    def execute(self):
        aggregated_embeddings = self.aggregate_embeddings()
        embeddings_list = list(aggregated_embeddings.values())
        labels = list(aggregated_embeddings.keys())
        reduced_embeddings = self.reduce_dimensions_2d(embeddings_list)

        self.df_2d = pd.DataFrame(reduced_embeddings, columns=['Dim1', 'Dim2'])
        self.df_2d['Labels'] = labels

        reduced_embeddings_3d = self.reduce_dimensions_3d(embeddings_list)

        self.df_3d = pd.DataFrame(reduced_embeddings_3d, columns=['Dim1', 'Dim2', 'Dim3'])
        self.df_3d['Labels'] = labels

        self.plot_embeddings_with_plotly_2d(reduced_embeddings, labels=labels)
        self.plot_embeddings_with_plotly_3d(reduced_embeddings_3d, labels=labels)

    # use to retrieve labels for specific area, check phenotypes, should be similar
    def get_cluster_labels(self, x_bounds, y_bounds):
        cluster_df = self.df_2d[
            (self.df_2d['Dim1'] >= x_bounds[0]) &
            (self.df_2d['Dim1'] <= x_bounds[1]) &
            (self.df_2d['Dim2'] >= y_bounds[0]) &
            (self.df_2d['Dim2'] <= y_bounds[1])
        ]
        return cluster_df['Labels'].tolist()

