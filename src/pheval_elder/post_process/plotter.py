import matplotlib.pyplot as plt

from pheval_elder.post_process.umap_reducer import UMAPreducer
import plotly.express as px

class DiseaseAverageEmbeddingsPlotter:
    def __init__(self):
        self.reduced_embeddings = UMAPreducer.reduce_dimensions()
        pass

    def plot_embeddings_with_plotly(embeddings, labels=None, title="UMAP Projection of Disease Embeddings"):
        """
        Plots the UMAP-reduced embeddings using Plotly for an interactive plot.

        :param embeddings: 2D NumPy array of reduced embeddings.
        :param labels: Optional labels for each point (e.g., disease names).
        :param title: Title of the plot.
        """
        # Create a DataFrame for Plotly
        import pandas as pd
        df = pd.DataFrame(embeddings, columns=['UMAP Dimension 1', 'UMAP Dimension 2'])
        if labels is not None:
            df['Labels'] = labels

        # Create the Plotly figure
        fig = px.scatter(df, x='UMAP Dimension 1', y='UMAP Dimension 2', text='Labels' if labels is not None else None)
        fig.update_traces(textposition='top center')
        fig.update_layout(title=title, xaxis_title='UMAP Dimension 1', yaxis_title='UMAP Dimension 2')
        fig.show()
