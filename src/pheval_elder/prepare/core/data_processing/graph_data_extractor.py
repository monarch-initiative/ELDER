import pandas as pd
import numpy as np
import re
import os


class GraphEmbeddingExtractor:
    def __init__(self):
        self.deepwalk_embeddings = None
        self.line_embeddings = None

    def parse_deep_embeddings(self):
        if self.deepwalk_embeddings is None:

            pattern = re.compile(r'HP:\d+')
            df = pd.read_csv("/Users/carlo/Downloads/deepwalk_embedding.csv")

            embeddings_dict = {}

            for index, row in df.iterrows():
                if pattern.match(row[0]):
                    embeddings_dict[row[0]] = row.values[1:]

            self.deepwalk_embeddings = embeddings_dict

        return self.deepwalk_embeddings

    def parse_line_embeddings(self) -> dict:
        if self.line_embeddings is None:
            pattern = re.compile(r'HP:\d+')
            df = pd.read_csv("/Users/carlo/downloads/line_embedding.csv")

            embeddings_dict = {}

            for index, row in df.iterrows():
                if pattern.match(row[0]):
                    embeddings_dict[row[0]] = row.values[1:]
            self.line_embeddings = embeddings_dict

        return self.line_embeddings