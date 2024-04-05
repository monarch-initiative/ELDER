import pandas
import pandas as pd
import numpy as np
import re
import os


class GraphEmbeddingExtractor:
    def __init__(self):
        self.deepwalk_embeddings = None
        self.line_embeddings = None
        self.filtered_deepwalk_df = None
        self.filtered_line_df = None
        self.deepwalk_df = None
        self.line_df = None

    def parse_deep_embeddings(self):
        if self.deepwalk_embeddings is None:

            pattern = re.compile(r'HP:\d+')
            self.deepwalk_df = pd.read_csv("/Users/carlo/Downloads/deepwalk_embedding.csv")

            # self.filtered_deepwalk_df = self.deepwalk_df[self.deepwalk_df.iloc[:, 0].apply(lambda x: pattern.match(x))]

            embeddings_dict = {}
            filtered_rows = []

            for index, row in self.deepwalk_df.iterrows():
                if pattern.match(row[0]):
                    embeddings_dict[row[0]] = np.array(row.values[1:], dtype=np.float32)
                    filtered_rows.append(row)
            self.filtered_deepwalk_df = pd.DataFrame(filtered_rows)
            self.deepwalk_embeddings = embeddings_dict

        return self.deepwalk_embeddings

    def parse_line_embeddings(self) -> dict:
        if self.line_embeddings is None:
            pattern = re.compile(r'HP:\d+')
            self.line_df = pd.read_csv("/Users/carlo/downloads/line_embedding.csv")
            # self.filtered_line_df = self.line_df [self.line_df.df.iloc[:, 0].apply(lambda x: pattern.match(x))]

            embeddings_dict = {}
            filtered_rows = []

            for index, row in self.line_df.iterrows():
                if pattern.match(row[0]):
                    embeddings_dict[row[0]] = np.array(row.values[1:], dtype=np.float32)
                    filtered_rows.append(row)

            self.filtered_line_df = pd.DataFrame(filtered_rows)

            self.line_embeddings = embeddings_dict

        return self.line_embeddings
