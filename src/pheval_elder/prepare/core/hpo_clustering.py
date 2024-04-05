import functools

import networkx
from tqdm import tqdm
import pandas as pd
import tarfile
import tempfile
import networkx as nx
from typing import List, Tuple
import wget
import os
import requests
import logging

# from pheval_elder.prepare.elder_core.data_processor import DataProcessor

# Set up logging to file
logging.basicConfig(filename='missing_hpo_terms.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

class HPOClustering():
    def __init__(self):
        # self.cached_all_hps = functools.lru_cache(maxsize=len(self.all_hps))(self.data_processor.hp_embeddings.keys())
        self.closures, self.graph = self.make_hpo_closures_and_graph()
        self.phenotypic_abnormality_id = 'HP:0000118'
        self.all_organs = self.get_organ_systems(self.graph, root_node='HP:0000118')
        # self.cached_organ_systems = functools.lru_cache(maxsize=len(self.all_organs))(self.get_organ_system)

    @functools.lru_cache(maxsize=None)
    def get_organ_system(self, term_id: str):
        try:
            return self._find_highest_parent(term_id=term_id)
        except networkx.exception.NetworkXError:
            logging.info(f"The node {term_id} is not in the digraph.")
            return None

    def make_hpo_closures_and_graph(
            self,
            url="https://kg-hub.berkeleybop.io/kg-obo/hp/2023-04-05/hp_kgx_tsv.tar.gz",
            pred_col="predicate",
            subject_prefixes=["HP:"],
            object_prefixes=["HP:"],
            predicates=["biolink:subclass_of"],
            root_node_to_use="HP:0000118",
            include_self_in_closure=False,
    ) -> (List[Tuple], nx.DiGraph):
        tmpdir = tempfile.TemporaryDirectory()
        tmpfile = tempfile.NamedTemporaryFile().file.name
        wget.download(url, tmpfile)

        this_tar = tarfile.open(tmpfile, "r:gz")
        this_tar.extractall(path=tmpdir.name)

        edge_files = [f for f in os.listdir(tmpdir.name) if "edges" in f]
        if len(edge_files) != 1:
            raise RuntimeError(
                "Didn't find exactly one edge file in {}".format(tmpdir.name)
            )
        edge_file = edge_files[0]

        edges_df = pd.read_csv(os.path.join(tmpdir.name, edge_file), sep="\t")
        if pred_col not in edges_df.columns:
            raise RuntimeError(
                "Didn't find predicate column {} in {} cols: {}".format(
                    pred_col, edge_file, "\n".join(edges_df.columns)
                )
            )

        # get edges of interest
        edges_df = edges_df[edges_df[pred_col].isin(predicates)]
        # get edges involving nodes of interest
        edges_df = edges_df[edges_df["subject"].str.startswith(tuple(subject_prefixes))]
        edges_df = edges_df[edges_df["object"].str.startswith(tuple(object_prefixes))]

        # make into list of tuples
        # note that we are swapping order of edges (object -> subject) so that descendants are leaf terms
        # and ancestors are root nodes (assuming edges are subclass_of edges)
        edges = list(edges_df[["object", "subject"]].itertuples(index=False, name=None))

        # Create a directed graph using NetworkX
        graph = nx.DiGraph(edges)

        # Create a subgraph from the descendants of phenotypic_abnormality
        descendants = nx.descendants(graph, root_node_to_use)
        pa_subgraph = graph.subgraph(descendants)

        def compute_closure(node):
            return set(nx.ancestors(graph, node))

        closures = []

        for node in tqdm(pa_subgraph.nodes(), desc="Computing closures"):
            if include_self_in_closure:
                closures.append((node, "dummy_predicate", node))
            for anc in compute_closure(node):
                closures.append((node, "dummy_predicate", anc))

        return closures, graph

    @functools.lru_cache(maxsize=None)
    def _find_highest_parent(self, term_id: str):
        highest_term = None
        ancestors = nx.ancestors(self.graph, term_id)
        ancestors.discard(self.phenotypic_abnormality_id)

        for ancestor in ancestors:
            if self.phenotypic_abnormality_id in nx.ancestors(self.graph, ancestor):
                if highest_term is None or len(nx.ancestors(self.graph, ancestor)) < len(
                        nx.ancestors(self.graph, highest_term)):
                    highest_term = ancestor

        return highest_term

        #   PA
        #  /  \
        # A    B
        # |    |
        # C    D
        # |
        # E
        #

    # PA represents "Phenotypic abnormality"
    # A, B, C, D, and E are other terms in the graph.
    # Now, let's apply _find_highest_parent to the term E.
    #
    # Ancestors of E: We get {C, A, PA} as the ancestors of E.
    # Discard PA: Now the set becomes {C, A}.
    # Iterate Over Ancestors: We check C and A. Both have PA in their ancestors.
    # Ancestors of C are {A, PA}, and ancestors of A are {PA}.
    # Find Closest: A is closer to PA as it has fewer ancestors. So, A is returned as the highest parent of E.

    # In this method, the initial removal of PA from the set of ancestors is to avoid directly returning PA as
    # the highest parent (as it's presumably a generic or root term).
    # The subsequent checks ensure that we only consider ancestors that eventually lead to PA.

    def get_organ_systems(self, graph, root_node) -> List[str]:
        organ_systems = list(graph.successors(root_node))
        return organ_systems
