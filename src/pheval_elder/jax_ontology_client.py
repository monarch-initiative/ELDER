#!/usr/bin/env python3
"""
A Python client for the JAX Ontology REST API (Human Phenotype Ontology).
Demonstrates searching, retrieving terms, and using Pydantic for data validation.
"""

import logging
import requests
from typing import List, Optional, Set, Dict
from pydantic import BaseModel, Field

from pheval_elder.prepare.core.data_processing.data_processor import DataProcessor
from pheval_elder.prepare.core.store.chromadb_manager import ChromaDBManager


class OntologyTerm(BaseModel):
    id: str = Field(..., description="Ontology ID, e.g. HP:0000001")
    label: Optional[str] = Field(None, description="Human-readable label")
    definition: Optional[str] = Field(None, description="Definition of the term")
    synonyms: Optional[List[str]] = Field(default_factory=list, description="List of synonyms")


class JaxOntologyClient:
    def __init__(self, base_url: str = "https://ontology.jax.org/api/hp"):
        self.base_url = base_url
        self.session = requests.Session()

    def get_term_by_id(self, term_id: str) -> OntologyTerm:
        endpoint = f"{self.base_url}/terms/{term_id}"
        response = self.session.get(endpoint, timeout=10)
        self._check_response(response)
        json_data = response.json()
        return OntologyTerm(**json_data)

    @staticmethod
    def _check_response(response: requests.Response) -> None:
        if not response.ok:
            logging.error("HTTP %d - %s", response.status_code, response.text)
            response.raise_for_status()


# -----------------------------
# Main Processing
# -----------------------------
def main():
    old_hpo_ids: List[str] = [
        "HP:0006919",
        "HP:0002355",
        "HP:0030214",
        "HP:0001388",
        "HP:0005692",
        "HP:0001388",
        "HP:0040180",
        "HP:0000735",
        "HP:0040083",
        "HP:0006957",
        "HP:0002355",
        "HP:0001006",
        "HP:0000976",
        "HP:0000368"
    ]

    unique_old_hpo_ids: Set[str] = set(old_hpo_ids)
    logging.info("Unique HPO IDs to process: %s", unique_old_hpo_ids)
    client = JaxOntologyClient()
    mapping: Dict[str, str] = {}

    for old_id in unique_old_hpo_ids:
        try:
            term = client.get_term_by_id(old_id)
            mapping[old_id] = term.id
            logging.info("Mapped %s -> %s", old_id, term.id)
        except Exception as e:
            logging.error("Error retrieving term for %s: %s", old_id, e)

    print("Mapping from old to new HPO IDs:")
    print(mapping)

    #{
    # 'HP:0001388': 'HP:0001382',
    # 'HP:0000735': 'HP:0012760',
    # 'HP:0000368': 'HP:0000358',
    # 'HP:0006957': 'HP:0002505',
    # 'HP:0002355': 'HP:0001288',
    # 'HP:0005692': 'HP:0001382',
    # 'HP:0006919': 'HP:0000718',
    # 'HP:0000976': 'HP:0000964',
    # 'HP:0001006': 'HP:0008070',
    # 'HP:0040083': 'HP:0030051',
    # 'HP:0030214': 'HP:5200321',
    # 'HP:0040180': 'HP:0032152'
    # }

    # 3. Load hp_embeddings using DataProcessor
    collection_name = "large3_lrd_hpo_embeddings"
    path = "/Users/ck/Monarch/elder/emb_data/models/large3"
    manager = ChromaDBManager(path=path, collection_name=collection_name)
    data_processor = DataProcessor(manager)
    embeddings_dict = data_processor.hp_embeddings

    # 4. Check for each mapping if the new id exists in hp_embeddings
    mapping_in_embeddings: Dict[str, str] = {}
    for old_id, new_id in mapping.items():
        if new_id in embeddings_dict:
            mapping_in_embeddings[old_id] = new_id
            logging.info("Found embedding for %s -> %s", old_id, new_id)
        else:
            logging.warning("No embedding found for new id %s (mapped from %s)", new_id, old_id)

    print("Final mapping (only those with embeddings):")
    print(mapping_in_embeddings)


if __name__ == "__main__":
    main()