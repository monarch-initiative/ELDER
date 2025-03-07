HPO_ID_MAPPING = {
    'HP:0000735': 'HP:0012760',
    'HP:0001388': 'HP:0001382',
    'HP:0000368': 'HP:0000358',
    'HP:0030214': 'HP:5200321',
    'HP:0040180': 'HP:0032152',
    'HP:0006957': 'HP:0002505',
    'HP:0000976': 'HP:0000964',
    'HP:0040083': 'HP:0030051',
    'HP:0005692': 'HP:0001382',
    'HP:0001006': 'HP:0008070',
    'HP:0002355': 'HP:0001288',
    'HP:0006919': 'HP:0000718'
}

def update_hpo_id(hpo_id: str) -> str:
    """
    Given an HPO ID, return the updated ID if it exists in the mapping and is not None;
    otherwise, return the original HPO ID.
    """
    updated = HPO_ID_MAPPING.get(hpo_id)
    return updated if updated is not None else hpo_id