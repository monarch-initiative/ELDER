from pheval.utils.file_utils import all_files
from pheval.utils.phenopacket_utils import phenopacket_reader, PhenopacketUtil

from pheval_elder.jax_ontology_client import JaxOntologyClient
from pheval_elder.prepare.core.data_processing.data_processor import DataProcessor
from pheval_elder.prepare.core.store.chromadb_manager import ChromaDBManager
from pheval_elder.dis_avg_emb_runner import repo_root, ALL_PHENOPACKETS, Z_PHENOPACKET_TEST, LIRICAL_PHENOPACKETS

if __name__ == "__main__":
    path = repo_root / LIRICAL_PHENOPACKETS
    file_list = all_files(path)
    unique_input_hps = set()

    for file_path in file_list:
        phenopacket = phenopacket_reader(file_path)
        phenopacket_util = PhenopacketUtil(phenopacket)
        observed_phenotypes = phenopacket_util.observed_phenotypic_features()

        for observed_phenotype in observed_phenotypes:
            unique_input_hps.add(observed_phenotype.type.id)

    print("Unique Disease Phenotypes", len(unique_input_hps))
    collection_name = "large3_lrd_hpo_embeddings"
    path = "/Users/ck/Monarch/elder/emb_data/models/large3"
    manager = ChromaDBManager(path=path, collection_name=collection_name)
    data_processor = DataProcessor(manager)
    embeddings_dict = data_processor.hp_embeddings
    disease_dict = data_processor.disease_to_hps

    input_phenotypes_not_in_hp_embeddings = set()
    print("Analyse Input Phenotypes")
    for hpo_id in unique_input_hps:
        if hpo_id not in embeddings_dict:
            print(hpo_id)
            input_phenotypes_not_in_hp_embeddings.add(hpo_id)

    print("Analyse Disease Phenotypes")
    unique_disease_phenotypes = set()
    # for hpo_id in unique_hpo_ids:
    disease_phenotypes = None
    for disease_id, disease_data in disease_dict.items():
        disease_phenotypes = disease_data["phenotypes"]
        for p in disease_phenotypes:
            unique_disease_phenotypes.add(p)
    print("Unique Disease Phenotypes", len(unique_disease_phenotypes))


    disaese_phenotypes_not_in_hp_embeddings = set()
    for id_ in disease_phenotypes:
        if id_ not in embeddings_dict:
            print(id_)
            disaese_phenotypes_not_in_hp_embeddings.add(id_)

    combined_set_of_obsolete_hps = disaese_phenotypes_not_in_hp_embeddings.union(input_phenotypes_not_in_hp_embeddings)

    jax = JaxOntologyClient()
    terms = {}
    for hp in combined_set_of_obsolete_hps:
        new_hp = jax.get_term_by_id(hp)
        terms[hp] = new_hp

    print("TERMS")
    print(terms)
