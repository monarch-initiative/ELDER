import json
from typing import Dict, List
import pandas as pd
from pathlib import Path
from typing import List, Dict
import pheval
from utils.phenopacket_utils import PhenopacketUtil

from core.OMIMHPOExtractor import OMIMHPOExtractor
from core.hpo_clustering import HPOClustering
import matplotlib.pyplot as plt
import seaborn as sns

class DiseaseDataAnalyzer:
    def __init__(self, tsv_file_path: str):
        self.tsv_file_path = tsv_file_path
        self.data = self._load_data_numeric()
        file_path = "/Users/carlo/PycharmProjects/chroma_db_playground/phenotype.hpoa"
        data = OMIMHPOExtractor.read_data_from_file(file_path)
        self.hpo_clustering = HPOClustering()
        self._disease_to_hps_from_omim = OMIMHPOExtractor.extract_omim_hpo_mappings(data)

    def _load_data(self):
        return pd.read_csv(self.tsv_file_path, sep="\t", header=None,
                           names=["Rank", "FilePath", "Disease", "ExomiserScore", "ElderScore", "ScoreDifference"])

    def _load_data_numeric(self):
        df = pd.read_csv(self.tsv_file_path, sep="\t", header=None, names=["Rank", "FilePath", "Disease", "ExomiserScore", "ElderScore", "ScoreDifference"])
        df['ExomiserScore'] = pd.to_numeric(df['ExomiserScore'], errors='coerce')
        df['ElderScore'] = pd.to_numeric(df['ElderScore'], errors='coerce')
        return df

    def diseases_ranked_higher_in_elder_top_10(self) -> Dict:
        self.data['ExomiserScore'] = pd.to_numeric(self.data['ExomiserScore'], errors='coerce')
        self.data['ElderScore'] = pd.to_numeric(self.data['ElderScore'], errors='coerce')

        filtered = self.data[(self.data['ElderScore'] <= 10) & (self.data['ExomiserScore'] > 10)]
        better_ranked_diseases = dict(zip(filtered['Disease'], filtered['FilePath']))

        return better_ranked_diseases

    def diseases_ranked_higher_in_elder(self) -> List:
        higher_in_elder = self.data[self.data['ElderScore'] > self.data['ExomiserScore']]['Disease'].unique()
        return list(higher_in_elder)

    def compare_phenopackets_with_expected2(self, diseases: List[str]):
        comparison_data = []
        disease_to_phenopacket = self._map_diseases_to_phenopackets()

        for disease in diseases:
            path = disease_to_phenopacket.get(disease)
            if not path:
                continue

            observed_phenotypes = self.get_phenotypes_from_phenopacket(path)
            expected_phenotypes = self._disease_to_hps_from_omim.get(disease, [])

            for phenotype in observed_phenotypes:
                organ_system = self.hpo_clustering.get_organ_system(
                    phenotype)  # assuming hpo_clustering is an instance attribute
                linked_diseases = self._get_linked_diseases(phenotype, self._disease_to_hps_from_omim)
                is_expected = phenotype in expected_phenotypes

                comparison_data.append({
                    'Phenopacket Path': path,
                    'Disease': disease,
                    'HP Term': phenotype,
                    'Is Expected': is_expected,
                    'Organ System': organ_system,
                    'Linked Diseases': linked_diseases
                })

        return pd.DataFrame(comparison_data)

    def _map_diseases_to_phenopackets(self) -> Dict:
        # Assuming 'Disease' and 'FilePath' columns map diseases to phenopacket paths
        return dict(zip(self.data['Disease'], self.data['FilePath']))

    def compare_phenopackets_with_expected(self, phenopacket_paths: List[str]):
        comparison_data = []

        for path in phenopacket_paths:
            observed_phenotypes = self.get_phenotypes_from_phenopacket(path)
            observed_disease = self._extract_disease_from_phenopacket(path)
            expected_phenotypes = self._disease_to_hps_from_omim.get(observed_disease, [])

            for phenotype in observed_phenotypes:
                organ_system = self.hpo_clustering.get_organ_system(phenotype)
                linked_diseases = self._get_linked_diseases(phenotype, self._disease_to_hps_from_omim)
                is_expected = phenotype in expected_phenotypes

                comparison_data.append({
                    'Phenopacket Path': path,
                    'Disease': observed_disease,
                    'HP Term': phenotype,
                    'Is Expected': is_expected,
                    'Organ System': organ_system,
                    'Linked Diseases': linked_diseases
                })

        return pd.DataFrame(comparison_data)

    def _extract_disease_from_phenopacket(self, path: str) -> str:
        with open(path, 'r') as file:
            phenopacket_data = json.load(file)

        phenopacket_util = PhenopacketUtil(phenopacket_data)
        diagnoses = phenopacket_util.diagnoses()

        primary_diagnosis = diagnoses[0] if diagnoses else 'Unknown Disease'
        return primary_diagnosis.disease_identifier if primary_diagnosis else 'Unknown Disease'

    def _get_linked_diseases(self, hp_term: str, disease_to_hps: Dict):
        linked_diseases = [disease for disease, hps in disease_to_hps.items() if hp_term in hps]
        return linked_diseases

    def get_disease_data(self, disease_key: str, rank_type: str):
        if rank_type.lower() == 'exomiser':
            return self.data[self.data['Disease'] == disease_key][['Disease', 'ExomiserScore', 'FilePath']]
        elif rank_type.lower() == 'elder':
            return self.data[self.data['Disease'] == disease_key][['Disease', 'ElderScore', 'FilePath']]
        else:
            raise ValueError("Rank type must be either 'exomiser' or 'elder'")

    def list_all_diseases(self) -> List:
        return self.data['Disease'].unique().tolist()

    def get_file_paths(self):
        return self.data.groupby('Disease')['FilePath'].apply(list).to_dict()

    def compare_files(self, file_paths: List[str], disease_key: str):
        comparison_data = []
        for file_path in file_paths:
            file_data = pd.read_csv(file_path, sep="\t", header=None,
                                    names=["Rank", "FilePath", "Disease", "ExomiserScore", "ElderScore",
                                           "ScoreDifference"])
            disease_data = file_data[file_data['Disease'] == disease_key]
            if not disease_data.empty:
                comparison_data.append({
                    'file': file_path,
                    'exomiser_score': disease_data['ExomiserScore'].iloc[0],
                    'elder_score': disease_data['ElderScore'].iloc[0]
                })
        return pd.DataFrame(comparison_data)

    def process_phenopackets(self, file_path: str) -> List:
        path = Path(file_path)
        file_list = [f for f in path.glob('**/*.json')]
        phenotypic_features = []

        for file in file_list:
            phenopacket = self._read_phenopacket(file)
            phenopacket_util = PhenopacketUtil(phenopacket)
            observed_features = phenopacket_util.observed_phenotypic_features()
            for feature in observed_features:
                phenotypic_features.append((file.name, feature.type.id))

        return phenotypic_features

    def _read_phenopacket(self, file_path: Path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def get_organ_systems(self, hpo_clustering: HPOClustering, phenotype_id: str) -> str:
        return hpo_clustering.get_organ_system(phenotype_id)

    def map_disease_to_hp_terms(self, omim_hpo_extractor: OMIMHPOExtractor) -> Dict[str, List[str]] :
        file_path = "/Users/carlo/PycharmProjects/chroma_db_playground/phenotype.hpoa"
        data = omim_hpo_extractor.read_data_from_file(file_path)
        return omim_hpo_extractor.extract_omim_hpo_mappings(data)

    def analyze_hp_term_frequency(self, disease_to_hp_mapping: Dict) -> Dict:
        hp_term_frequency = {}
        for hp_terms in disease_to_hp_mapping.values():
            for term in hp_terms:
                hp_term_frequency[term] = hp_term_frequency.get(term, 0) + 1
        return hp_term_frequency

    # def visualize_hp_term_frequency(self, hp_term_frequency: Dict, top_n: int = 10):
    #     top_terms = sorted(hp_term_frequency.items(), key=lambda x: x[1], reverse=True)[:top_n]
    #     terms, frequencies = zip(*top_terms)
    #
    #     plt.figure(figsize=(10, 6))
    #     sns.barplot(list(terms), list(frequencies))
    #     plt.xticks(rotation=45)
    #     plt.xlabel('HP Terms')
    #     plt.ylabel('Frequency')
    #     plt.title('Top HP Terms by Frequency')
    #     plt.show()

    def get_phenotypes_from_phenopacket(self, phenopacket_path: str) -> List:
        phenopacket = self._read_phenopacket(Path(phenopacket_path))
        phenopacket_util = PhenopacketUtil(phenopacket)
        return [feature.type.id for feature in phenopacket_util.observed_phenotypic_features()]

    def compare_phenotypes_across_phenopackets(self, disease_key: str, phenopacket_paths: List[str]):
        all_phenotypes = {path: self.get_phenotypes_from_phenopacket(path) for path in phenopacket_paths}
        unique_phenotypes = set.union(*map(set, all_phenotypes.values())) - set.intersection(
            *map(set, all_phenotypes.values()))

        return unique_phenotypes, all_phenotypes


    def get_phenotypes_for_diseases(self, diseases, phenopacket_paths):
        disease_phenotypes = {}
        for disease in diseases:
            all_phenotypes = []
            for path in phenopacket_paths:
                if self._extract_disease_from_phenopacket(path) == disease:
                    phenotypes = self.get_phenotypes_from_phenopacket(path)
                    all_phenotypes.extend(phenotypes)

            # Remove duplicates
            all_phenotypes = list(set(all_phenotypes))
            disease_phenotypes[disease] = all_phenotypes

        return disease_phenotypes

    # def analyze_impact_on_rankings(self, disease_key: str, phenopacket_paths: List[str], tsv_file_paths: List[str]):
    #     unique_phenotypes, all_phenotypes = self.compare_phenotypes_across_phenopackets(disease_key, phenopacket_paths)
    #     ranking_impacts = {}
    #
    #     for tsv_path in tsv_file_paths:
    #         self.tsv_file_path = tsv_path
    #         ranking_data = self.get_disease_data(disease_key, 'both')
    #         for phenotype in unique_phenotypes:
    #             present_in_phenopacket = any(phenotype in phenotypes for phenotypes in all_phenotypes.values())
    #             ranking_impact = self._analyze_ranking_for_phenotype(ranking_data, phenotype, present_in_phenopacket)
    #             ranking_impacts[phenotype] = ranking_impact
    #
    #     return ranking_impacts
    #
    # def _analyze_ranking_for_phenotype(self, ranking_data, phenotype, present_in_phenopacket):
    #     ranking_impact = {
    #         'presence': None,
    #         'absence': None,
    #         'difference': None
    #     }
    #
    #     if present_in_phenopacket:
    #         ranking_impact['presence'] = ranking_data['Rank'].mean()  # Example calculation
    #     else:
    #         ranking_impact['absence'] = ranking_data['Rank'].mean()  # Example calculation
    #
    #     if ranking_impact['presence'] is not None and ranking_impact['absence'] is not None:
    #         ranking_impact['difference'] = ranking_impact['presence'] - ranking_impact['absence']
    #
    #     return ranking_impact


