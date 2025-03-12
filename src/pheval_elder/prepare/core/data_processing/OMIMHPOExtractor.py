import json
import os
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Dict, Union, Optional

from pheval_elder.prepare.core.utils.obsolete_hp_mapping import update_hpo_id


class OMIMHPOExtractor:
    @staticmethod
    def extract_omim_hpo_mappings_default(data) -> Dict[str, Dict[str, Union[str, list]]]:
        """
        Extracts OMIM to HPO mappings from the provided data.

        :param data: String containing the data with OMIM and HPO information.
        :return: Dictionary with OMIM IDs as keys and dictionaries containing disease name and phenotype lists.
        """
        omim_hpo_dict = defaultdict(lambda: {"disease_name": None, "phenotypes": []})
        lines = data.split('\n')
        header_skipped = False

        for line in lines:
            if line.startswith("#"):
                continue

            if "database_id" in line and "disease_name" in line and "qualifier" in line and "hpo_id" in line:
                header_skipped = True
                continue
            if not header_skipped:
                continue

            parts = line.split('\t')
            if len(parts) < 8:
                continue

            omim_id, disease_name, hpo_id = parts[0].strip(), parts[1].strip(), parts[3].strip()

            updated_id = update_hpo_id(hpo_id)
            omim_hpo_dict[omim_id]['phenotypes'].append(updated_id)
            omim_hpo_dict[omim_id]['disease_name'] = disease_name

        final_omim_hpo_dict = {k: {"disease_name": v["disease_name"], "phenotypes": sorted(v["phenotypes"])} 
                              for k, v in omim_hpo_dict.items()}

        return final_omim_hpo_dict

    @staticmethod
    def extract_omim_hpo_mappings_with_frequencies_1(data) -> Dict[str, Dict[str, Union[str, Dict[str, float]]]]:
        """
        Extracts OMIM to HPO mappings from the provided data with frequency information.

        :param data: String containing the data with OMIM and HPO information.
        :return: Dictionary with OMIM IDs as keys and dictionaries containing disease name and phenotype frequencies.
        """
        omim_hpo_dict = defaultdict(lambda: {"disease_name": None, "phenotypes_and_frequencies": {}})
        lines = data.split("\n")
        header_found = False
        # Frequency mapping for special HPO terms referring to freq in DAG
        # TODO: those are ranges, i hardcoded for specific values inside those ranges, adapt ...
        special_frequencies = {
            "HP:0040281": 80.0,  # Very frequent
            "HP:0040283": 20.0,  # Occasional
            "HP:0040280": 100.0,  # Obligate
            "HP:0040285": 0.0,   # Excluded
            "HP:0040282": 50.0,  # Frequent
            "HP:0040284": 1.0    # Very rare
        }

        for line in lines:
            if line.startswith("#"):
                continue
            if "database_id" in line and "disease_name" in line and "qualifier" in line and "hpo_id" in line:
                header_found = True
                continue
            if not header_found:
                continue

            parts = [part.strip() for part in line.split("\t")]

            if len(parts) < 8:
                continue

            omim_id, disease_name, qualifier, hpo_id, frequency = parts[0], parts[1], parts[2], parts[3], parts[7]
            hpo_id = update_hpo_id(hpo_id)
            if not omim_id or not hpo_id:
                continue

            # Determine frequency as proportion
            frequency_proportion = 0.5  # Default value
            
            if frequency in special_frequencies:
                frequency_proportion = special_frequencies[frequency] / 100
            elif '/' in frequency:
                numerator, denominator = map(float, frequency.split('/'))
                frequency_proportion = numerator / denominator

            if qualifier != "NOT":
                omim_hpo_dict[omim_id]['phenotypes_and_frequencies'][hpo_id] = frequency_proportion
                omim_hpo_dict[omim_id]['disease_name'] = disease_name

        return omim_hpo_dict

    @staticmethod
    def read_data_from_file(file_path: Union[str, Path]) -> str:
        """
        Reads data from a file at the given file path.

        :param file_path: Path to the file to read.
        :return: String containing the file's content.
        """
        with open(file_path, 'r') as file:
            data = file.read()
        return data

    @staticmethod
    def save_results_as_pretty_json_string(data: Dict, outfile: str, output_dir: Optional[str] = None) -> None:
        """
        Saves the given Dictionary in a nicely formatted JSON file.
        
        :param data: The dictionary data to save
        :param outfile: The filename to save to
        :param output_dir: Optional directory to save to (defaults to current directory)
        """
        if output_dir is None:
            output_dir = "."
        result_path = os.path.join(output_dir, outfile)
        dump = json.dumps(data, sort_keys=True, indent=4)
        with open(result_path, "w") as file:
            file.write(dump)

