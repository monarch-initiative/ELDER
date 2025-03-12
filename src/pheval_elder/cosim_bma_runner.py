"""
Best Match Runner (cosine Best Match Average Comparison).

This module provides a runner for analyzing phenotype sets using
the best match algorithm (term-set pairwise comparison).
"""

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Iterator, Union

from pheval.post_processing.post_processing import PhEvalDiseaseResult, generate_pheval_result
from pheval.utils.file_utils import all_files
from pheval.utils.phenopacket_utils import PhenopacketUtil, phenopacket_reader
from tqdm import tqdm

from pheval_elder.base_runner import BaseElderRunner
from pheval_elder.prepare.config.config_loader import load_config_path
from pheval_elder.prepare.config.unified_config import RunnerType
from pheval_elder.prepare.core.utils.obsolete_hp_mapping import update_hpo_id


@dataclass
class BestMatchRunner(BaseElderRunner):
    """
    Runner for analyzing phenotype sets using the best match algorithm.
    
    This runner uses term-set pairwise comparison to find matches between
    phenotype sets and diseases.
    """

    def run(self) -> None:
        """
        Run the best match analysis on all phenotype sets.
        
        This method:
        1. Reads phenopackets from the specified directory
        2. Extracts observed phenotypes from each phenopacket
        3. Runs the term-set pairwise comparison analysis
        4. Processes the results
        """
        total_phenopackets = int(self.elder_runner.nr_of_phenopackets)
        # print(f"Total PhenoPackets: {total_phenopackets}")
        
        path = self.get_phenopackets_dir(self.config)
        # print(f"Reading phenopackets from {path}")
        file_list = all_files(path)
        
        phenotype_sets = []
        file_names = []

        # start_time = time.time()
        # print("Collecting phenotype sets...")
        
        for i, file_path in tqdm(enumerate(file_list, start=1), total=total_phenopackets):
            phenopacket = phenopacket_reader(file_path)
            phenopacket_util = PhenopacketUtil(phenopacket)
            observed_phenotypes = phenopacket_util.observed_phenotypic_features()
            observed_phenotypes_hpo_ids = [
                update_hpo_id(observed_phenotype.type.id) for observed_phenotype in observed_phenotypes
            ]
            file_names.append(file_path.name)
            phenotype_sets.append(observed_phenotypes_hpo_ids)

        if self.elder_runner is not None and self.elder_runner.strategy == "tpc":
            all_results = self.elder_runner.tcp_analysis_optimized(
                phenotype_sets, 
                self.config.runner.nr_of_results
            )
            
            for file_name, result_set in zip(file_names, all_results):
                if result_set:
                    self.current_file_name = file_name
                    self.results = result_set
                    self.post_process()
        else:
            raise RuntimeError(f"Invalid strategy: {self.elder_runner.strategy}")

        # end_time = time.time()
        # print(f"Total processing time: {end_time - start_time:.2f} seconds")

    def post_process(self) -> None:
        """
        Process the results of the analysis.
        
        This method:
        1. Creates the output directory structure
        2. Generates the PhEval result files
        3. Moves the result files to the appropriate directory
        """
        if self.input_dir_config.disease_analysis and self.results:
            output_file_name = f"{self.current_file_name}"
            self.tmp_dir = self.pheval_disease_results_dir / "pheval_disease_results/"
            dest_dir = self.pheval_disease_results_dir / self.elder_runner.results_dir_name / self.elder_runner.results_sub_dir
            
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            generate_pheval_result(
                pheval_result=self.results,
                sort_order_str="DESCENDING",  # Descending for TPC, ascending for others
                output_dir=self.pheval_disease_results_dir,
                tool_result_path=Path(output_file_name),
            )
            
            for file in self.tmp_dir.iterdir():
                # print(f"Moving file {file} to {dest_dir / file.name}")
                if file.is_file():
                    shutil.move(file, dest_dir / file.name)
        else:
            pass # put logging
            # print("No results to process")


if __name__ == "__main__":
    # if multi fails here, try to uncomment line below
    # mp.set_start_method('fork', force=True)
    config_path = load_config_path()
    repo_root = Path(__file__).parent.parents[1]
    runner = BestMatchRunner.from_config(
        config_path=config_path,
        config_overrides={
            "runner_type": RunnerType.BEST_MATCH.value,
            "model_type": "ada",
            "nr_of_phenopackets": "10",
            "nr_of_results": 10,
            "collection_name": "ada002_lrd_hpo_embeddings",
            "db_collection_path": f"{str(repo_root)}/emb_data/models/ada002",
        }
    )

    runner.prepare()
    runner.run()