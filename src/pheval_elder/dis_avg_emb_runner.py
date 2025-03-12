"""
Disease Average Embedding Runner.

This module provides a runner for analyzing phenotype sets using
the average embeddings strategy.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any, Optional, Union

import multiprocessing as mp
from pheval.post_processing.post_processing import PhEvalDiseaseResult, generate_pheval_result
from pheval.utils.file_utils import all_files
from pheval.utils.phenopacket_utils import PhenopacketUtil, phenopacket_reader
from tqdm import tqdm

import pheval_elder.prepare.core.collections.globals as g
from pheval_elder.base_runner import BaseElderRunner
from pheval_elder.prepare.config.config_loader import load_config_path
from pheval_elder.prepare.config.unified_config import RunnerType
from pheval_elder.prepare.core.utils.obsolete_hp_mapping import update_hpo_id


@dataclass
class DiseaseAvgEmbRunner(BaseElderRunner):
    """
    Runner for analyzing phenotype sets using the average embeddings strategy.
    
    This runner calculates average embeddings for each phenotype set and
    queries for similar diseases.
    """

    def run(self) -> None:
        """
        Run the average embeddings analysis on all phenotype sets.
        
        This method:
        1. Reads phenopackets from the specified directory
        2. Extracts observed phenotypes from each phenopacket
        3. Runs the average embeddings analysis
        4. Processes the results
        """
        total_phenopackets = int(self.elder_runner.nr_of_phenopackets)
        # print(f"Running {total_phenopackets} phenopackets")
        
        path = self.get_phenopackets_dir(self.config)
        # print(f"Reading phenopackets from {path}")
        file_list = all_files(path)
        
        phenotype_sets = []
        file_names = []

        for i, file_path in tqdm(enumerate(file_list, start=1), total=total_phenopackets):
            self.current_file_name = file_path.stem
            phenopacket = phenopacket_reader(file_path)
            phenopacket_util = PhenopacketUtil(phenopacket)
            observed_phenotypes = phenopacket_util.observed_phenotypic_features()
            observed_phenotypes_hpo_ids = [
                update_hpo_id(observed_phenotype.type.id) for observed_phenotype in observed_phenotypes
            ]
            file_names.append(file_path.name)
            phenotype_sets.append(observed_phenotypes_hpo_ids)

        # Initialize the global collection for multiprocessing
        g.global_avg_disease_emb_collection = self.elder_runner.disease_service.disease_new_avg_embeddings_collection
        
        if self.elder_runner is not None and self.elder_runner.strategy == "avg":
            self.results = self.elder_runner.optimized_avg_analysis(phenotype_sets, self.config.runner.nr_of_results)
            for file_name, result_set in zip(file_names, self.results):
                if result_set:
                    self.current_file_name = file_name
                    self.results = result_set
                    self.post_process()
        else:
            raise RuntimeError(f"Invalid strategy: {self.elder_runner.strategy}")

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
                sort_order_str="ASCENDING",  # Ascending for average, descending for TPC
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
    mp.set_start_method('fork', force=True)
    config_path = load_config_path()
    repo_root = Path(__file__).parent.parents[1]
    runner = DiseaseAvgEmbRunner.from_config(
        config_path=config_path,
        config_overrides={
            "runner_type": RunnerType.AVERAGE.value,
            "model_type": "ada",
            "nr_of_phenopackets": "10",
            "nr_of_results": 10,
            "collection_name": "ada002_lrd_hpo_embeddings",
            "db_collection_path": f"{str(repo_root)}/emb_data/models/ada002",
        }
    )

    runner.prepare()
    runner.run()