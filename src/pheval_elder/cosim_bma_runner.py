"""
Best Match Runner (Term-set Pairwise Comparison).

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
        # Get total number of phenopackets to process
        total_phenopackets = int(self.elder_runner.nr_of_phenopackets)
        print(f"Total PhenoPackets: {total_phenopackets}")
        
        # Get phenopackets directory and file list
        path = self.get_phenopackets_dir(self.config)
        print(f"Reading phenopackets from {path}")
        file_list = all_files(path)
        
        # Initialize lists for phenotypes and file names
        phenotype_sets = []
        file_names = []

        # Start timing
        start_time = time.time()
        print("Collecting phenotype sets...")
        
        # Process each phenopacket
        for i, file_path in tqdm(enumerate(file_list, start=1), total=total_phenopackets):
            # Read phenopacket
            phenopacket = phenopacket_reader(file_path)
            phenopacket_util = PhenopacketUtil(phenopacket)
            # Extract observed phenotypes
            observed_phenotypes = phenopacket_util.observed_phenotypic_features()
            observed_phenotypes_hpo_ids = [
                update_hpo_id(observed_phenotype.type.id) for observed_phenotype in observed_phenotypes
            ]
            # Store file name and phenotypes
            file_names.append(file_path.name)
            phenotype_sets.append(observed_phenotypes_hpo_ids)

        # Run analysis and process results
        if self.elder_runner is not None and self.elder_runner.strategy == "tpc":
            # Run analysis on all phenotype sets
            all_results = self.elder_runner.tcp_analysis_optimized(
                phenotype_sets, 
                self.config.runner.nr_of_results
            )
            
            # Process each result
            for file_name, result_set in zip(file_names, all_results):
                if result_set:
                    self.current_file_name = file_name
                    self.results = result_set
                    self.post_process()
        else:
            raise RuntimeError(f"Invalid strategy: {self.elder_runner.strategy}")

        # End timing
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")

    def post_process(self) -> None:
        """
        Process the results of the analysis.
        
        This method:
        1. Creates the output directory structure
        2. Generates the PhEval result files
        3. Moves the result files to the appropriate directory
        """
        if self.input_dir_config.disease_analysis and self.results:
            # Create output file name and directories
            output_file_name = f"{self.current_file_name}"
            self.tmp_dir = self.pheval_disease_results_dir / "pheval_disease_results/"
            dest_dir = self.pheval_disease_results_dir / self.elder_runner.results_dir_name / self.elder_runner.results_sub_dir
            
            # Create directories if they don't exist
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate PhEval result
            generate_pheval_result(
                pheval_result=self.results,
                sort_order_str="DESCENDING",  # Descending for TPC, ascending for others
                output_dir=self.pheval_disease_results_dir,
                tool_result_path=Path(output_file_name),
            )
            
            # Move result files to the destination directory
            for file in self.tmp_dir.iterdir():
                print(f"Moving file {file} to {dest_dir / file.name}")
                if file.is_file():
                    shutil.move(file, dest_dir / file.name)
        else:
            print("No results to process")


if __name__ == "__main__":
    # Create and run runner from configuration
    runner = BestMatchRunner.from_config(
        runner_type=RunnerType.BEST_MATCH.value,
        model_type="mxbai",
        nr_of_phenopackets="5084",
        nr_of_results=10,
        collection_name="mxbai_lrd_hpo_embeddings",
        db_collection_path="/Users/ck/Monarch/elder/emb_data/models/mxbai-l",
    )
    
    # Run analysis
    runner.prepare()
    runner.run()