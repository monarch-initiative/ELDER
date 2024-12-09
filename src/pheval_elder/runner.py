import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any

from pheval.post_processing.post_processing import PhEvalDiseaseResult, generate_pheval_result
from pheval.runners.runner import PhEvalRunner
from pheval.utils.file_utils import all_files
from pheval.utils.phenopacket_utils import PhenopacketUtil, phenopacket_reader
from tqdm import tqdm
from pheval_elder.prepare.core.elder import ElderRunner
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures

# root = Path(__file__).parents[2]

@dataclass
class ElderPhEvalRunner(PhEvalRunner):
    """_summary_"""

    input_dir: Path
    testdata_dir: Path
    tmp_dir: Path
    output_dir: Path
    config_file: Path
    version: str
    results: List[Any] = field(default_factory=list)
    results_path: str = None


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elder_runner = ElderRunner(
            similarity_measure=SimilarityMeasures.COSINE,
            collection_name="lrd_hpo",
            strategy="avg",
            embedding_model="large"
        )
        self.current_file_name = None

    def prepare(self):
        """prepare"""
        self.elder_runner.initialize_data()
        self.elder_runner.setup_collections()

    def run(self):
        path = Path("/Users/carlo/pheval/corpora/lirical/default/phenopackets")
        file_list = all_files(path)
        print(f"Processing {len(file_list)} files...")
        for i, file_path in tqdm(enumerate(file_list, start=1), total=385):
            print(f"Processing file {i}: {file_path}")  # Print the file being processed
            self.current_file_name = file_path.stem
            # print(f"self.current file name  = {self.current_file_name}")
            phenopacket = phenopacket_reader(file_path)
            phenopacket_util = PhenopacketUtil(phenopacket)
            observed_phenotypes = phenopacket_util.observed_phenotypic_features()
            observed_phenotypes_hpo_ids = [
                observed_phenotype.type.id for observed_phenotype in observed_phenotypes
            ]
            # print(observed_phenotypes_hpo_ids)

            if self.elder_runner is not None:
                self.results = self.elder_runner.run_analysis(observed_phenotypes_hpo_ids)
                # print("Running with custom pheval runner")
                self.post_process()  # Call post_process here for each file
            else:
                print("Main system is not initialized")

    def post_process(self):
        """post_process"""
        if self.input_dir_config.disease_analysis and self.results:
            disease_results = self.create_disease_results(self.results)
            output_file_name = f"{self.current_file_name}_disease_results.tsv"
            add_sub_dir = self.pheval_disease_results_dir / "pheval_disease_results/"
            dest_dir = self.pheval_disease_results_dir / self.elder_runner.results_dir_name / self.elder_runner.results_sub_dir
            add_sub_dir.mkdir(parents=True, exist_ok=True)
            dest_dir.mkdir(parents=True, exist_ok=True)
            generate_pheval_result(
                pheval_result=disease_results,
                sort_order_str="DESCENDING",
                output_dir=self.pheval_disease_results_dir,
                tool_result_path=Path(output_file_name),
            )
            for file in add_sub_dir.iterdir():
                if file.is_file():
                    shutil.copy(file, dest_dir / file.name)
        else:
            print("No results to process")

            # somehow pheval_disease_results is always added after but not created
            # print(f"\n\n{results_path}\n\n")
            # print(self.pheval_disease_results_dir)
    @staticmethod
    def create_disease_results(query_results):
        return [
            PhEvalDiseaseResult(disease_name=disease_id, disease_identifier=disease_id, score=distance)
            for disease_id, distance in query_results
        ]
