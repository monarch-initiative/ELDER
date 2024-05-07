import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any

from pheval.post_processing.post_processing import PhEvalDiseaseResult, generate_pheval_result
from pheval.runners.runner import PhEvalRunner
from pheval.utils.file_utils import all_files
from pheval.utils.phenopacket_utils import PhenopacketUtil, phenopacket_reader
from tqdm import tqdm

from pheval_elder.prepare.config.config_loader import ElderConfig
from pheval_elder.prepare.core.elder import ElderRunner
from pheval_elder.prepare.simple_service import SimpleService
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = ElderConfig.parse_obj(self.input_dir_config.tool_specific_configuration_options)
        self.simple_service = SimpleService()
        self.simple_runner = ElderRunner(config=config,
                                         similarity_measure=SimilarityMeasures.COSINE)
        self.current_file_name = None

    def prepare(self):
        """prepare"""
        print("preparing and working")
        self.simple_service.greet()
        start_init = time.time()
        self.simple_runner.initialize_data()
        init_time = time.time() - start_init
        print("initialized simple runner")
        print(init_time)
        start_setup = time.time()
        self.simple_runner.setup_collections()
        print("set up collections")
        setup_time = time.time() - start_setup
        print(setup_time)

    def run(self):
        print("Phenopacket analysis")
        path = Path("/Users/carlo/Carlo/pheval/pheval/corpora/lirical/default/phenopackets")
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
            print(observed_phenotypes_hpo_ids)

            if self.simple_runner is not None:
                self.results = self.simple_runner.run_analysis(observed_phenotypes_hpo_ids)
                # print("Running with custom pheval runner")
                self.postpost_process()  # Call post_process here for each file
            else:
                print("Main system is not initialized")

    def post_process(self):
        print("i do i do u do")

    def postpost_process(self):
        """post_process"""
        print("post processing")
        if self.input_dir_config.disease_analysis and self.results:
            disease_results = self.create_disease_results(self.results)
            output_file_name = f"{self.current_file_name}_disease_results.tsv"
            generate_pheval_result(
                pheval_result=disease_results,
                sort_order_str="ASCENDING",
                output_dir=self.pheval_disease_results_dir,
                tool_result_path=Path(output_file_name),
            )
            print(f"generated pheval results for {output_file_name}")
        else:
            print("No results to process")

    @staticmethod
    def create_disease_results(query_results):
        return [
            PhEvalDiseaseResult(disease_name=disease_id, disease_identifier=disease_id, score=distance)
            for disease_id, distance in query_results
        ]
