import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any

from pheval.post_processing.post_processing import PhEvalDiseaseResult, generate_pheval_result
from pheval.runners.runner import PhEvalRunner
from pheval.utils.file_utils import all_files
from pheval.utils.phenopacket_utils import PhenopacketUtil, phenopacket_reader
from tqdm import tqdm

from pheval_elder.prepare.core.query.termsetpairwise import TermSetPairWiseComparisonQuery
from pheval_elder.prepare.core.run.elder import ElderRunner
from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures

current_dir = Path(__file__).parent
repo_root = current_dir.parents[1]
LIRICAL_PHENOPACKETS = "LIRICAL_phenopackets"
Z_PHENOPACKET_TEST = "10_z_phenopackets"
ALL_PHENOPACKETS = "5213_phenopackets"
output_dir = Path(repo_root / "output")

@dataclass
class ElderPhEvalRunner(PhEvalRunner):
    """_summary_"""

    input_dir: Path
    testdata_dir: Path
    tmp_dir: Path
    output_dir: Path
    config_file: Path
    version: str
    similarity_measures: SimilarityMeasures
    collection_name: str
    strategy: str
    embedding_model: str
    nr_of_phenopackets_str: str
    db_collection_path: str
    results: List[Any] = field(default_factory=list)
    results_path: str = None
    tpc_comparison: bool = False


    def __init__(
            self,
            input_dir: Path,
            testdata_dir: Path,
            tmp_dir: Path,
            output_dir: Path,
            config_file: Path,
            version: str,
            similarity_measure: SimilarityMeasures,
            collection_name: str,
            strategy: str,
            embedding_model: str,
            nr_of_phenopackets: str,
            nr_of_results: int,
            db_collection_path: str,
            # tpc_comparison: bool = False,
            **kwargs,
    ):
        # Pass only arguments that belong to PhEvalRunner as this is the ParentClass
        super().__init__(
            input_dir=input_dir,
            testdata_dir=testdata_dir,
            tmp_dir=tmp_dir,
            output_dir=output_dir,
            config_file=config_file,
            version=version,
            **kwargs,
        )
        # Handle ElderPhEvalRunner-specific arguments
        self.similarity_measures = similarity_measure
        self.collection_name = collection_name
        self.strategy = strategy
        self.embedding_model = embedding_model
        self.nr_of_phenopackets_str = nr_of_phenopackets
        self.nr_of_results = nr_of_results
        self.db_collection_path = db_collection_path
        # self.tpc_comparison = tpc_comparison
        # for tpc comparison
        # if self.db_collection_path is not None and self.tpc_comparison == True:
        self.elder_runner = ElderRunner(
            similarity_measure=SimilarityMeasures.COSINE,
            collection_name=self.collection_name,
            strategy=self.strategy,
            embedding_model=self.embedding_model,
            nr_of_phenopackets=self.nr_of_phenopackets_str,
            db_collection_path = self.db_collection_path,
            nr_of_results = self.nr_of_results
        )
        self.current_file_name = None

    def prepare(self):
        """prepare"""
        self.elder_runner.initialize_data()
        self.elder_runner.setup_collections()

    def run(self):
        # better to have seperate runners, but for now
        # if not self.tpc_comparison:
        total_phenopackets = int(self.elder_runner.nr_of_phenopackets)
        print(f"Running {total_phenopackets} phenopackets")
        path = self.phenopackets(total_phenopackets)
        print(path)
        file_list = all_files(path)
        for i, file_path in tqdm(enumerate(file_list, start=1), total=total_phenopackets):
            self.current_file_name = file_path.stem
            phenopacket = phenopacket_reader(file_path)
            phenopacket_util = PhenopacketUtil(phenopacket)
            observed_phenotypes = phenopacket_util.observed_phenotypic_features()
            observed_phenotypes_hpo_ids = [
                observed_phenotype.type.id for observed_phenotype in observed_phenotypes
            ]

            if self.elder_runner is not None and self.elder_runner.strategy == "avg":
                self.results = self.elder_runner.avg_analysis(observed_phenotypes_hpo_ids, self.nr_of_results)
            elif self.elder_runner is not None and self.elder_runner.strategy == "wgt_avg":
                self.results = self.elder_runner.wgt_avg_analysis(observed_phenotypes_hpo_ids, self.nr_of_results)
            elif self.elder_runner is not None and self.elder_runner.strategy == "tpc": # best match average
                self.results = self.elder_runner.tcp_analysis(observed_phenotypes_hpo_ids, self.nr_of_results)
            else:
                raise RuntimeError()
            self.post_process()
        # tpc ccomparison
        # else:
        #     path = self.phenopackets(int(self.nr_of_phenopackets_str))
        #     file_list = all_files(path)
        #     for i, file_path in tqdm(enumerate(file_list, start=1), total=int(self.nr_of_phenopackets_str)):
        #         self.current_file_name = file_path.stem
        #         phenopacket = phenopacket_reader(file_path)
        #         phenopacket_util = PhenopacketUtil(phenopacket)
        #         observed_phenotypes = phenopacket_util.observed_phenotypic_features()
        #         observed_phenotypes_hpo_ids = [
        #             observed_phenotype.type.id for observed_phenotype in observed_phenotypes
        #         ]
        #         self.results = self.elder_runner.tcp_analysis(observed_phenotypes_hpo_ids, self.nr_of_results)




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
                sort_order_str="ASCENDING", # Descending for TPC else ASCENDING
                output_dir=self.pheval_disease_results_dir,
                tool_result_path=Path(output_file_name),
            )
            for file in add_sub_dir.iterdir():
                if file.is_file():
                    shutil.copy(file, dest_dir / file.name)
        else:
            print("No results to process")


    @staticmethod
    def create_disease_results(query_results):
        return [
            PhEvalDiseaseResult(disease_name=disease_id, disease_identifier=disease_id, score=distance)
            for disease_id, distance in query_results
        ]

    @staticmethod
    def phenopackets(nr_of_phenopackets: int) -> Path:
        phenopackets_dir = None

        if nr_of_phenopackets < 400:
            phenopackets_dir = repo_root / LIRICAL_PHENOPACKETS
        elif nr_of_phenopackets >= 5000:
            phenopackets_dir = repo_root / ALL_PHENOPACKETS

        if phenopackets_dir.exists() and any(phenopackets_dir.iterdir()):
            path = Path(phenopackets_dir).resolve()
            return path
        else:
            ValueError("Check phenopacket directories")

