import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any

from pheval.post_processing.post_processing import PhEvalDiseaseResult, generate_pheval_result
from pheval.runners.runner import PhEvalRunner
from pheval.utils.file_utils import all_files
from pheval.utils.phenopacket_utils import PhenopacketUtil, phenopacket_reader
from tqdm import tqdm

from pheval_elder.prepare.core.run.elder import ElderRunner
from pheval_elder.prepare.core.utils.obsolete_hp_mapping import update_hpo_id
from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures


current_dir = Path(__file__).parent
repo_root = current_dir.parents[1]
LIRICAL_PHENOPACKETS = "LIRICAL_phenopackets"
Z_PHENOPACKET_TEST = "10_z_phenopackets"
ALL_PHENOPACKETS = "5084_phenopackets"
output_dir = Path(repo_root / "output")

@dataclass
class DiseaseAvgEmbRunner(PhEvalRunner):

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
        # handle this class-specific args
        self.similarity_measures = similarity_measure
        self.collection_name = collection_name
        self.strategy = strategy
        self.embedding_model = embedding_model
        self.nr_of_phenopackets_str = nr_of_phenopackets
        self.nr_of_results = nr_of_results
        self.db_collection_path = db_collection_path
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
        self.data_processor = self.elder_runner.data_processor
        self.disease_service = self.elder_runner.disease_service

    def prepare(self):
        self.elder_runner.initialize_data()
        self.elder_runner.setup_collections()

    def run(self):
        total_phenopackets = int(self.elder_runner.nr_of_phenopackets)
        print(f"Running {total_phenopackets} phenopackets")
        path = self.phenopackets(total_phenopackets)
        print(path)
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

        if self.elder_runner is not None and self.elder_runner.strategy == "avg":
            self.results = self.elder_runner.optimized_avg_analysis(phenotype_sets, self.nr_of_results)
            for file_name, result_set in zip(file_names, self.results):
                if result_set:
                    self.current_file_name = file_name
                    self.results = result_set
                    self.post_process()
        else:
            raise RuntimeError()


    def post_process(self):
        if self.input_dir_config.disease_analysis and self.results:
            output_file_name = f"{self.current_file_name}"
            self.tmp_dir = self.pheval_disease_results_dir / "pheval_disease_results/"
            dest_dir = self.pheval_disease_results_dir / self.elder_runner.results_dir_name / self.elder_runner.results_sub_dir
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
            dest_dir.mkdir(parents=True, exist_ok=True)
            generate_pheval_result(
                pheval_result=self.results,
                sort_order_str="ASCENDING",  # Descending for TPC else ASCENDING
                output_dir=self.pheval_disease_results_dir,
                tool_result_path=Path(output_file_name),
            )
            for file in self.tmp_dir.iterdir():
                print("moving file", file, "to", dest_dir / file.name)
                if file.is_file():
                    shutil.move(file, dest_dir / file.name)

        else:
            print("No results to process")

    @staticmethod
    def create_disease_results(query_results):
        return [
            PhEvalDiseaseResult(disease_name=disease_name, disease_identifier=disease_id, score=distance)
            for disease_id, disease_name, distance in query_results
        ]

    @staticmethod
    def phenopackets(nr_of_phenopackets: int = None) -> Path:
        phenopackets_dir = None
        if nr_of_phenopackets:
            phenopackets_dir = repo_root / ALL_PHENOPACKETS
        return phenopackets_dir


if __name__ == "__main__":
    # TODO: this is for avg, new runner for wgt (below for now)
    import multiprocessing as mp
    import pheval_elder.prepare.core.collections.globals as g
    mp.set_start_method('fork', force=True)

    runner = DiseaseAvgEmbRunner(
        input_dir=repo_root,
        testdata_dir=Path(".."),
        tmp_dir=Path(".."),
        output_dir=output_dir,
        config_file=Path(".."),
        version="0.3.2",
        similarity_measure=SimilarityMeasures.COSINE,
        collection_name="ada002_lrd_hpo_embeddings",
        strategy="avg",
        embedding_model="xxadaxx",
        nr_of_phenopackets="5084",
        nr_of_results=10,
        db_collection_path="/Users/ck/Monarch/elder/emb_data/models/ada002",
    )
    runner.prepare()
    # init and assign global collection for multiprocess
    g.global_avg_disease_emb_collection = runner.disease_service.disease_new_avg_embeddings_collection
    runner.run()



