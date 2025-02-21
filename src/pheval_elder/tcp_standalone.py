
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any, Iterable, Iterator, Union
import os
from pheval.post_processing.post_processing import PhEvalDiseaseResult, generate_pheval_result
from pheval.runners.runner import PhEvalRunner
from pheval.utils.file_utils import all_files
from pheval.utils.phenopacket_utils import PhenopacketUtil, phenopacket_reader
from tqdm import tqdm

from pheval_elder.prepare.core.run.elder import ElderRunner
from pheval_elder.prepare.core.utils.obsolete_hp_mapping import update_hpo_id
from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures
from pheval_elder.runner import Z_PHENOPACKET_TEST, LIRICAL_PHENOPACKETS, ALL_PHENOPACKETS, output_dir, repo_root


@dataclass
class BestMatchRunner(PhEvalRunner):
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
    tmp_dir: Path
    results: Union[List[Any], Iterator] = field(default_factory=list)
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
        print(f"Config File: {config_file}")
        # Handle ElderPhEvalRunner-specific arguments
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
        self.tmp_dir = Path(".")

    def prepare(self):
        """prepare"""
        self.elder_runner.initialize_data()
        self.elder_runner.setup_collections()

    # In your TCPRunner class
    def run(self):
        total_phenopackets = int(self.elder_runner.nr_of_phenopackets)
        print(f"Total PhenoPackets: {total_phenopackets}")
        path = self.phenopackets(total_phenopackets)
        file_list = all_files(path)

        #collect all phenotype sets from files
        phenotype_sets = []
        file_names = []

        s = time.time()
        print("Collecting phenotype sets...")
        for i, file_path in tqdm(enumerate(file_list, start=1), total=total_phenopackets):
            # print(f"Processing file {i}: {file_path}")
            phenopacket = phenopacket_reader(file_path)
            phenopacket_util = PhenopacketUtil(phenopacket)
            observed_phenotypes = phenopacket_util.observed_phenotypic_features()
            observed_phenotypes_hpo_ids = [
                update_hpo_id(observed_phenotype.type.id) for observed_phenotype in observed_phenotypes
            ]
            file_names.append(file_path.name)
            phenotype_sets.append(observed_phenotypes_hpo_ids)

        if self.elder_runner is not None and self.elder_runner.strategy == "tpc":

            # results for all phenotype sets
            all_results = self.elder_runner.tcp_analysis_optimized(phenotype_sets, self.nr_of_results)

            # post process
            for file_name, result_set in zip(file_names, all_results):
                if result_set:
                    self.current_file_name = file_name
                    self.results = result_set
                    self.post_process()

        e = time.time()
        print(f"Total processing time: {e - s:.2f} seconds")

    def post_process(self):
        """post_process"""
        if self.input_dir_config.disease_analysis and self.results:
            output_file_name = f"{self.current_file_name}"
            self.tmp_dir = self.pheval_disease_results_dir / "pheval_disease_results/"
            dest_dir = self.pheval_disease_results_dir / self.elder_runner.results_dir_name / self.elder_runner.results_sub_dir
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
            dest_dir.mkdir(parents=True, exist_ok=True)
            generate_pheval_result(
                pheval_result=self.results,
                sort_order_str="DESCENDING", # Descending for TPC else ASCENDING
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
    def phenopackets(nr_of_phenopackets: int = None) -> Path:
        phenopackets_dir = None
        if nr_of_phenopackets:
            phenopackets_dir = repo_root / ALL_PHENOPACKETS
        return phenopackets_dir


if __name__ == "__main__":
    # TODO run all again (update_hpo)
    #  mxbai - done
    # rest all running
    tcp_runner = BestMatchRunner(
        input_dir=repo_root,
        testdata_dir=Path(".."),
        tmp_dir=Path(".."),
        output_dir=output_dir,
        config_file=Path(".."),
        version="0.3.2",
        similarity_measure=SimilarityMeasures.COSINE,
        collection_name="large3_lrd_hpo_embeddings",
        strategy="tpc",
        embedding_model="large3",
        nr_of_phenopackets="5213",
        nr_of_results=10,
        db_collection_path="/Users/ck/Monarch/elder/emb_data/models/large3",
    )
    tcp_runner.prepare()
    tcp_runner.run()
    # tcp_runner.post_process()
    #
    # tcp_runner = BestMatchRunner(
    #     input_dir=repo_root,
    #     testdata_dir=Path(".."),
    #     tmp_dir=Path(".."),
    #     output_dir=output_dir,
    #     config_file=Path(".."),
    #     version="0.3.2",
    #     similarity_measure=SimilarityMeasures.COSINE,
    #     collection_name="small3_lrd_hpo_embeddings",
    #     strategy="tpc",
    #     embedding_model="small3",
    #     nr_of_phenopackets="5213",
    #     nr_of_results=10,
    #     db_collection_path="/Users/ck/Monarch/elder/emb_data/models/small3",
    # )
    # tcp_runner.prepare()
    # tcp_runner.run()
    #
    # tcp_runner = BestMatchRunner(
    #     input_dir=repo_root,
    #     testdata_dir=Path(".."),
    #     tmp_dir=Path(".."),
    #     output_dir=output_dir,
    #     config_file=Path(".."),
    #     version="0.3.2",
    #     similarity_measure=SimilarityMeasures.COSINE,
    #     collection_name="ada002_lrd_hpo_embeddings",
    #     strategy="tpc",
    #     embedding_model="ada",
    #     nr_of_phenopackets="5213",
    #     nr_of_results=10,
    #     db_collection_path="/Users/ck/Monarch/elder/emb_data/models/ada002",
    # )
    # tcp_runner.prepare()
    # tcp_runner.run()
    #
    # tcp_runner = BestMatchRunner(
    #     input_dir=repo_root,
    #     testdata_dir=Path(".."),
    #     tmp_dir=Path(".."),
    #     output_dir=output_dir,
    #     config_file=Path(".."),
    #     version="0.3.2",
    #     similarity_measure=SimilarityMeasures.COSINE,
    #     collection_name="bge-m3_lrd_hpo_embeddings",
    #     strategy="tpc",
    #     embedding_model="bge-m3",
    #     nr_of_phenopackets="5213",
    #     nr_of_results=10,
    #     db_collection_path="/Users/ck/Monarch/elder/emb_data/models/bge-m3",
    # )
    # tcp_runner.prepare()
    # tcp_runner.run()
    #
    # tcp_runner = BestMatchRunner(
    #     input_dir=repo_root,
    #     testdata_dir=Path(".."),
    #     tmp_dir=Path(".."),
    #     output_dir=output_dir,
    #     config_file=Path(".."),
    #     version="0.3.2",
    #     similarity_measure=SimilarityMeasures.COSINE,
    #     collection_name="nomic_lrd_hpo_embeddings",
    #     strategy="tpc",
    #     embedding_model="nomic",
    #     nr_of_phenopackets="5213",
    #     nr_of_results=10,
    #     db_collection_path="/Users/ck/Monarch/elder/emb_data/models/nomic",
    # )
    # tcp_runner.prepare()
    # tcp_runner.run()
    #
