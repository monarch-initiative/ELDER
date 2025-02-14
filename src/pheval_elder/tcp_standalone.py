
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
from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures
from pheval_elder.runner import Z_PHENOPACKET_TEST

current_dir = Path(__file__).parent
repo_root = current_dir.parents[1]
LIRICAL_PHENOPACKETS = "/Users/ck/Monarch/elder/LIRICAL_phenopackets"

ALL_PHENOPACKETS = "5213_phenopackets"
# LIRICAL_PHENOPACKETS = "LIRICAL_phenopackets"
OUTPUT = Path("src/pheval_disease_results")

@dataclass
class TCPRunner(PhEvalRunner):
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
    results: Union[List[Any], Iterator] = field(default_factory=list)
    results_path: str = None
    tpc_comparison: bool = False
    file_names: List[str] = field(default_factory=list)


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

    def prepare(self):
        """prepare"""
        self.elder_runner.initialize_data()
        self.elder_runner.setup_collections()
        print("done")

    ## uncomment for basic run that does nothing special, no precompute, no parallel
    # def run(self):
    #     total_phenopackets = int(self.elder_runner.nr_of_phenopackets)
    #     print(f"Total PhenoPackets: {total_phenopackets}")
    #     path = self.phenopackets(total_phenopackets)
    #     file_list = all_files(path)
    #     s = time.time()
    #
    #     for i, file_path in tqdm(enumerate(file_list, start=1), total=total_phenopackets):
    #         print(f"Processing file {i}: {file_path}")
    #         self.current_file_name = file_path.stem
    #         phenopacket = phenopacket_reader(file_path)
    #         phenopacket_util = PhenopacketUtil(phenopacket)
    #         observed_phenotypes = phenopacket_util.observed_phenotypic_features()
    #         observed_phenotypes_hpo_ids = [
    #             observed_phenotype.type.id for observed_phenotype in observed_phenotypes
    #         ]
    #
    #     # process all sets at once
    #         if self.elder_runner is not None and self.elder_runner.strategy == "tpc": # best match average
    #             self.results = self.elder_runner.tcp_analysis_normal(observed_phenotypes_hpo_ids, self.nr_of_results)
    #             self.post_process()
    #         else:
    #             raise RuntimeError()
    #     e = time.time()
    #
    #     print(f"Elapsed time: {e - s}")

    # In your TCPRunner class
    def run(self):
        total_phenopackets = int(self.elder_runner.nr_of_phenopackets)
        print(f"Total PhenoPackets: {total_phenopackets}")
        path = self.phenopackets(total_phenopackets)
        file_list = all_files(path)

        # First collect all phenotype sets
        phenotype_sets = []
        # file_names = []

        s = time.time()
        print("Collecting phenotype sets...")
        for i, file_path in tqdm(enumerate(file_list, start=1), total=total_phenopackets):
            print(f"Processing file {i}: {file_path}")
            phenopacket = phenopacket_reader(file_path)
            phenopacket_util = PhenopacketUtil(phenopacket)
            observed_phenotypes = phenopacket_util.observed_phenotypic_features()
            observed_phenotypes_hpo_ids = [
                observed_phenotype.type.id for observed_phenotype in observed_phenotypes
            ]
            self.file_names.append(file_path.name)
            phenotype_sets.append(observed_phenotypes_hpo_ids)

        if self.elder_runner is not None and self.elder_runner.strategy == "tpc":

            self.results = self.elder_runner.tcp_analysis_optimized(phenotype_sets, self.nr_of_results)
            self.post_process()

            print(f"post process done {os.getpid()}")

        e = time.time()
        print(f"Total processing time: {e - s:.2f} seconds")

    def post_process(self):
        """post_process"""
        if self.input_dir_config.disease_analysis and self.results:
            output_file_name = f"{self.current_file_name}"
            add_sub_dir = self.pheval_disease_results_dir / "pheval_disease_results/"
            dest_dir = self.pheval_disease_results_dir / self.elder_runner.results_dir_name / self.elder_runner.results_sub_dir
            add_sub_dir.mkdir(parents=True, exist_ok=True)
            dest_dir.mkdir(parents=True, exist_ok=True)
            generate_pheval_result(
                pheval_result=self.results,
                sort_order_str="DESCENDING", # Descending for TPC else ASCENDING
                output_dir=self.pheval_disease_results_dir,
                tool_result_path=Path(output_file_name),
            )
        else:
            print("No results to process")


    @staticmethod
    def phenopackets(nr_of_phenopackets: int = None) -> Path:
        phenopackets_dir = None
        if nr_of_phenopackets:
            phenopackets_dir = repo_root / ALL_PHENOPACKETS
        return phenopackets_dir


def batch_write_results(file_names: List[str],
                       all_results: List[List],
                       output_base_dir: Path,
                       results_dir_name: str,
                       results_sub_dir: str):
    """Fast batch writing of results using generate_pheval_result"""

    # Create output directory once
    output_dir = output_base_dir / "pheval_disease_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all files
    for file_name, results in zip(file_names, all_results):
        if not results:
            continue

        output_file_name = f"{file_name}_disease_results.tsv"

        # Use the existing generate_pheval_result function
        generate_pheval_result(
            pheval_result=results,
            sort_order_str="DESCENDING",
            output_dir=output_base_dir,
            tool_result_path=Path(output_file_name)
        )

# Alternatively, we could batch the calls to avoid many small writes:
def batch_write_results_buffered(file_names: List[str],
                               all_results: List[List],
                               output_base_dir: Path,
                               results_dir_name: str,
                               results_sub_dir: str,
                               batch_size: int = 100):
    """Buffered batch writing using generate_pheval_result"""

    # Create output directory once
    add_sub_dir = output_base_dir / "pheval_disease_results"
    dest_dir = output_base_dir / results_dir_name / results_sub_dir
    output_dir = output_base_dir / "pheval_disease_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    add_sub_dir.mkdir(parents=True, exist_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Create batches
    batched_data = []
    current_batch = []

    for file_name, results in zip(file_names, all_results):
        if not results:
            continue

        current_batch.append((file_name, results))

        if len(current_batch) >= batch_size:
            batched_data.append(current_batch)
            current_batch = []

    if current_batch:  # Don't forget the last batch
        batched_data.append(current_batch)

    # Process batches
    print(f"Processing {len(batched_data)} batches with size {batch_size}")

    for batch_idx, batch in enumerate(batched_data):
        print(f"Processing batch {batch_idx + 1}/{len(batched_data)}")

    # Process batches
    # for batch in batched_data:
        # Create all DataFrames for this batch
        for file_name, results in batch:
            output_file_name = f"{file_name}_disease_results.tsv"
            generate_pheval_result(
                pheval_result=results,
                sort_order_str="DESCENDING",
                output_dir=output_base_dir,
                tool_result_path=Path(output_file_name)
            )

        # Copy from pheval_disease_results to final destination
        source_file = add_sub_dir / f"{output_file_name.replace('.tsv', '-pheval_disease_result.tsv')}"
        if source_file.exists():
            shutil.copy(source_file, dest_dir / source_file.name)
        else:
            print(f"Warning: Expected source file not found: {source_file}")

def parallel_post_process(args):
    """Standalone post-processing function that doesn't rely on self"""
    file_name, result_set, base_dir, results_dir_name, results_sub_dir = args

    if not result_set:
        return

    # Create required paths
    output_file_name = f"{file_name}_disease_results.tsv"
    pheval_disease_results_dir = Path(base_dir)
    add_sub_dir = pheval_disease_results_dir / "pheval_disease_results/"
    dest_dir = pheval_disease_results_dir / results_dir_name / results_sub_dir

    # Create directories
    add_sub_dir.mkdir(parents=True, exist_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Generate result
    generate_pheval_result(
        pheval_result=result_set,
        sort_order_str="DESCENDING",
        output_dir=pheval_disease_results_dir,
        tool_result_path=Path(output_file_name),
    )

    # Copy files
    for file in add_sub_dir.iterdir():
        if file.is_file():
            shutil.copy(file, dest_dir / file.name)

def process_all_results(file_names, all_results, base_dir, results_dir_name, results_sub_dir):
    """Process all results in parallel"""
    import multiprocessing as mp

    # Prepare arguments for parallel processing
    process_args = [
        (file_name, result_set, base_dir, results_dir_name, results_sub_dir)
        for file_name, result_set in zip(file_names, all_results)
    ]

    num_cores = mp.cpu_count()
    with mp.Pool(num_cores) as pool:
        list(tqdm(
            pool.imap(parallel_post_process, process_args),
            total=len(process_args),
            desc="Post-processing results"
        ))
        pool.close()
        pool.join()


if __name__ == "__main__":
    tcp_runner = TCPRunner(
        input_dir=Path("/Users/ck/Monarch/elder"),
        testdata_dir=Path(".."),
        tmp_dir=Path(".."),
        output_dir=Path("/Users/ck/Monarch/elder/output"),
        config_file=Path(".."),
        version="0.3.2",
        similarity_measure=SimilarityMeasures.COSINE,
        collection_name="nomic_lrd_hpo_embeddings",
        strategy="tpc",
        embedding_model="",
        nr_of_phenopackets="16",
        nr_of_results=10,
        db_collection_path="/Users/ck/Monarch/elder/emb_data/models/nomic",
    )
    tcp_runner.prepare()
    tcp_runner.run()
    # tcp_runner.post_process()