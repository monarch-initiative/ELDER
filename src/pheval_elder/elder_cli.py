from pathlib import Path
from pheval_elder.runner import ElderPhEvalRunner
from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures
import logging

import click
from click_default_group import __version__

__all__ = [
    "main",
]

@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet")
@click.version_option(__version__)
def main(verbose: int, quiet: bool):
    logging.basicConfig()
    logger = logging.root
    if verbose >= 2:
        logger.setLevel(level=logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(level=logging.INFO)
    else:
        logger.setLevel(level=logging.WARNING)
    if quiet:
        logger.setLevel(level=logging.ERROR)
    logger.info(f"Logger {logger.name} set to level {logger.level}")


@click.group()
def elder():
    """Elder CLI for running phenopacket analysis."""
    pass


def run_elder(strategy: str, embedding_model: str, nr_of_phenopackets: str, collection_name: str, nr_of_results: int,  db_collection_path: str = None, tpc_comparison: bool = False):
    """Reusable function to initialize and run the ElderPhEvalRunner."""
    runner = ElderPhEvalRunner(
        input_dir=Path("."),
        testdata_dir=Path("."),
        tmp_dir=Path("."),
        output_dir=Path("."),
        config_file=Path("."),
        version="0.3.2",
        similarity_measure=SimilarityMeasures.COSINE,
        collection_name=collection_name,
        strategy=strategy,
        embedding_model=embedding_model,
        nr_of_phenopackets=nr_of_phenopackets,
        nr_of_results=nr_of_results,
        db_collection_path=db_collection_path,
        # tpc_comparison=tpc_comparison,
    )
    runner.prepare()
    runner.run()
    runner.post_process()


@elder.command()
@click.argument("embedding_model", type=click.Choice(["small", "large", "ada"], case_sensitive=False))
@click.argument("nr_of_phenopackets", type=str)
@click.argument("nr_of_results", type=int)
@click.argument("db_collection_path", type=str)
@click.option("--collection_name", default="definition_hpo", help="Name of the collection to use.")
def average(embedding_model, nr_of_phenopackets, nr_of_results, db_collection_path, collection_name):
    """
    Run analysis using the 'average' strategy.

    EMBEDDING_MODEL: Choose 'small', 'large', or 'ada'.

    NR_OF_PHENOPACKETS: Number of phenopackets to process (e.g., 385 or 7702).

    Examples:
        elder average small 385 --collection_name custom_collection
        elder average ada 7702 --collection_name definition_hpo
    """
    run_elder(
        strategy="avg",
        embedding_model=embedding_model,
        nr_of_phenopackets=nr_of_phenopackets,
        collection_name=collection_name,
        nr_of_results=nr_of_results,
        db_collection_path=db_collection_path,
    )

@elder.command()
@click.argument("embedding_model", type=click.Choice(["small", "large", "ada"], case_sensitive=False))
@click.argument("nr_of_phenopackets", type=str)
@click.argument("nr_of_results", type=int, default=10)
@click.argument("db_collection_path", type=str)
@click.option("--collection_name", default="definition_hpo", help="Name of the collection to use.")
def weighted_average(embedding_model, nr_of_phenopackets, nr_of_results, db_collection_path, collection_name):
    """
    Run analysis using the 'weighted_average' strategy.

    EMBEDDING_MODEL: Choose 'small', 'large', or 'ada'.

    NR_OF_PHENOPACKETS: Number of phenopackets to process (e.g., 385 or 7702).

    Examples:
        elder weighted-average small 385 --collection_name custom_collection
        elder weighted-average ada 7702 --collection_name definition_hpo
    """
    run_elder(
        strategy="wgt_avg",
        embedding_model=embedding_model,
        nr_of_phenopackets=nr_of_phenopackets,
        collection_name=collection_name,
        nr_of_results=nr_of_results,
        db_collection_path=db_collection_path,
    )
@elder.command()
@click.argument("embedding_model", type=click.Choice(["small", "large", "ada"], case_sensitive=False))
@click.argument("nr_of_phenopackets", type=str)
@click.argument("nr_of_results", type=int, default=10)
@click.option("--collection_name", default="definition_hpo", help="Name of the collection to use.")
def termset_pairwise_comparison(embedding_model, nr_of_phenopackets, nr_of_results, collection_name):
    """
    Run analysis using the 'termset_pairwise_comparison' strategy.

    EMBEDDING_MODEL: Choose 'small', 'large', or 'ada'.

    NR_OF_PHENOPACKETS: Number of phenopackets to process (e.g., 385 or 7702).

    Examples:
        elder termset-pairwise-comparison small 385 --collection_name custom_collection
        elder termset-pairwise-comparison ada 7702 --collection_name definition_hpo
    """
    run_elder(
        strategy="tpc",
        embedding_model=embedding_model,
        nr_of_phenopackets=nr_of_phenopackets,
        collection_name=collection_name,
        nr_of_results=nr_of_results,
        # tpc_comparison=tpc_comparison
    )

#     elder termset_pairwise_comparison large 385 best_match_col



if __name__ == "__main__":
    # Uncomment one of the lines below for testing specific commands in an IDE:

    # Testing 'average' strategy
    # sys.argv += ["avg", "small", "385"]
    # sys.argv += ["avg", "large", "385"]
    # sys.argv += ["avg", "ada", "385"]
    # sys.argv += ["avg", "small", "7702"]
    # sys.argv += ["avg", "large", "7702"]
    # sys.argv += ["avg", "ada", "7702"]

    # Testing 'weighted_average' strategy
    # sys.argv += ["wgt_avg", "small", "385"]
    # sys.argv += ["wgt_avg", "large", "385"]
    # sys.argv += ["wgt_avg", "ada", "385"]
    # sys.argv += ["wgt_avg", "small", "7702"]
    # sys.argv += ["wgt_avg", "large", "7702"]
    # sys.argv += ["wgt_avg", "ada", "7702"]

    elder()
