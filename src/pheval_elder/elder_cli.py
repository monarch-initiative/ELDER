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


def run_elder(strategy: str, embedding_model: str, nr_of_phenopackets: str, collection_name: str):
    """Reusable function to initialize and run the ElderPhEvalRunner."""
    runner = ElderPhEvalRunner(
        similarity_measure=SimilarityMeasures.COSINE,
        collection_name=collection_name,
        strategy=strategy,
        embedding_model=embedding_model,
        nr_of_phenopackets=nr_of_phenopackets,
    )
    runner.prepare()
    runner.run()
    runner.post_process()


@elder.command()
@click.argument("embedding_model", type=click.Choice(["small", "large", "ada"], case_sensitive=False))
@click.argument("nr_of_phenopackets", type=str)
@click.option("--collection_name", default="definition_hpo", help="Name of the collection to use.")
def average(embedding_model, nr_of_phenopackets, collection_name):
    """
    Run analysis using the 'average' strategy.

    EMBEDDING_MODEL: Choose 'small', 'large', or 'ada'.

    NR_OF_PHENOPACKETS: Number of phenopackets to process (e.g., 385 or 5000).

    Examples:
        elder average small 385 --collection_name custom_collection
        elder average ada 5000 --collection_name definition_hpo
    """
    run_elder(strategy="avg", embedding_model=embedding_model, nr_of_phenopackets=nr_of_phenopackets, collection_name=collection_name)

@elder.command()
@click.argument("embedding_model", type=click.Choice(["small", "large", "ada"], case_sensitive=False))
@click.argument("nr_of_phenopackets", type=str)
@click.option("--collection_name", default="definition_hpo", help="Name of the collection to use.")
def weighted_average(embedding_model, nr_of_phenopackets, collection_name):
    """
    Run analysis using the 'weighted_average' strategy.

    EMBEDDING_MODEL: Choose 'small', 'large', or 'ada'.

    NR_OF_PHENOPACKETS: Number of phenopackets to process (e.g., 385 or 5000).

    Examples:
        elder weighted_average small 385 --collection_name custom_collection
        elder weighted_average ada 5000 --collection_name definition_hpo
    """
    run_elder(strategy="wgt_avg", embedding_model=embedding_model, nr_of_phenopackets=nr_of_phenopackets, collection_name=collection_name)

if __name__ == "__main__":
    # Uncomment one of the lines below for testing specific commands in an IDE:

    # Testing 'average' strategy
    # sys.argv += ["average", "small", "385"]
    # sys.argv += ["average", "large", "385"]
    # sys.argv += ["average", "ada", "385"]
    # sys.argv += ["average", "small", "5000"]
    # sys.argv += ["average", "large", "5000"]
    # sys.argv += ["average", "ada", "5000"]

    # Testing 'weighted_average' strategy
    # sys.argv += ["weighted_average", "small", "385"]
    # sys.argv += ["weighted_average", "large", "385"]
    # sys.argv += ["weighted_average", "ada", "385"]
    # sys.argv += ["weighted_average", "small", "5000"]
    # sys.argv += ["weighted_average", "large", "5000"]
    # sys.argv += ["weighted_average", "ada", "5000"]

    elder()
