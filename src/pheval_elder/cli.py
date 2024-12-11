import click
from pathlib import Path
from src.pheval_elder.runner import ElderPhEvalRunner
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures

@click.group()
def elder():
    """Elder CLI for running phenopacket analysis."""
    pass


def run_elder(strategy: str, embedding_model: str, nr_of_phenopackets: int):
    """Reusable function to initialize and run the ElderPhEvalRunner."""
    runner = ElderPhEvalRunner(
        similarity_measure=SimilarityMeasures.COSINE,
        collection_name="definition_hpo",
        strategy=strategy,
        embedding_model=embedding_model,
        nr_of_phenopackets=nr_of_phenopackets,
        input_dir=Path("./notebooks"),  # Adjust as necessary
        output_dir=Path("./output"),
        tmp_dir=Path("./tmp"),
    )
    runner.prepare()
    runner.run()
    runner.post_process()


@elder.command()
@click.argument("embedding_model", type=click.Choice(["small", "large", "ada"], case_sensitive=False))
@click.argument("nr_of_phenopackets", type=int)
def average(embedding_model, nr_of_phenopackets):
    """
    Run analysis using the 'average' strategy.

    EMBEDDING_MODEL: Choose 'small', 'large', or 'ada'.

    NR_OF_PHENOPACKETS: Number of phenopackets to process (e.g., 385 or 5000).

    Examples:
        elder average small 385
        elder average large 5000
    """
    run_elder(strategy="avg", embedding_model=embedding_model, nr_of_phenopackets=nr_of_phenopackets)


@elder.command()
@click.argument("embedding_model", type=click.Choice(["small", "large", "ada"], case_sensitive=False))
@click.argument("nr_of_phenopackets", type=int)
def weighted_average(embedding_model, nr_of_phenopackets):
    """
    Run analysis using the 'weighted_average' strategy.

    EMBEDDING_MODEL: Choose 'small', 'large', or 'ada'.

    NR_OF_PHENOPACKETS: Number of phenopackets to process (e.g., 385 or 5000).

    Examples:
        elder weighted_average small 385
        elder weighted_average ada 5000
    """
    run_elder(strategy="wgt_avg", embedding_model=embedding_model, nr_of_phenopackets=nr_of_phenopackets)


if __name__ == "__main__":
    import sys

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
