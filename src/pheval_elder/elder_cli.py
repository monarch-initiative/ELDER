"""
Elder CLI for running phenopacket analysis.

This module provides a command-line interface for running Elder analysis
using the unified configuration system.
"""

import logging
from pathlib import Path
from typing import Optional

import click
from click_default_group import __version__

from pheval_elder.dis_avg_emb_runner import DiseaseAvgEmbRunner
from pheval_elder.dis_wgt_avg_emb_runner import DisWgtAvgEmbRunner
from pheval_elder.cosim_bma_runner import BestMatchRunner
from pheval_elder.prepare.config.unified_config import (
    RunnerType, ModelType, get_config, set_config, ConfigLoader
)
from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures

__all__ = [
    "main",
]


@click.option("-v", "--verbose", count=True, help="Increase verbosity (can be used multiple times)")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all output except errors")
@click.version_option(__version__)
def main(verbose: int, quiet: bool):
    """Configure logging for the Elder CLI."""
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
@click.option(
    "--config", 
    "-c", 
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to Elder configuration file"
)
@click.pass_context
def elder(ctx, config: Optional[Path]):
    """
    Elder CLI for running phenopacket analysis.
    
    Elder can analyze phenopackets using different embedding strategies.
    """
    # Store configuration path in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@elder.command()
@click.option(
    "--model", 
    "-m", 
    type=click.Choice([m.value for m in ModelType], case_sensitive=False),
    help="Embedding model to use"
)
@click.option(
    "--phenopackets", 
    "-p", 
    type=str,
    help="Number of phenopackets to process (e.g., 385 or 5084)"
)
@click.option(
    "--results", 
    "-r", 
    type=int,
    help="Number of results to return"
)
@click.option(
    "--collection", 
    "-n", 
    type=str,
    help="Name of the collection to use"
)
@click.option(
    "--db-path", 
    "-d", 
    type=str,
    help="Path to the ChromaDB directory"
)
@click.pass_context
def average(ctx, model, phenopackets, results, collection, db_path):
    """
    Run analysis using the 'average' strategy.
    
    This strategy averages embeddings for phenotype terms and queries for
    similar diseases.
    
    Example:
        elder average --model large --phenopackets 5084 --results 10
    """
    # Create runner from configuration
    runner = DiseaseAvgEmbRunner.from_config(
        config_path=ctx.obj.get("config_path"),
        runner_type=RunnerType.AVERAGE.value,
        model_type=model,
        nr_of_phenopackets=phenopackets,
        nr_of_results=results,
        collection_name=collection,
        db_collection_path=db_path,
    )
    
    # Run analysis
    runner.prepare()
    runner.run()


@elder.command()
@click.option(
    "--model", 
    "-m", 
    type=click.Choice([m.value for m in ModelType], case_sensitive=False),
    help="Embedding model to use"
)
@click.option(
    "--phenopackets", 
    "-p", 
    type=str,
    help="Number of phenopackets to process (e.g., 385 or 5084)"
)
@click.option(
    "--results", 
    "-r", 
    type=int,
    help="Number of results to return"
)
@click.option(
    "--collection", 
    "-n", 
    type=str,
    help="Name of the collection to use"
)
@click.option(
    "--db-path", 
    "-d", 
    type=str,
    help="Path to the ChromaDB directory"
)
@click.pass_context
def weighted(ctx, model, phenopackets, results, collection, db_path):
    """
    Run analysis using the 'weighted average' strategy.
    
    This strategy uses weighted average embeddings for phenotype terms
    and queries for similar diseases, accounting for phenotype frequencies.
    
    Example:
        elder weighted --model ada --phenopackets 5084 --results 10
    """
    # Create runner from configuration
    runner = DisWgtAvgEmbRunner.from_config(
        config_path=ctx.obj.get("config_path"),
        runner_type=RunnerType.WEIGHTED_AVERAGE.value,
        model_type=model,
        nr_of_phenopackets=phenopackets,
        nr_of_results=results,
        collection_name=collection,
        db_collection_path=db_path,
    )
    
    # Run analysis
    runner.prepare()
    runner.run()


@elder.command()
@click.option(
    "--model", 
    "-m", 
    type=click.Choice([m.value for m in ModelType], case_sensitive=False),
    help="Embedding model to use"
)
@click.option(
    "--phenopackets", 
    "-p", 
    type=str,
    help="Number of phenopackets to process (e.g., 385 or 5084)"
)
@click.option(
    "--results", 
    "-r", 
    type=int,
    help="Number of results to return"
)
@click.option(
    "--collection", 
    "-n", 
    type=str,
    help="Name of the collection to use"
)
@click.option(
    "--db-path", 
    "-d", 
    type=str,
    help="Path to the ChromaDB directory"
)
@click.pass_context
def bestmatch(ctx, model, phenopackets, results, collection, db_path):
    """
    Run analysis using the 'best match' strategy.
    
    This strategy uses a pairwise comparison approach to find the best matches
    between phenotype terms and diseases.
    
    Example:
        elder bestmatch --model mxbai --phenopackets 5084 --results 10
    """
    # Create runner from configuration
    runner = BestMatchRunner.from_config(
        config_path=ctx.obj.get("config_path"),
        runner_type=RunnerType.BEST_MATCH.value,
        model_type=model,
        nr_of_phenopackets=phenopackets,
        nr_of_results=results,
        collection_name=collection,
        db_collection_path=db_path,
    )
    
    # Run analysis
    runner.prepare()
    runner.run()


@elder.command()
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(file_okay=False, path_type=Path),
              help="Directory to write the config file")
@click.pass_context
def generate_config(ctx, config_file, output):
    """
    Generate a configuration file from a template.
    
    Example:
        elder generate-config template.yaml --output ./configs
    """
    # Load template configuration
    config_data = ConfigLoader.load_from_yaml(config_file)
    
    # Determine output path
    output_path = output or Path(".")
    output_path.mkdir(exist_ok=True, parents=True)
    output_file = output_path / "elder_config.yaml"
    
    # Write configuration
    with open(output_file, "w") as f:
        import yaml
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    
    click.echo(f"Configuration file generated at: {output_file}")


if __name__ == "__main__":
    elder(obj={})