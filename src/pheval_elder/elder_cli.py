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
    default="elder_config.yaml",
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
    runner = DiseaseAvgEmbRunner.from_config(
            config_path=ctx.obj.get("config_path"),
            config_overrides={
            "runner_type": RunnerType.AVERAGE.value,
            "model_type" : model,
            "nr_of_phenopackets" : phenopackets,
            "nr_of_results" : results,
            "collection_name" : collection,
            "db_collection_path" : db_path,
        }
    )
    
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
    runner = DisWgtAvgEmbRunner.from_config(
        config_path=ctx.obj.get("config_path"),
        config_overrides={
            "runner_type": RunnerType.WEIGHTED_AVERAGE.value,
            "model_type": model,
            "nr_of_phenopackets": phenopackets,
            "nr_of_results": results,
            "collection_name": collection,
            "db_collection_path": db_path,
        }
    )
    
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
    runner = BestMatchRunner.from_config(
        config_path=ctx.obj.get("config_path"),
        config_overrides={
            "runner_type": RunnerType.BEST_MATCH.value,
            "model_type" : model,
            "nr_of_phenopackets" : phenopackets,
            "nr_of_results" : results,
            "collection_name" : collection,
            "db_collection_path" : db_path,
        }
    )
    
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
    config_data = ConfigLoader.load_from_yaml(config_file)
    
    output_path = output or Path(".")
    output_path.mkdir(exist_ok=True, parents=True)
    output_file = output_path / "elder_config.yaml"
    
    with open(output_file, "w") as f:
        import yaml
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    
    click.echo(f"Configuration file generated at: {output_file}")


@elder.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to save the downloaded data"
)
@click.option(
    "--repo-id",
    default="iQuxLE/ELDER_STATS",
    show_default=True,
    help="Hugging Face repository ID containing ELDER data"
)
@click.option(
    "--embeddings",
    type=click.Choice(["ada002", "nomic", "small3", "mxbai-l", "large3", "bge-m3", "all"]),
    help="Embedding collection(s) to download. Use 'all' for all collections."
)
@click.option(
    "--phenopackets",
    is_flag=True,
    help="Download phenopackets dataset"
)
@click.option(
    "--exomiser",
    is_flag=True,
    help="Download exomiser results"
)
@click.option(
    "--force",
    is_flag=True,
    help="Force download even if files exist"
)
def download(output_dir, repo_id, embeddings, phenopackets, exomiser, force):
    """
    Download collections from Hugging Face.

    This command downloads pre-indexed embedding collections, phenopackets, and/or
    exomiser results from the Hugging Face repository.

    Examples:
        # Download all embedding collections
        elder download --output-dir ./data --embeddings all

        # Download just the ada002 embedding collection
        elder download --output-dir ./data --embeddings ada002

        # Download phenopackets and exomiser results
        elder download --output-dir ./data --phenopackets --exomiser

        # Download everything
        elder download --output-dir ./data --embeddings all --phenopackets --exomiser
    """
    import os
    import shutil
    import tempfile
    import requests
    from tqdm import tqdm
    from huggingface_hub import snapshot_download, hf_hub_download, HfFileSystem

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize HuggingFace filesystem for listing files
    hf_fs = HfFileSystem()

    def download_file(url, dest_path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))

            with open(dest_path, 'wb') as f, tqdm(
                    desc=f"Downloading {os.path.basename(dest_path)}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

    # Download embedding collections
    if embeddings:
        collections = ["ada002", "nomic", "small3", "mxbai-l", "large3", "bge-m3"] if embeddings == "all" else [
            embeddings]

        for collection in collections:
            click.echo(f"\nDownloading {collection} embedding collection...")
            collection_dir = output_dir / collection

            if collection_dir.exists() and not force:
                click.echo(f"Directory {collection_dir} already exists. Use --force to overwrite.")
                continue

            collection_dir.mkdir(parents=True, exist_ok=True)

            try:
                files = hf_fs.ls(f"{repo_id}/{collection}", detail=False)

                if not files:
                    click.echo(f"No files found in {repo_id}/{collection}")
                    continue

                for file_path in files:
                    file_name = os.path.basename(file_path)
                    local_path = collection_dir / file_name

                    hf_hub_download(
                        repo_id=repo_id,
                        filename=f"{collection}/{file_name}",
                        local_dir=collection_dir,
                        local_dir_use_symlinks=False,
                        force_download=force
                    )

                click.echo(f"Successfully downloaded {collection} embedding collection to {collection_dir}")
            except Exception as e:
                click.echo(f"Error downloading {collection} collection: {str(e)}")

    # Download phenopackets
    if phenopackets:
        click.echo("\nDownloading phenopackets...")
        phenopackets_dir = output_dir / "5084_phenopackets"

        if phenopackets_dir.exists() and not force:
            click.echo(f"Directory {phenopackets_dir} already exists. Use --force to overwrite.")
        else:
            phenopackets_dir.mkdir(parents=True, exist_ok=True)

            try:
                phenopacket_files = hf_fs.ls(f"{repo_id}/5084_phenopackets", detail=False)

                if not phenopacket_files:
                    click.echo(f"No phenopacket files found in {repo_id}/5084_phenopackets")
                else:
                    for file_path in phenopacket_files:
                        file_name = os.path.basename(file_path)

                        hf_hub_download(
                            repo_id=repo_id,
                            filename=f"5084_phenopackets/{file_name}",
                            local_dir=output_dir,
                            local_dir_use_symlinks=False,
                            force_download=force
                        )

                click.echo(f"Successfully downloaded phenopackets to {phenopackets_dir}")
            except Exception as e:
                click.echo(f"Error downloading phenopackets: {str(e)}")

    # Download exomiser results
    if exomiser:
        click.echo("\nDownloading exomiser results...")
        exomiser_dir = output_dir / "exomiser-results"

        if exomiser_dir.exists() and not force:
            click.echo(f"Directory {exomiser_dir} already exists. Use --force to overwrite.")
        else:
            exomiser_dir.mkdir(parents=True, exist_ok=True)

            try:
                exomiser_files = hf_fs.ls(f"{repo_id}/exomiser-results", detail=False)

                if not exomiser_files:
                    click.echo(f"No exomiser files found in {repo_id}/exomiser-results")
                else:
                    for file_path in exomiser_files:
                        file_name = os.path.basename(file_path)

                        hf_hub_download(
                            repo_id=repo_id,
                            filename=f"exomiser-results/{file_name}",
                            local_dir=output_dir,
                            local_dir_use_symlinks=False,
                            force_download=force
                        )

                click.echo(f"Successfully downloaded exomiser results to {exomiser_dir}")
            except Exception as e:
                click.echo(f"Error downloading exomiser results: {str(e)}")

    # Show summary
    click.echo("\nDownload Summary:")
    if embeddings:
        collections = ["ada002", "nomic", "small3", "mxbai-l", "large3", "bge-m3"] if embeddings == "all" else [
            embeddings]
        for collection in collections:
            collection_dir = output_dir / collection
            if collection_dir.exists():
                click.echo(f"- {collection} embedding collection: ✅")
            else:
                click.echo(f"- {collection} embedding collection: ❌")

    if phenopackets:
        phenopackets_dir = output_dir / "5084_phenopackets"
        if phenopackets_dir.exists():
            click.echo(f"- Phenopackets: ✅")
        else:
            click.echo(f"- Phenopackets: ❌")

    if exomiser:
        exomiser_dir = output_dir / "exomiser-results"
        if exomiser_dir.exists():
            click.echo(f"- Exomiser results: ✅")
        else:
            click.echo(f"- Exomiser results: ❌")

    if not any([embeddings, phenopackets, exomiser]):
        click.echo("❗ No data specified for download. Use --embeddings, --phenopackets, or --exomiser.")
        click.echo("Example: elder download --output-dir ./data --embeddings ada002 --phenopackets")

if __name__ == "__main__":
    # call group callback and initialize click context object
    elder(obj={})