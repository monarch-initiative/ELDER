import logging
from pathlib import Path

import click
import pandas as pd
import yaml
from click_default_group import DefaultGroup, __version__
from venomx.model.venomx import Index

from pheval_elder.huggingface.huggingface_agent import HuggingFaceAgent

__all__ = [
    "main",
]

from pheval_elder.metadata.metadata import Metadata

from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager

collection_option = click.option("-c", "--collection", help="Collection within the database.")
path_option = click.option("-p", "--path", help="Path to a file or directory for database.")

@click.group(
    cls=DefaultGroup,
    default="search",
    default_if_no_args=True,
)
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

@main.group()
def embeddings():
    """Command group for handling embeddings."""
    pass

@embeddings.command(name="download")
@path_option
@collection_option
@click.option(
    "--repo-id",
    required=True,
    help="Repository ID on Hugging Face, e.g., 'biomedical-translator/[repo_name]'.",
)
@click.option(
    "--embeddings-filename", "-ef",
    type=str,
    required=True,
    default="embeddings.parquet"
)
@click.option(
    "--metadata-filename", "-mf",
    type=str,
    required=False,
    default="metadata.yaml"
)
# @database_type_option
def download_embeddings(path, collection, repo_id, embeddings_filename, metadata_filename):
    """
    Download dataset and insert into a collection

    Example:
    embeddings download -p ./db --repo-id iQuxLE/example7 --collection hf_collection --embeddings-filename embeddings.parquet --metadata-filename metadata.yaml
    """

    db = ChromaDBManager(path=path)
    parquet_download = None
    metadata_download = None
    store_objects = None
    agent = HuggingFaceAgent()
    try:
        if embeddings_filename:
            embedding_filename = repo_id + "/" + embeddings_filename
            parquet_download = agent.cached_download(repo_id=repo_id,
                                     repo_type="dataset",
                                     filename=embedding_filename
                                     )
        if metadata_filename:
            metadata_filename = repo_id + "/" + metadata_filename
            metadata_download = agent.api.hf_hub_download(repo_id=repo_id,
                                               repo_type="dataset",
                                               filename=metadata_filename
            )

    except Exception as e:
        click.echo(f"Error meanwhile downloading: {e}")

    try:
        if parquet_download.endswith(".parquet"):
            df = pd.read_parquet(Path(parquet_download))
            store_objects = [
                {
                    "metadata": row.iloc[0]['metadata'],
                    "embeddings": row.iloc[1],
                    "document": row.iloc[2]
                } for _, row in df.iterrows()
            ]

        if metadata_download.endswith(".yaml"):
            # populate venomx from file
            with open(metadata_download, "r") as infile:
                _meta = yaml.safe_load(infile)
                try:
                    venomx_data = _meta.pop("venomx", None)
                    venomx_obj = Index(**venomx_data) if venomx_data else None
                    metadata_obj = Metadata(
                        **_meta,
                        venomx=venomx_obj
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error parsing metadata file: {e}. Downloaded metadata is not in the correct format.") from e

        objects = [{k:v for k, v in obj.items()} for obj in store_objects]

        # print(objects[0])

        # print(store_objects[0])
        # print("\n\n\n")
        # print(store_objects[1]['metadata'].get('id', None))
        # for idx, obj in enumerate(store_objects):
        #     metadata = obj.get('metadata', {})
        #     id_value = metadata.get('id', None)
        #
        #     if id_value is None:
        #         print(f"Object at index {idx} has a None 'id'.")
        #         print(f"Label: {metadata.get('label')}")
        #         print(f"Document: {obj.get('document')}")
        db.insert_from_huggingface(collection=collection, objs=objects, venomx=metadata_obj)
    except Exception as e:
        raise e