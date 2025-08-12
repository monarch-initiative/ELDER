#!/usr/bin/env python3
"""
CLI for indexing ontologies using CurateGPT's functionality directly.
"""

import os
import sys
import logging
import time

import click
from pathlib import Path
from typing import List, Optional

from curategpt.store.batch_processor import BatchEnhancementProcessor
from curategpt.store.direct_processor import CborgAsyncEnhancementProcessor

from pheval_elder.prepare.core.graph.hpo_clustering import HPOClustering


from pheval_elder.prepare.config import config_loader

try:
    from curategpt.store import ChromaDBAdapter, EnhancedChromaDBAdapter, get_store
    from curategpt.wrappers.ontology import OntologyWrapper
    from oaklib import get_adapter
    from venomx.model.venomx import Index, Model
except ImportError as e:
    sys.exit(1)


def setup_logging():
    logging.basicConfig(
         level=logging.INFO,  # or logging.DEBUG for more verbosity
         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
         stream=sys.stdout
    )

@click.group()
def cli():
    """Index ontologies with CurateGPT for use with ELDER."""
    if not os.environ.get("OPENAI_API_KEY"):
        click.echo("WARNING: OPENAI_API_KEY environment variable is not set.")
        click.echo("Enhanced descriptions will not be generated.")
        click.echo("Set this environment variable before running this command:")
        click.echo("  export OPENAI_API_KEY=your-key-here")


@cli.command(name="index-ontology")
@click.option(
    "--db-path",
    "-p",
    type=click.Path(file_okay=False, path_type=Path),
    help="Path to store the ChromaDB database (defaults to config value)"
)
@click.option(
    "--collection",
    "-c",
    default="hp_standard",
    show_default=True,
    help="Collection name for the indexed ontology"
)
@click.option(
    "--ontology",
    "-o",
    default="sqlite:obo:hp",
    show_default=True,
    help="OAK-style source locator for the ontology"
)
@click.option(
    "--index-fields",
    default="label,definition,relationships",
    show_default=True,
    help="Comma-separated list of fields to index"
)
@click.option(
    "--model",
    "-m",
    default="large3",
    show_default=True,
    help="Embedding model to use (supports shorthand names like ada, small3, large3)"
)
@click.option(
    "--batch-size",
    type=int,
    default=50,
    show_default=True,
    help="Number of terms to process in each batch"
)
@click.option(
    "--enhanced-descriptions",
    is_flag=True,
    help="Enable enhanced descriptions using OpenAI's o1 model"
)
@click.option(
    "--database-type",
    "-D",
    default="enhanced_chromadb",
    show_default=True,
    help="Adapter to use for database, e.g. chromadb.",
)
@click.option(
    "--restrict",
    is_flag=True,
)
def index_ontology(
        db_path: Optional[Path],
        collection: str,
        ontology: str,
        index_fields: str,
        model: str,
        batch_size: int,
        enhanced_descriptions: bool,
        database_type: str,
        restrict: bool
):
    """
    Index an ontology for use with ELDER using CurateGPT.

    This command indexes an ontology using standard descriptions by default.
    With the --enhanced-descriptions flag, it uses OpenAI's o1 model to generate
    rich, detailed descriptions for each term.

    The Human Phenotype Ontology (HP) works best with ELDER's existing analysis tools,
    but other ontologies can also be indexed.

    Examples:
        # Index HP ontology with standard descriptions
        curate-index index-ontology --db-path ./my_db --collection hp_standard

        # Index with enhanced descriptions for richer semantic understanding
        curate-index index-ontology --enhanced-descriptions --collection hp_enhanced

        # Use different model shorthand
        curate-index index-ontology --model ada002
        curate-index index-ontology --model large3

        # Index with custom fields
        curate-index index-ontology --index-fields "label,definition"
    """
    import pdb
    import time
    if enhanced_descriptions and not os.environ.get("OPENAI_API_KEY"):
        click.echo("ERROR: OPENAI_API_KEY environment variable is required for enhanced descriptions.")
        click.echo("Set this environment variable before running this command:")
        click.echo("  export OPENAI_API_KEY=your-key-here")
        sys.exit(1)

    fields_list = [field.strip() for field in index_fields.split(',') if field.strip()]
    include_aliases = "aliases" in fields_list

    if not db_path:
        config = config_loader.load_config()
        db_path = config.get("chroma_db_path")
        if not db_path:
            click.echo("Error: No database path provided and none found in config")
            sys.exit(1)

    if not db_path.exists():
        click.echo(f"Creating directory {db_path}")
        db_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Initializing {'enhanced' if enhanced_descriptions else 'standard'} ChromaDB adapter")
    click.echo(f"Database path: {db_path}")
    click.echo(f"Collection: {collection}")
    click.echo(f"Ontology: {ontology}")
    click.echo(f"Index fields: {', '.join(fields_list)}")
    click.echo(f"Model: {model}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Database type: {database_type}")

    try:
        if enhanced_descriptions:
            adapter = get_store(name=database_type, path=str(db_path))
            click.echo("Using EnhancedChromaDBAdapter with o1 model for rich term descriptions")
        else:
            adapter = get_store(name=database_type,path=str(db_path))
            click.echo("Using standard ChromaDBAdapter")

        oak_adapter = get_adapter(ontology)
        view = OntologyWrapper(oak_adapter=oak_adapter)


        def text_lookup(obj):
            """Custom text extraction function that combines specified fields."""
            parts = []
            for field in fields_list:
                if field != "aliases" and field in obj and obj[field]:
                    if field == "relationships" and isinstance(obj[field], list):
                        # Flatten relationships
                        rel_texts = []
                        for rel in obj[field]:
                            if isinstance(rel, dict):
                                rel_texts.append(f"{rel.get('predicate', '')}: {rel.get('target', '')}")
                        parts.append(" ".join(rel_texts))
                    else:
                        parts.append(str(obj[field]))

            if include_aliases and "aliases" in obj and obj["aliases"]:
                parts.append("Aliases: " + ", ".join(obj["aliases"]))

            return " ".join(parts)

        adapter.text_lookup = text_lookup

        click.echo(f"Loading terms from {ontology}...")

        venomx_obj = Index(
            id=collection,
            embedding_model=Model(name=model if model else None)
        )

        if collection in adapter.list_collection_names():
            click.echo(f"Removing existing collection: {collection}")
            adapter.remove_collection(collection)

        click.echo(f"Indexing terms (this may take a while)...")
        if restrict:
            start = time.time()
            click.echo(f"IRESTRICTe)...")

            adapter.insert(
                view.filtered_o(),
                collection=collection,
                model=model,
                venomx=venomx_obj,
                batch_size=batch_size,
                object_type="OntologyClass"

            )
            end = time.time()
            click.echo(f"Indexed {collection} in {end - start} seconds")

        adapter.insert(
            view.objects(),
            collection=collection,
            model=model,
            venomx=venomx_obj,
            batch_size=batch_size,
            object_type="OntologyClass"

        )

        click.echo(f" Successfully indexed {len(list(view.objects()))} terms in collection '{collection}'")
        click.echo(f"You can now use this collection with ELDER's analysis commands.")

    except Exception as e:
        click.echo(f" Error indexing ontology: {str(e)}")
        import traceback
        click.echo(traceback.format_exc())
        sys.exit(1)

@cli.command(name="index_batch_restricted_ontology")
@click.option(
    "--db-path",
    type=click.Path(),
    help="Path to the database directory"
)
@click.option(
    "--collection",
    "-c",
    required=True,
    help="Name of the collection to create"
)
@click.option(
    "--ontology",
    default="sqlite:obo:hp",
    help="Name of the ontology to index (default: hp)"
)
@click.option(
    "--index-fields",
    default="label,definition,relationships",
    help="Comma-separated list of fields to index"
)
@click.option(
    "--model",
    default="large3",
    help="Embedding model to use"
)
@click.option(
    "--batch-size",
    default=100,
    type=int,
    help="Batch size for processing"
)
@click.option(
    "--openai-model",
    default="openai/gpt-4o",
    help="OpenAI model to use for enhancement"
)
@click.option(
    "--batch-dir",
    type=click.Path(),
    default="./batch_output",
    help="Directory for batch files and results"
)
@click.option(
    "--database-type",
    default="chromadb",
    help="Type of database to use"
)
@click.option(
    "--restrict",
    is_flag=True,
    help="Restrict to HPO terms only"
)
@click.option(
    "--completion-window",
    default="24h",
    help="Completion window for batch API"
)
@click.option(
    "--cache",
    default="batch_output",
    help="Cache directory for openai responses in jsonl format"
)
@click.option(
    "--base-url",
    default="https://api.cborg.lbl.gov",
    help="Base URL for the OpenAI client (or proxy)")
@click.option(
    "--cborg-async",
    is_flag=True,
    default=False,
    help="Use CBOR for API calls and async processing instead of batch API use"
)
@click.option(
    "--test-mode",
    is_flag=True,
    default=False,
    help="Run in test mode with only a few high-numbered HP terms"
)
@click.option(
    "--max-concurrency",
    default=20,
    type=int,
    help="Maxmum number of concurrent API requests (for CBORG async mode)"
)
@click.option(
    "--cborg-api-key",
    type=str,
    help="CBORG API key"
)
def index_with_batch(
        db_path: Optional[Path],
        collection: str,
        ontology: str,
        index_fields: str,
        model: str,
        batch_size: int,
        openai_model: str,
        batch_dir: str,
        database_type: str,
        restrict: bool,
        completion_window: str,
        cache: str,
        base_url: str,
        cborg_async: bool,
        test_mode: bool,
        max_concurrency: int,
        cborg_api_key: str
):
    """
    Index an ontology with enhanced descriptions using OpenAI's Batch API.

    This command indexes an ontology using OpenAI's batch API to generate
    rich, detailed descriptions for each term before storing them in the database.

    Examples:
        # Index HP ontology with enhanced descriptions using batch API
        curate-index index-with-batch --collection hp_enhanced --batch-size 100

        # Use a specific OpenAI model and embedding model
        curate-index index-with-batch --openai-model o1 --model large3 --collection hp_custom
    """
    if not os.environ.get("CBORG_API_KEY") or cborg_api_key:
        import dotenv
        dotenv.load_dotenv()
        cborg_api_key = os.environ.get("CBORG_API_KEY")

    fields_list = [field.strip() for field in index_fields.split(',') if field.strip()]
    include_aliases = "aliases" in fields_list

    batch_output_dir = Path(batch_dir)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Initializing batch-enhanced ChromaDB adapter")
    click.echo(f"Database path: {db_path}")
    click.echo(f"Collection: {collection}")
    click.echo(f"Ontology: {ontology}")
    click.echo(f"Index fields: {', '.join(fields_list)}")
    click.echo(f"Embedding model: {model}")
    click.echo(f"OpenAI model for enhancement: {openai_model}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Batch directory: {batch_output_dir}")
    click.echo(f"Database type: {database_type}")
    click.echo(f"Completion window: {completion_window}")
    click.echo(f"Cache path: {cache}")
    click.echo(f"Async procssing via CBORG: {cborg_async}")
    click.echo(f"Test mode: {test_mode}")
    click.echo(f"Max concurrency: {max_concurrency}")

    try:
        adapter = get_store(name=database_type, path=str(db_path))
        click.echo(f"Using {database_type} adapter")

        oak_adapter = get_adapter(ontology)
        view = OntologyWrapper(oak_adapter=oak_adapter)

        if not cborg_async:
            processor = BatchEnhancementProcessor(
                batch_size=batch_size,
                model=openai_model,
                completion_window=completion_window,
                cache_dir=Path(cache)
            )
        print(cborg_api_key)
        if cborg_async and cborg_api_key:
            processor = CborgAsyncEnhancementProcessor(
                cborg_api_key=cborg_api_key,
                batch_size=1000,
                model="openai/gpt-4o",
                cache_dir=Path("batch_output"),
                max_concurrency=5
            )

        def text_lookup(obj):
            """Custom text extraction function that combines specified fields and uses enhanced description."""
            parts = []
            if "enhanced_description" in obj and obj.get("original_id", "").startswith("HP:"):
                parts.append(obj["enhanced_description"])
            for field in fields_list:
                if field != "aliases" and field in obj and obj[field]:
                    if field == "relationships" and isinstance(obj[field], list):
                        rel_texts = []
                        for rel in obj[field]:
                            if isinstance(rel, dict):
                                rel_texts.append(f"{rel.get('predicate', '')}: {rel.get('target', '')}")
                        parts.append(" ".join(rel_texts))
                    elif field != "definition" or "enhanced_description" not in obj:
                        parts.append(str(obj[field]))

            if include_aliases and "aliases" in obj and obj["aliases"]:
                parts.append("Aliases: " + ", ".join(obj["aliases"]))

            return " ".join(parts)

        adapter.text_lookup = text_lookup

        click.echo(f"Loading terms from {ontology}...")

        venomx_obj = Index(
            id=collection,
            embedding_model=Model(name=model if model else None)
        )

        if collection in adapter.list_collection_names():
            click.echo(f"Removing existing collection: {collection}")
            adapter.remove_collection(collection)

        click.echo(f"Processing terms with batch API (this may take a while)...")

        start_time = time.time()

        # Filter for a small set of high-numbered HP terms
        if test_mode:
            click.echo(f"Running in test mode with high-numbered HP terms...")
            high_hp_terms = []

            filtered_objs = list(view.filtered_o() if restrict else view.objects())

            for obj in filtered_objs:
                term_id = obj.get("original_id", "")
                if term_id.startswith("HP:") and term_id > "HP:0009550":
                    high_hp_terms.append(obj)
                    if len(high_hp_terms) >= 12:
                        break
            click.echo(f"{high_hp_terms}")

            if not high_hp_terms:
                click.echo("No high-numbered HP terms found. Using the first 10 HP terms instead.")
                high_hp_terms = [obj for obj in filtered_objs if obj.get("original_id", "").startswith("HP:")][:10]

            click.echo(
                f"Testing with {len(high_hp_terms)} HP terms: {[obj.get('original_id') for obj in high_hp_terms]}")
            enhanced_objects = processor.process_ontology_in_batches(high_hp_terms, batch_output_dir)

        elif restrict:
            click.echo(f"Using restricted set (HP terms only)...")
            enhanced_objects = processor.process_ontology_in_batches(
                view.filtered_o(),
                batch_output_dir
            )
        else:
            click.echo(f"Using all ontology terms...")
            enhanced_objects = processor.process_ontology_in_batches(
                view.objects(),
                batch_output_dir
            )


        click.echo(f"Indexing enhanced terms...")
        adapter.insert(
            enhanced_objects,
            collection=collection,
            model=model,
            venomx=venomx_obj,
            batch_size=batch_size,
            object_type="OntologyClass"
        )

        end_time = time.time()
        click.echo(f" Successfully indexed collection '{collection}' in {end_time - start_time:.2f} seconds")
        click.echo(f"You can now use this collection with ELDER's analysis commands.")

    except Exception as e:
        click.echo(f" Error indexing ontology: {str(e)}")
        import traceback
        click.echo(traceback.format_exc())
        sys.exit(1)



@cli.command(name="restore_enhanced_descriptions")
@click.option(
    "--db-path",
    type=click.Path(),
    help="Path to the database directory"
)
@click.option(
    "--collection",
    required=True,
    help="Name of the collection to create or update"
)
@click.option(
    "--jsonl-file",
    type=click.Path(exists=True),
    help="Path to the JSONL file containing enhanced descriptions"
)
@click.option(
    "--batch-dir",
    type=click.Path(exists=True),
    help="Directory containing batch results to recover from"
)
@click.option(
    "--model",
    default="large3",
    help="Embedding model to use"
)
@click.option(
    "--batch-size",
    default=100,
    type=int,
    help="Batch size for processing"
)
@click.option(
    "--database-type",
    default="chromadb",
    help="Type of database to use"
)
@click.option(
    "--index-fields",
    default="label,definition,relationships",
    help="Comma-separated list of fields to index"
)
@click.option(
    "--ontology",
    default="hp",
    help="Name of the ontology to use for term info"
)
@click.option(
    "--recreate-collection",
    is_flag=True,
    help="Recreate the collection if it exists"
)
def restore_enhanced_descriptions(
        db_path: Optional[Path],
        collection: str,
        jsonl_file: Optional[str],
        batch_dir: Optional[str],
        model: str,
        batch_size: int,
        database_type: str,
        index_fields: str,
        ontology: str,
        recreate_collection: bool
):
    """
    Restore and index enhanced descriptions from a saved JSONL file or batch results.

    This command allows you to recover from a failed indexing operation by using
    previously generated enhanced descriptions. It can restore from:

    1. A JSONL file containing enhanced descriptions (--jsonl-file)
    2. Batch API results in a batch directory (--batch-dir)

    You must specify at least one of these sources.

    Examples:
        # Restore from a JSONL file
        curate-index ontology restore_enhanced_descriptions \\
          --collection hp_enhanced \\
          --jsonl-file ./batch_output/enhanced_descriptions.jsonl

        # Restore from batch results directory
        curate-index ontology restore_enhanced_descriptions \\
          --collection hp_enhanced \\
          --batch-dir ./batch_output
    """
    import json
    import glob
    import time

    if not jsonl_file and not batch_dir:
        click.echo("ERROR: You must specify either --jsonl-file or --batch-dir")
        sys.exit(1)

    fields_list = [field.strip() for field in index_fields.split(',') if field.strip()]
    include_aliases = "aliases" in fields_list

    click.echo(f"Initializing restore operation")
    click.echo(f"Database path: {db_path}")
    click.echo(f"Collection: {collection}")
    click.echo(f"Database type: {database_type}")
    click.echo(f"Index fields: {', '.join(fields_list)}")
    click.echo(f"Embedding model: {model}")

    try:
        adapter = get_store(name=database_type, path=str(db_path))
        click.echo(f"Using {database_type} adapter")

        oak_adapter = get_adapter(ontology)
        view = OntologyWrapper(oak_adapter=oak_adapter)

        def text_lookup(obj):
            """Custom text extraction function that prioritizes enhanced descriptions."""
            parts = []
            if "enhanced_description" in obj and obj.get("original_id", "").startswith("HP:"):
                parts.append(obj["enhanced_description"])
            for field in fields_list:
                if field != "aliases" and field in obj and obj[field]:
                    if field == "relationships" and isinstance(obj[field], list):
                        rel_texts = []
                        for rel in obj[field]:
                            if isinstance(rel, dict):
                                rel_texts.append(f"{rel.get('predicate', '')}: {rel.get('target', '')}")
                        parts.append(" ".join(rel_texts))
                    elif field != "definition" or "enhanced_description" not in obj:
                        parts.append(str(obj[field]))

            if include_aliases and "aliases" in obj and obj["aliases"]:
                parts.append("Aliases: " + ", ".join(obj["aliases"]))

            return " ".join(parts)

        adapter.text_lookup = text_lookup

        enhanced_descriptions = {}

        if jsonl_file:
            click.echo(f"Loading enhanced descriptions from {jsonl_file}")
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if "id" in record and "enhanced_description" in record:
                            enhanced_descriptions[record["id"]] = record
                    except json.JSONDecodeError:
                        continue
            click.echo(f"Loaded {len(enhanced_descriptions)} descriptions from JSONL file")

        if batch_dir:
            click.echo(f"Loading enhanced descriptions from batch results in {batch_dir}")
            batch_path = Path(batch_dir)

            result_files = glob.glob(str(batch_path / "*.jsonl"))
            processed_count = 0

            for file_path in result_files:
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                if "custom_id" in data and "response" in data and data["response"].get(
                                        "status_code") == 200:
                                    term_id = data["custom_id"]
                                    body = data["response"].get("body", {})
                                    choices = body.get("choices", [])

                                    if choices and term_id.startswith("HP:"):
                                        content = choices[0].get("message", {}).get("content", "")
                                        if content and term_id not in enhanced_descriptions:
                                            enhanced_descriptions[term_id] = {
                                                "id": term_id,
                                                "enhanced_description": content
                                            }
                                            processed_count += 1
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    click.echo(f"Error processing file {file_path}: {str(e)}")

            click.echo(f"Loaded {processed_count} additional descriptions from batch results")

        if not enhanced_descriptions:
            click.echo("No enhanced descriptions found in the specified sources")
            sys.exit(1)

        click.echo(f"Total enhanced descriptions loaded: {len(enhanced_descriptions)}")

        venomx_obj = Index(
            id=collection,
            embedding_model=Model(name=model if model else None)
        )

        if recreate_collection and collection in adapter.list_collection_names():
            click.echo(f"Removing existing collection: {collection}")
            adapter.remove_collection(collection)

        if collection not in adapter.list_collection_names():
            click.echo(f"Creating new collection: {collection}")

        click.echo(f"Loading base ontology objects...")

        def enhanced_objects_generator():
            """Generate enhanced objects from base ontology."""
            base_objects_iter = view.filtered_o() if all(
                term_id.startswith("HP:") for term_id in enhanced_descriptions) else view.objects()

            for obj in base_objects_iter:
                term_id = obj.get("original_id", "")
                if term_id in enhanced_descriptions:
                    obj["enhanced_description"] = enhanced_descriptions[term_id]["enhanced_description"]

                    saved_record = enhanced_descriptions[term_id]
                    for field in ["label", "definition", "relationships"]:
                        if field in saved_record and field not in obj:
                            obj[field] = saved_record[field]

                yield obj

        click.echo(f"Indexing enhanced objects...")
        start_time = time.time()

        adapter.insert(
            enhanced_objects_generator(),
            collection=collection,
            model=model,
            venomx=venomx_obj,
            batch_size=batch_size,
            object_type="OntologyClass"
        )

        end_time = time.time()
        click.echo(
            f" Successfully restored and indexed collection '{collection}' in {end_time - start_time:.2f} seconds")

    except Exception as e:
        click.echo(f" Error restoring enhanced descriptions: {str(e)}")
        import traceback
        click.echo(traceback.format_exc())
        sys.exit(1)


import json
from pathlib import Path


def save_cache_to_jsonl(embedding_function, output_dir: Optional[Path] = None):
    """
    Save the enhanced descriptions cache from the given enhanced embedding function instance
    to a JSONL file. Each JSON object will have the keys:
      - "term_id": the term's identifier
      - "enhanced_description": the generated enhanced description
    """
    if output_dir is None:
        output_dir = Path.cwd() / "jsonl_enhanced_description"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "enhanced_descriptions_cache.jsonl"

    with output_file.open("w", encoding="utf-8") as f:
        for term_id, description in embedding_function.enhanced_descriptions_cache.items():
            record = {
                "term_id": term_id,
                "enhanced_description": description
            }
            f.write(json.dumps(record) + "\n")

    print(f"Enhanced descriptions cache saved to {output_file}")

@cli.command(name="save-cache")
@click.option("--output-dir", type=click.Path(), default="./jsonl_enhanced_description",
              help="Directory to save the JSONL file")
def save_cache_cli(output_dir):
    """
    Save the enhanced descriptions that have been generated (and cached) into a JSONL file.
    Each line of the file is a JSON object with keys "term_id" and "enhanced_description".
    """
    adapter = get_store(name="enhanced_chromadb", path="None")
    ef = adapter._embedding_function()
    save_cache_to_jsonl(ef, Path(output_dir))



@cli.command(name="search")
@click.option(
    "--db-path",
    type=click.Path(file_okay=False, path_type=Path),
    help="Path to ChromaDB database (defaults to config value)"
)
@click.option(
    "--collection",
    default="hp_standard",
    show_default=True,
    help="Collection name containing the indexed terms"
)
@click.option(
    "--limit",
    type=int,
    default=10,
    show_default=True,
    help="Maximum number of results to return"
)
@click.option(
    "--enhanced",
    is_flag=True,
    help="Use enhanced adapter (required if collection was indexed with enhanced descriptions)"
)
@click.argument("query")
def search(
    db_path: Optional[Path],
    collection: str,
    limit: int,
    enhanced: bool,
    query: str
):
    """
    Search an indexed ontology.
    
    This command searches terms in a collection indexed by CurateGPT,
    making it useful for testing and exploring the embeddings.
    
    Examples:
        # Search for terms related to heart problems
        curate-index search "cardiac arrhythmia"
        
        # Search in a specific collection with enhanced adapter
        curate-index search --collection hp_enhanced --enhanced "joint pain"
    """
    if not db_path:
        config = config_loader.load_config()
        db_path = config.get("chroma_db_path")
        if not db_path:
            click.echo("Error: No database path provided and none found in config")
            sys.exit(1)

    try:
        if enhanced:
            adapter = EnhancedChromaDBAdapter(path=str(db_path))
        else:
            adapter = ChromaDBAdapter(path=str(db_path))

        if collection not in adapter.list_collection_names():
            click.echo(f"Error: Collection '{collection}' not found in database at {db_path}")
            sys.exit(1)

        click.echo(f"Searching for '{query}' in collection '{collection}'...")
        results = list(adapter.search(
            query,
            collection=collection,
            limit=limit
        ))

        if not results:
            click.echo("No results found.")
            return

        click.echo(f"\nFound {len(results)} results:\n")
        for i, (obj, score, _) in enumerate(results, 1):
            click.echo(f"{i}. [{obj.get('original_id')}] {obj.get('label')} (Score: {score:.4f})")
            if 'definition' in obj:
                definition = obj.get('definition', '')
                if len(definition) > 100:
                    definition = definition[:97] + "..."
                click.echo(f"   Definition: {definition}")
            click.echo("")

    except Exception as e:
        click.echo(f"Error searching: {str(e)}")
        import traceback
        click.echo(traceback.format_exc())
        sys.exit(1)


@cli.command(name="info")
@click.option(
    "--db-path",
    type=click.Path(file_okay=False, path_type=Path),
    help="Path to ChromaDB database (defaults to config value)"
)
@click.option(
    "--collection",
    help="Collection name to show info for (defaults to all collections)"
)
def info(
    db_path: Optional[Path],
    collection: Optional[str]
):
    """
    Show information about indexed collections.
    
    This command displays information about the collections in a database,
    including counts and sample entries.
    
    Examples:
        # Show info for all collections
        curate-index info
        
        # Show info for a specific collection
        curate-index info --collection hp_enhanced
    """
    if not db_path:
        config = config_loader.load_config()
        db_path = config.get("chroma_db_path")
        if not db_path:
            click.echo("Error: No database path provided and none found in config")
            sys.exit(1)
    
    try:
        adapter = ChromaDBAdapter(path=str(db_path))

        if collection and collection not in adapter.list_collection_names():
            click.echo(f"Error: Collection '{collection}' not found in database at {db_path}")
            sys.exit(1)
        
        collection_names = adapter.list_collection_names()
        if not collection_names:
            click.echo("No collections found in database.")
            return
        
        if collection:
            coll_obj = adapter.client.get_collection(collection)
            count = coll_obj.count()
            metadata = coll_obj.metadata

            click.echo(f"Collection: {collection}")
            click.echo(f"Count: {count} terms")
            click.echo(f"Metadata: {metadata}")
            for key, value in metadata.items():
                if key == "_venomx":
                    click.echo("  venomx: [complex object]")
                else:
                    click.echo(f"  {key}: {value}")
            
            click.echo("\nSample entries:")
            peek = coll_obj.peek(limit=3)
            for i, (id, doc, meta) in enumerate(zip(peek["ids"], peek["documents"], peek["metadatas"]), 1):
                click.echo(f"{i}. [{id}] {meta.get('label', 'No label')}")
                doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
                click.echo(f"   Text: {doc_preview}")

        else:
            click.echo(f"Found {len(collection_names)} collections:")
            for coll_name in collection_names:
                try:
                    count = adapter.client.get_collection(coll_name).count()
                    click.echo(f"- {coll_name} ({count} entries)")
                except Exception as e:
                    click.echo(f"- {coll_name} (Error getting count: {e})")
        
    except Exception as e:
        click.echo(f"Error getting info: {str(e)}")
        import traceback
        click.echo(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    setup_logging()
    cli()