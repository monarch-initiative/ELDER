# ELDER: Embeddings-based Large-scale Differential Diagnostic Engine with Retrieval

ELDER is an algorithm that uses text embeddings for differential diagnosis. It takes phenotype terms as input and queries a vector database of diseases to find the most similar diseases.

## Overview

ELDER provides several strategies for analyzing phenotype terms:

- **Average Embedding**: Calculates the average embedding for all phenotype terms and queries for similar diseases
- **Weighted Average Embedding**: Calculates a weighted average embedding based on phenotype frequencies
- **Best Match (Term-set Pairwise Comparison)**: Finds the best matches between phenotype terms and diseases

## Installation

```bash
# Clone the repository
git clone https://github.com/monarch-initiative/ELDER.git
cd elder

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
poetry install
```

## Data Requirements

ELDER requires the following data:

- **Phenopackets**: JSON files containing phenotype terms
- **ChromaDB Collections**: Vector database collections with embeddings
- **exomiser-results**: TSV files containing the exomiser results when run in phenotype only. Only needed for Pheval Benchmark.

The paths to these data sources can be specified in the configuration file.

### Fetching Existing Embeddings
Pre-built embeddings are available on [HuggingFace](https://huggingface.co/iQuxLE) for the following collections
```bash

    embeddings download -p /path/to/chromadb/ --repo-id iQuxLE/large3_lrd_hpo_embedding --collection lrd_hpo_embeddings --embeddings-filename embeddings.parquet --metadata-filename metadata.yaml

```

## Configuration

ELDER uses a YAML configuration file for settings. By default, it looks for `elder_config.yaml` in the current directory.

Example configuration:

```yaml
# Database settings
database:
  chroma_db_path: "/path/to/chromadb/"
  collection_name: "lrd_hpo_embeddings"
  similarity_measure: "COSINE"

# Runner settings
runner:
  runner_type: "avg"  # avg, wgt_avg, tpc
  model_type: "large"  # small, large, ada, mxbai, bge, nomic, custom
  nr_of_phenopackets: "5084"
  nr_of_results: 10

# Data paths
data_paths:
  phenopackets_dir: "5084_phenopackets"
```

### Creating a Configuration File

You can create a configuration file in several ways:

1. **Use a template**:
   ```bash
   # Generate a config file from a template
   elder generate-config templates/elder_config_template.yaml --output .
   ```

2. **Copy an example**:
   ```bash
   # Copy an example configuration for average embedding strategy
   cp examples/configs/average_config.yaml elder_config.yaml
   ```

3. **Create from scratch**:
   Create a new file named `elder_config.yaml` and add the necessary settings.

## Usage

### Command Line Interface

ELDER provides a command-line interface (CLI) for running analysis:

```bash
# Run with average strategy
elder average --model large --phenopackets 5084 --results 10 --collection lrd_hpo_embeddings --db-path /path/to/chromadb

# Run with weighted average strategy
elder weighted --model ada --phenopackets 5084 --results 10 --collection ada002_lrd_hpo_embeddings --db-path /path/to/chromadb

# Run with best match strategy
elder bestmatch --model mxbai --phenopackets 5084 --results 10 --collection mxbai_lrd_hpo_embeddings --db-path /path/to/chromadb
```

If you have a configuration file, you can use it like this:

```bash
# Use the default configuration file elder_config.yaml
elder average

# Override configuration values
elder --config path/to/custom_elder_config.yaml average  --model ada --results 20
```

Available commands:

- `average`: Run analysis with average embedding strategy
- `weighted`: Run analysis with weighted average embedding strategy
- `bestmatch`: Run analysis with best match (term-set pairwise comparison) strategy
- `generate-config`: Generate a configuration file from a template
- `curate-index`: Index ontologies, search, and manage collections using CurateGPT integration

Run `elder --help` for more information about the available commands and options.

### Python API

You can also use ELDER programmatically in your Python code:

```python
from pheval_elder.dis_avg_emb_runner import DiseaseAvgEmbRunner
from pheval_elder.prepare.config.unified_config import RunnerType

# Create and run a runner with configuration
runner = DiseaseAvgEmbRunner.from_config(
    config_path="elder_config.yaml",
    config_overrides={
                "runner_type": RunnerType.AVERAGE.value,
            }
)

# Run analysis
runner.prepare()
runner.run()
```

### Example Scripts

The `examples` directory contains sample scripts for running ELDER:

```bash
# Run with a specific configuration
python examples/run_with_config.py --config examples/configs/average_config.yaml --strategy avg
```

## Creating Your Own Embeddings
EDIT: This is currently setup to work only with the CBORG AI Portal
If you prefer to create your own embeddings rather than using pre-built ones, you can use the `curate-index` command which provides direct integration with CurateGPT:

```bash
# Index HP ontology with standard descriptions
curate-index index-ontology --db-path ./my_db --collection hp_standard

# Index with enhanced descriptions (requires OpenAI API key)
curate-index index-ontology --enhanced-descriptions --collection hp_enhanced

# Include aliases in the indexed fields
curate-index index-ontology --index-fields "label,definition,relationships,aliases"

# Use different model shorthand names
curate-index index-ontology --model ada002  # OpenAI's text-embedding-ada-002
curate-index index-ontology --model large3  # OpenAI's text-embedding-3-large
curate-index index-ontology --model bge-m3  # BAAI/bge-m3

# Search in indexed collections
curate-index search "cardiac arrhythmia"

# View collection information
curate-index info
```

The Human Phenotype Ontology (HP) works best with ELDER's existing analysis tools, as that's what the phenopackets reference.

#### Enhanced Descriptions for Better Quality

For the best quality embeddings, especially for rare phenotypes, the enhanced descriptions option uses OpenAI's o1 model to generate:

- Detailed clinical information about etiology and associated conditions
- Anatomical structures and physiological processes involved
- Presentation across different severities and contexts
- Distinguishing features from similar phenotypes

This results in embeddings that better capture the semantic meaning and clinical context of each term.

## Development

For development, you can use the example scripts and templates provided:

```bash
# Run with a specific configuration
python examples/run_with_config.py --config examples/configs/average_config.yaml --strategy avg
```

## License
