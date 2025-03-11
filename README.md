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
git clone https://github.com/yourusername/elder.git
cd elder

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
poetry install
```

## Configuration

ELDER uses a YAML configuration file for settings. By default, it looks for `elder_config.yaml` in the current directory.

Example configuration:

```yaml
# Database settings
database:
  chroma_db_path: "/path/to/chromadb/directory"
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

For more details, see the [Configuration Documentation](docs/configuration.md).

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
# Use a configuration file
elder average --config path/to/elder_config.yaml

# Override configuration values
elder average --config path/to/elder_config.yaml --model ada --results 20
```

Available commands:

- `average`: Run analysis with average embedding strategy
- `weighted`: Run analysis with weighted average embedding strategy
- `bestmatch`: Run analysis with best match (term-set pairwise comparison) strategy
- `generate-config`: Generate a configuration file from a template

Run `elder --help` for more information about the available commands and options.

### Python API

You can also use ELDER programmatically in your Python code:

```python
from pheval_elder.dis_avg_emb_runner import DiseaseAvgEmbRunner
from pheval_elder.prepare.config.unified_config import RunnerType

# Create and run a runner with configuration
runner = DiseaseAvgEmbRunner.from_config(
    config_path="elder_config.yaml",
    runner_type=RunnerType.AVERAGE.value,
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

## Data Requirements

ELDER requires the following data:

- **Phenopackets**: JSON files containing phenotype terms
- **ChromaDB Collections**: Vector database collections with embeddings

The paths to these data sources can be specified in the configuration file.

## Development

For development, you can use the example scripts and templates provided:

```bash
# Run with a specific configuration
python examples/run_with_config.py --config examples/configs/average_config.yaml --strategy avg
```

## License

[Your License]