# Elder Configuration System

The Elder project uses a unified configuration system that allows for consistent configuration across different components and runners. This document describes how to use the configuration system.

## Configuration File

Elder uses a YAML configuration file to store settings. By default, it looks for a file named `elder_config.yaml` in the current directory or one level up. You can also specify a configuration file using the `--config` option in the CLI or by setting the `ELDER_CONFIG` environment variable.

## Configuration Structure

The configuration file is organized into sections:

### Database Settings

```yaml
database:
  chroma_db_path: "/path/to/chromadb/directory"
  collection_name: "lrd_hpo_embeddings"
  similarity_measure: "COSINE"  # COSINE, EUCLIDEAN, DOT_PRODUCT
```

### Runner Settings

```yaml
runner:
  runner_type: "avg"  # avg, wgt_avg, tpc, graph_emb
  model_type: "large"  # small, large, ada, mxbai, bge, nomic, custom
  nr_of_phenopackets: "5084"
  nr_of_results: 10
  custom_model_name: ""  # Optional, used when model_type is "custom"
  model_path: ""  # Optional, if empty will be derived from model_type or custom_model_name
```

### Processing Settings

```yaml
processing:
  use_multiprocessing: true
  num_workers: null  # null = use all available cores
  batch_size: 100
```

### Output Settings

```yaml
output:
  output_dir: "./output"
  results_dir_name: ""  # Optional, will be auto-generated if empty
  results_sub_dir: ""  # Optional, will be auto-generated if empty
```

### Data Paths

```yaml
data_paths:
  phenopackets_dir: "5084_phenopackets"
  test_phenopackets_dir: "10_z_phenopackets"
  lirical_phenopackets_dir: "LIRICAL_phenopackets"
```

### Feature Flags

```yaml
features:
  use_custom_descriptions: false
  use_graph_embeddings: false
```

## Using Configuration in Code

### Loading Configuration

```python
from pheval_elder.prepare.config.unified_config import get_config

# Load configuration from default locations
config = get_config()

# Load configuration from a specific file
config = get_config("/path/to/config.yaml")
```

### Setting Configuration Programmatically

```python
from pheval_elder.prepare.config.unified_config import set_config, ElderConfig, DatabaseConfig, RunnerConfig

# Create configuration objects
db_config = DatabaseConfig(chroma_db_path="/path/to/chromadb")
runner_config = RunnerConfig(runner_type="avg", model_type="large", nr_of_phenopackets="5084")

# Create and set Elder configuration
config = ElderConfig(db=db_config, runner=runner_config)
set_config(config)
```

### Using Runners with Configuration

```python
from pheval_elder.dis_avg_emb_runner import DiseaseAvgEmbRunner
from pheval_elder.prepare.config.unified_config import RunnerType

# Create runner using configuration
runner = DiseaseAvgEmbRunner.from_config()

# Create runner with specific configuration
runner = DiseaseAvgEmbRunner.from_config(
    config_path="path/to/config.yaml",
    runner_type=RunnerType.AVERAGE.value
)

# Run analysis
runner.prepare()
runner.run()
```

## Command-Line Interface

Elder provides a command-line interface for running analysis with the unified configuration system:

```bash
# Run with average strategy
elder average --config path/to/config.yaml

# Run with weighted average strategy
elder weighted --config path/to/config.yaml

# Run with best match strategy
elder bestmatch --config path/to/config.yaml

# Generate configuration file from template
elder generate-config template.yaml --output ./configs
```

You can override configuration values using command-line options:

```bash
elder average --model large --phenopackets 5084 --results 10 --collection lrd_hpo_embeddings
```

## Configuration Templates

Elder provides configuration templates for common use cases:

- `templates/elder_config_template.yaml`: Base template with comments
- `examples/configs/average_config.yaml`: Configuration for average embedding strategy
- `examples/configs/weighted_config.yaml`: Configuration for weighted average embedding strategy
- `examples/configs/bestmatch_config.yaml`: Configuration for best match strategy

You can use these templates as a starting point for your own configuration.