# Elder Examples

This directory contains examples and templates for using the Elder project.

## Configuration Examples

The `configs` directory contains example configuration files for different analysis strategies:

- `average_config.yaml`: Configuration for the average embedding strategy
- `weighted_config.yaml`: Configuration for the weighted average embedding strategy
- `bestmatch_config.yaml`: Configuration for the best match (term-set pairwise comparison) strategy

## Running Examples

### Using the Example Script

The `run_with_config.py` script demonstrates how to use the unified configuration system:

```bash
# Run with average strategy
python run_with_config.py --config configs/average_config.yaml --strategy avg

# Run with weighted average strategy
python run_with_config.py --config configs/weighted_config.yaml --strategy wgt_avg

# Run with best match strategy
python run_with_config.py --config configs/bestmatch_config.yaml --strategy tpc
```

### Using the Elder CLI

You can also use the Elder CLI to run analysis:

```bash
# Run with average strategy
elder average --config configs/average_config.yaml

# Run with weighted average strategy
elder weighted --config configs/weighted_config.yaml

# Run with best match strategy
elder bestmatch --config configs/bestmatch_config.yaml
```

## Configuration Templates

The `templates` directory in the project root contains configuration templates that you can use as a starting point for your own configuration:

```bash
# Generate a configuration file from a template
elder generate-config ../templates/elder_config_template.yaml --output ./configs
```

## Documentation

For more information about the configuration system, see the [Configuration Documentation](../docs/configuration.md).