"""
Unified configuration system for Elder.

This module provides a centralized configuration system that works across all runners
and components in the Elder project. It combines file-based configuration with
programmatic override options.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import yaml

from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures


class RunnerType(str, Enum):
    """Types of runners available in Elder."""
    AVERAGE = "avg"
    WEIGHTED_AVERAGE = "wgt_avg"
    BEST_MATCH = "tpc"
    GRAPH_EMBEDDINGS = "graph_emb"


class ModelType(str, Enum):
    """Embedding model types supported by Elder."""
    SMALL = "small"
    LARGE = "large"
    ADA = "ada"
    MXBAI = "mxbai"
    BGE = "bge"
    NOMIC = "nomic"
    CUSTOM = "custom"


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    chroma_db_path: str
    collection_name: str = "lrd_hpo_embeddings"
    similarity_measure: SimilarityMeasures = SimilarityMeasures.COSINE


@dataclass
class RunnerConfig:
    """Configuration for a specific runner."""
    runner_type: RunnerType
    model_type: ModelType
    nr_of_phenopackets: str
    nr_of_results: int = 10
    custom_model_name: Optional[str] = None
    model_path: Optional[str] = None


@dataclass
class ProcessingConfig:
    """Configuration for processing options."""
    use_multiprocessing: bool = True
    num_workers: Optional[int] = None
    batch_size: int = 100


@dataclass
class OutputConfig:
    """Configuration for output options."""
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    results_dir_name: Optional[str] = None
    results_sub_dir: Optional[str] = None


@dataclass
class ElderConfig:
    """Main configuration class for Elder."""
    db: DatabaseConfig
    runner: RunnerConfig
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    misc: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set derived configuration values after initialization."""
        # Set default paths if not provided
        if not self.runner.model_path:
            embedding_model = self.runner.custom_model_name or self.runner.model_type.value
            self.runner.model_path = f"emb_data/models/{embedding_model}"

        # Format result directory names based on configuration
        if not self.output.results_dir_name:
            model_name = self.runner.custom_model_name or self.runner.model_type.value
            self.output.results_dir_name = (
                f"{model_name}_{self.runner.runner_type.value}_{self.runner.nr_of_phenopackets}pp"
                f"_top{self.runner.nr_of_results}"
            )
        
        if not self.output.results_sub_dir:
            model_name = self.runner.custom_model_name or self.runner.model_type.value
            self.output.results_sub_dir = (
                f"{model_name}_{self.runner.runner_type.value}_{self.db.collection_name}"
                f"_{self.runner.nr_of_phenopackets}pp_top{self.runner.nr_of_results}"
            )


class ConfigLoader:
    """Loads and manages configuration from files and environment variables."""

    @staticmethod
    def load_from_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    @classmethod
    def create_database_config(cls, config_data: Dict[str, Any]) -> DatabaseConfig:
        """Create a DatabaseConfig object from config data."""
        db_config = config_data.get('database', {})
        return DatabaseConfig(
            chroma_db_path=db_config.get('chroma_db_path', "emb_data/models/large3"),
            collection_name=db_config.get('collection_name', "lrd_hpo_embeddings"),
            similarity_measure=SimilarityMeasures[db_config.get('similarity_measure', "COSINE")]
        )

    @classmethod
    def create_runner_config(cls, config_data: Dict[str, Any]) -> RunnerConfig:
        """Create a RunnerConfig object from config data."""
        runner_config = config_data.get('runner', {})
        return RunnerConfig(
            runner_type=RunnerType(runner_config.get('runner_type', "avg")),
            model_type=ModelType(runner_config.get('model_type', "large")),
            nr_of_phenopackets=str(runner_config.get('nr_of_phenopackets', "385")),
            nr_of_results=int(runner_config.get('nr_of_results', 10)),
            custom_model_name=runner_config.get('custom_model_name'),
            model_path=runner_config.get('model_path')
        )

    @classmethod
    def create_processing_config(cls, config_data: Dict[str, Any]) -> ProcessingConfig:
        """Create a ProcessingConfig object from config data."""
        processing_config = config_data.get('processing', {})
        return ProcessingConfig(
            use_multiprocessing=processing_config.get('use_multiprocessing', True),
            num_workers=processing_config.get('num_workers'),
            batch_size=processing_config.get('batch_size', 100)
        )

    @classmethod
    def create_output_config(cls, config_data: Dict[str, Any]) -> OutputConfig:
        """Create an OutputConfig object from config data."""
        output_config = config_data.get('output', {})
        return OutputConfig(
            output_dir=Path(output_config.get('output_dir', "./output")),
            results_dir_name=output_config.get('results_dir_name'),
            results_sub_dir=output_config.get('results_sub_dir')
        )

    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> ElderConfig:
        """
        Load configuration from a file and create an ElderConfig object.
        
        If config_path is None, it will look for a config file in the following locations:
        1. Path specified in ELDER_CONFIG environment variable
        2. ./elder_config.yaml
        3. ../elder_config.yaml
        """
        # Try to find config file if not specified
        if config_path is None:
            if 'ELDER_CONFIG' in os.environ:
                config_path = os.environ['ELDER_CONFIG']
            elif os.path.exists('./elder_config.yaml'):
                config_path = './elder_config.yaml'
            elif os.path.exists('../elder_config.yaml'):
                config_path = '../elder_config.yaml'
            else:
                # Use default values if no config file is found
                return ElderConfig(
                    db=DatabaseConfig(chroma_db_path="emb_data/models/large3"),
                    runner=RunnerConfig(
                        runner_type=RunnerType.AVERAGE,
                        model_type=ModelType.LARGE,
                        nr_of_phenopackets="385"
                    )
                )
        
        # Load configuration from file
        config_data = cls.load_from_yaml(config_path)
        
        # Create config objects
        db_config = cls.create_database_config(config_data)
        runner_config = cls.create_runner_config(config_data)
        processing_config = cls.create_processing_config(config_data)
        output_config = cls.create_output_config(config_data)
        
        # Create and return ElderConfig
        misc = {k: v for k, v in config_data.items() 
                if k not in ['database', 'runner', 'processing', 'output']}
        
        return ElderConfig(
            db=db_config,
            runner=runner_config,
            processing=processing_config,
            output=output_config,
            misc=misc
        )


# Global configuration instance
_config: Optional[ElderConfig] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> ElderConfig:
    """
    Get the global configuration instance.
    
    If the configuration hasn't been loaded yet, it will be loaded from the specified
    config_path or from the default locations.
    """
    global _config
    if _config is None:
        _config = ConfigLoader.load(config_path)
    return _config


def set_config(config: ElderConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config