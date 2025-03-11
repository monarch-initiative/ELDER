"""
Base runner class for all Elder runners.

This module provides a base class that all Elder runners can inherit from,
using the unified configuration system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from pheval.runners.runner import PhEvalRunner

from pheval_elder.prepare.config.unified_config import (
    ElderConfig, get_config, RunnerType, ModelType
)
from pheval_elder.prepare.config.config_validator import (
    validate_config, ConfigValidationError
)
from pheval_elder.prepare.core.run.elder import ElderRunner
from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures
from pheval_elder.prepare.core.utils.logging import get_logger


@dataclass
class BaseElderRunner(PhEvalRunner):
    """
    Base class for all Elder runners, providing common functionality
    and configuration handling.
    """
    input_dir: Path
    testdata_dir: Path
    tmp_dir: Path
    output_dir: Path
    config_file: Path
    version: str
    
    # Configuration properties
    config: ElderConfig = field(default=None)
    results: List[Any] = field(default_factory=list)
    current_file_name: Optional[str] = None
    elder_runner: Optional[ElderRunner] = None
    
    # Logger
    logger = get_logger("base_runner")

    def __init__(
        self,
        input_dir: Path,
        testdata_dir: Path,
        tmp_dir: Path,
        output_dir: Path,
        config_file: Path,
        version: str,
        config_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initialize the base runner with common parameters and configuration.
        
        Args:
            input_dir: Directory containing input data
            testdata_dir: Directory containing test data
            tmp_dir: Directory for temporary files
            output_dir: Directory for output files
            config_file: PhEval configuration file
            version: Version string
            config_path: Path to the Elder configuration file (optional)
            **kwargs: Additional keyword arguments
        """
        # Initialize PhEvalRunner
        super().__init__(
            input_dir=input_dir,
            testdata_dir=testdata_dir,
            tmp_dir=tmp_dir,
            output_dir=output_dir,
            config_file=config_file,
            version=version,
            **kwargs,
        )
        
        # Load configuration
        self.logger.info(f"Loading configuration from {config_path or 'default locations'}")
        self.config = get_config(config_path)
        
        # Validate configuration
        try:
            warnings = validate_config(self.config)
            for warning in warnings:
                self.logger.warning(warning)
        except ConfigValidationError as e:
            self.logger.error(f"Configuration validation error: {e}")
            raise
        
        # Initialize ElderRunner
        self._init_elder_runner()

    def _init_elder_runner(self) -> None:
        """Initialize the ElderRunner with current configuration."""
        self.logger.info(f"Initializing ElderRunner with strategy: {self.config.runner.runner_type.value}")
        self.elder_runner = ElderRunner(
            similarity_measure=self.config.db.similarity_measure,
            collection_name=self.config.db.collection_name,
            strategy=self.config.runner.runner_type.value,
            embedding_model=(
                self.config.runner.custom_model_name or 
                self.config.runner.model_type.value
            ),
            nr_of_phenopackets=self.config.runner.nr_of_phenopackets,
            db_collection_path=self.config.db.chroma_db_path,
            nr_of_results=self.config.runner.nr_of_results
        )

    def prepare(self) -> None:
        """
        Prepare the runner before execution.
        
        This typically initializes data and sets up collections.
        """
        self.logger.info("Preparing runner")
        # Initialize data and set up collections
        self.elder_runner.initialize_data()
        self.elder_runner.setup_collections()

    def run(self) -> None:
        """
        Run the runner.
        
        This should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the run method")

    def post_process(self) -> None:
        """
        Process the results after running.
        
        This should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the post_process method")

    def get_phenopackets_dir(self, config: ElderConfig) -> Path:
        """
        Get the phenopackets directory from the configuration.
        
        Args:
            config: ElderConfig instance
            
        Returns:
            Path to phenopackets directory
        """
        repo_root = Path(__file__).parent.parents[1]
        phenopackets_dir = config.misc.get("data_paths", {}).get("phenopackets_dir", "5084_phenopackets")
        
        # Convert to absolute path if not already
        if not os.path.isabs(phenopackets_dir):
            phenopackets_dir = repo_root / phenopackets_dir
            
        return Path(phenopackets_dir)

    @classmethod
    def from_config(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        version: str = "0.3.2",
        **kwargs
    ):
        """
        Create a runner instance from configuration.
        
        Args:
            config_path: Path to the configuration file (optional)
            version: Version string (optional)
            **kwargs: Additional keyword arguments to override configuration values
            
        Returns:
            An instance of the runner
        """
        # Load configuration
        logger = get_logger("base_runner")
        logger.info(f"Creating runner from configuration: {config_path or 'default locations'}")
        config = get_config(config_path)
        
        # Apply kwargs overrides to configuration
        # This allows for command-line arguments to override configuration
        # For example: runner_type="avg" would override config.runner.runner_type
        for key, value in kwargs.items():
            if key == "runner_type" and value:
                config.runner.runner_type = RunnerType(value)
                logger.info(f"Overriding runner_type with {value}")
            elif key == "model_type" and value:
                config.runner.model_type = ModelType(value)
                logger.info(f"Overriding model_type with {value}")
            elif key == "nr_of_phenopackets" and value:
                config.runner.nr_of_phenopackets = str(value)
                logger.info(f"Overriding nr_of_phenopackets with {value}")
            elif key == "nr_of_results" and value:
                config.runner.nr_of_results = int(value)
                logger.info(f"Overriding nr_of_results with {value}")
            elif key == "collection_name" and value:
                config.db.collection_name = value
                logger.info(f"Overriding collection_name with {value}")
            elif key == "db_collection_path" and value:
                config.db.chroma_db_path = value
                logger.info(f"Overriding db_collection_path with {value}")
        
        # Create paths
        repo_root = Path(__file__).parent.parents[1]
        output_dir = repo_root / "output"
        
        # Create runner instance
        return cls(
            input_dir=repo_root,
            testdata_dir=Path(".."),
            tmp_dir=Path(".."),
            output_dir=output_dir,
            config_file=Path(".."),
            version=version,
            config_path=config_path,
            **kwargs
        )