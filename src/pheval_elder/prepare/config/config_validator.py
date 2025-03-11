"""
Configuration validation module.

This module provides functions for validating Elder configuration values.
"""

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple

from pheval_elder.prepare.config.unified_config import (
    ElderConfig, RunnerType, ModelType, DatabaseConfig, RunnerConfig
)


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


def validate_path_exists(path: Union[str, Path], path_name: str, required: bool = True) -> None:
    """
    Validate that a path exists.
    
    Args:
        path: Path to validate
        path_name: Name of the path for error messages
        required: Whether the path is required
        
    Raises:
        ConfigValidationError: If the path doesn't exist and is required
    """
    if not path:
        if required:
            raise ConfigValidationError(f"{path_name} is required")
        return
        
    if not os.path.exists(path):
        raise ConfigValidationError(f"{path_name} '{path}' does not exist")


def validate_phenopackets_path(path: Union[str, Path]) -> None:
    """
    Validate that a phenopackets path exists and contains phenopackets.
    
    Args:
        path: Path to validate
        
    Raises:
        ConfigValidationError: If the path doesn't exist or doesn't contain phenopackets
    """
    validate_path_exists(path, "Phenopackets path")
    
    # Check if the directory contains phenopackets (.json files)
    path_obj = Path(path)
    if not any(path_obj.glob("*.json")):
        raise ConfigValidationError(
            f"Phenopackets path '{path}' does not contain any JSON files"
        )


def validate_db_path(db_config: DatabaseConfig) -> None:
    """
    Validate database configuration.
    
    Args:
        db_config: Database configuration
        
    Raises:
        ConfigValidationError: If the database configuration is invalid
    """
    # Check that the ChromaDB path exists
    validate_path_exists(db_config.chroma_db_path, "ChromaDB path")
    
    # Check that the collection name is provided
    if not db_config.collection_name:
        raise ConfigValidationError("Collection name is required")


def validate_runner_config(runner_config: RunnerConfig) -> None:
    """
    Validate runner configuration.
    
    Args:
        runner_config: Runner configuration
        
    Raises:
        ConfigValidationError: If the runner configuration is invalid
    """
    # Check that the runner type is valid
    if runner_config.runner_type not in RunnerType:
        raise ConfigValidationError(
            f"Invalid runner type: {runner_config.runner_type}. "
            f"Must be one of: {', '.join(r.value for r in RunnerType)}"
        )
    
    # Check that the model type is valid
    if runner_config.model_type not in ModelType:
        raise ConfigValidationError(
            f"Invalid model type: {runner_config.model_type}. "
            f"Must be one of: {', '.join(m.value for m in ModelType)}"
        )
    
    # Check that the number of phenopackets is provided
    if not runner_config.nr_of_phenopackets:
        raise ConfigValidationError("Number of phenopackets is required")
    
    # Check that the number of results is a positive integer
    if runner_config.nr_of_results <= 0:
        raise ConfigValidationError("Number of results must be a positive integer")
    
    # If model type is custom, check that custom model name is provided
    if runner_config.model_type == ModelType.CUSTOM and not runner_config.custom_model_name:
        raise ConfigValidationError(
            "Custom model name is required when model type is 'custom'"
        )
    
    # If model path is provided, check that it exists
    if runner_config.model_path:
        validate_path_exists(runner_config.model_path, "Model path")


def validate_config(config: ElderConfig) -> List[str]:
    """
    Validate Elder configuration.
    
    Args:
        config: Elder configuration
        
    Returns:
        List of warnings (non-critical issues)
        
    Raises:
        ConfigValidationError: If the configuration is invalid
    """
    warnings = []
    
    try:
        # Validate database configuration
        validate_db_path(config.db)
    except ConfigValidationError as e:
        warnings.append(f"Database configuration warning: {e}")
    
    try:
        # Validate runner configuration
        validate_runner_config(config.runner)
    except ConfigValidationError as e:
        raise ConfigValidationError(f"Runner configuration error: {e}")
    
    # Validate phenopackets path if it's defined in misc.data_paths
    phenopackets_dir = config.misc.get("data_paths", {}).get("phenopackets_dir")
    if phenopackets_dir:
        try:
            # Only check if the path exists, don't require JSON files
            # since we might be using test data or a symlink
            validate_path_exists(phenopackets_dir, "Phenopackets directory")
        except ConfigValidationError as e:
            warnings.append(f"Phenopackets configuration warning: {e}")
    
    return warnings


def validate_config_file(config_path: Union[str, Path]) -> Tuple[ElderConfig, List[str]]:
    """
    Validate a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Tuple of (configuration, warnings)
        
    Raises:
        ConfigValidationError: If the configuration file is invalid
    """
    from pheval_elder.prepare.config.unified_config import ConfigLoader
    
    # Check that the configuration file exists
    validate_path_exists(config_path, "Configuration file")
    
    # Load the configuration
    config = ConfigLoader.load(config_path)
    
    # Validate the configuration
    warnings = validate_config(config)
    
    return config, warnings