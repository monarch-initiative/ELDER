"""
Example script to demonstrate how to use the unified configuration system.

This script shows how to use the configuration system to run Elder analysis
with different strategies.
"""

import multiprocessing as mp
from pathlib import Path

import click

from pheval_elder.dis_avg_emb_runner import DiseaseAvgEmbRunner
from pheval_elder.dis_wgt_avg_emb_runner import DisWgtAvgEmbRunner
from pheval_elder.cosim_bma_runner import BestMatchRunner
from pheval_elder.prepare.config.unified_config import RunnerType


@click.command()
@click.option(
    "--config", 
    "-c", 
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to Elder configuration file"
)
@click.option(
    "--strategy", 
    "-s", 
    type=click.Choice(["avg", "wgt_avg", "tpc"]),
    default="avg",
    help="Analysis strategy to use"
)
def main(config, strategy):
    """
    Run Elder analysis with the specified configuration.
    
    This script demonstrates how to use the unified configuration system to run
    different analysis strategies.
    
    Examples:
        python run_with_config.py --config ../elder_config.yaml --strategy avg
        python run_with_config.py --config ../elder_config.yaml --strategy wgt_avg
        python run_with_config.py --config ../elder_config.yaml --strategy tpc
    """
    mp.set_start_method('fork', force=True)
    
    if strategy == "avg":
        click.echo("Running average strategy...")
        runner = DiseaseAvgEmbRunner.from_config(
            config_path=config,
            config_overrides={
                "runner_type": RunnerType.AVERAGE.value,
            }
        )
    elif strategy == "wgt_avg":
        click.echo("Running weighted average strategy...")
        runner = DisWgtAvgEmbRunner.from_config(
            config_path=config,
            config_overrides={
                "runner_type": RunnerType.WEIGHTED_AVERAGE.value,
            }
        )
    elif strategy == "tpc":
        click.echo("Running best match (term-set pairwise comparison) strategy...")
        runner = BestMatchRunner.from_config(
            config_path=config,
            config_overrides={
                "runner_type": RunnerType.BEST_MATCH.value,
            }
        )
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
    
    runner.prepare()
    runner.run()


if __name__ == "__main__":
    main()