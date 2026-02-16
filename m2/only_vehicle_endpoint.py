import argparse
import importlib
from pathlib import Path

import numpy as np
from loguru import logger

from src.plot_utils import setup_matplotlib
from src.trial.base_vehicle_run import (
    SimulationConfig,
    run_trial,
)


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run straight simulation")
    parser.add_argument("--config", "-c", type=str, help="Config python path")
    args = parser.parse_args()

    # Dynamically import the config module
    config_module = importlib.import_module(f"configs.{Path(args.config).stem}")
    cfg: SimulationConfig = config_module.cfg

    # Set random seed
    np.random.seed(seed=cfg.seed)

    # Setup matplotlib
    setup_matplotlib()

    file_name = Path(args.config).stem

    # Create base figure directory
    base_fig_dir = Path("results") / file_name / str(cfg.seed)
    base_fig_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Random seed: {cfg.seed}")
    logger.info(f"Results will be saved to: {base_fig_dir}")

    run_trial(
        cfg=cfg,
        base_fig_dir=base_fig_dir,
    )


if __name__ == "__main__":
    main()
