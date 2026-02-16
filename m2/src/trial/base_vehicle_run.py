import json
import warnings
from dataclasses import dataclass
from pathlib import Path

from src.filter.ego_vehicle_localization import SelfLocalizationFilter
from src.filter.rsu_tp_localization import RSUFilter
from src.plot_utils import (
    plot_states,
)
from src.simulator.map_constructor import BaseMap
from src.simulator.moving_generator import MovingSimulator


@dataclass
class SimulationConfig:
    seed: int
    map: BaseMap
    ego_infos: list
    # ego_infos = [
    #     GNSS(pos_std=4.25, fps=1),
    #     WheelVel(vel_std=0.28, fps=10),
    #     LinkInfoSensor(map=map, fps=1)
    # ]
    node_id_list: list[int]

    time_step: float = 0.1
    # AU7684N2000E10
    # 0.5 deg/s = 0.00873 rad/s
    ego_sigma_ww: float = 0.01
    ego_sigma_aa: float = 0.10
    vehicle_vel: float = 30 / 3.6  # 25 km/h in m/s


warnings.filterwarnings("ignore")


def run_trial(
    cfg: SimulationConfig,
    base_fig_dir: Path,
) -> None:
    """Run self-, relative-, and integrated-localization blocks for all observers.

    Returns:
        observer_filters: EKF outputs for each observer vehicle
        relative_results: List of dicts with relative filter and metadata
        abs_filter: Integrated absolute filter for the cyclist
        first_global_obs: Earliest observation index across observers
        last_global_obs: Latest observation index across observers
        track_end_idx: Simulation index after the final fused prediction
    """
    movement = MovingSimulator(
        map=cfg.map,
        dt=0.1,
        node_id_list=cfg.node_id_list,
        tp_type="automobile",
        desired_vel=cfg.vehicle_vel
    )

    # save simulation config as json
    map_params = {
        "config_dir_path": getattr(cfg.map, "config_dir_path", None),
        "link_length": getattr(cfg.map, "link_length", None),
        "link_width": getattr(cfg.map, "link_width", None),
        "x_num": getattr(cfg.map, "x_num", None),
        "y_num": getattr(cfg.map, "y_num", None),
    }

    _cfg = cfg.__dict__.copy()
    _cfg["map"] = map_params
    _cfg["ego_infos"] = f"{[str(info) for info in cfg.ego_infos]}"
    with open(base_fig_dir / "sim_config.json", "w") as f:
        json.dump(_cfg, f, indent=4)

    # === STEP 1: RSU localization ===
    rsu_localization = RSUFilter(
        map=cfg.map,
        time_step=cfg.time_step,
    )
    rsu_filters = rsu_localization.localization(
        tp_movement=movement,
        sigma_ww=cfg.ego_sigma_aa,
        sigma_aa=cfg.ego_sigma_aa,
    )

    step1_fig_dir = base_fig_dir / "step1-rsu"
    step1_fig_dir.mkdir(exist_ok=True, parents=True)
    state_series = movement.state_timeseries
    plot_states(
        state_series,
        rsu_filters,
        step1_fig_dir,
        step="step1-rsu",
        map=cfg.map,
    )

    # === STEP 2: Vehicles self-localization ===
    self_localization = SelfLocalizationFilter(
        time_step=cfg.time_step,
        sigma_ww=cfg.ego_sigma_ww,
        sigma_aa=cfg.ego_sigma_aa,
    )
    step2_fig_dir = base_fig_dir / "step2-ego"
    step2_fig_dir.mkdir(exist_ok=True, parents=True)
    vehicle_filter = self_localization.localization(
        ego_movement=movement,
        rsu_filters=rsu_filters,
        self_infos=cfg.ego_infos,
        first_aggrigate_step=50,
    )
    state_series = movement.state_timeseries
    plot_states(
        state_series,
        vehicle_filter,
        step2_fig_dir,
        step="step2-ego",
        map=cfg.map,
    )
