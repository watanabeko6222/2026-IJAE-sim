import os
import warnings
import argparse
from pathlib import Path
import numpy as np
import dill

# Import simulation components
from src.kalman_filter import KalmanFilter
from src.moving_generator import MovingSimulatorLine, Point, StraightLine, TP_State
from src.sensor import GNSS, WheelVel

# Import section-specific functions
from src.self_localization import self_localization
from src.relative_localization import relative_localization
from src.integrated_localization import integrated_localization

# Import plotting utilities
from src.plot_utils import (
    setup_matplotlib,
    plot_states,
    plot_xy2d_sum,
    save_results,
    calculate_confidence_interval_statistics,
    plot_x_uncertainty
)

warnings.filterwarnings("ignore")


def setup_common_parameters():
    """Setup common simulation parameters."""
    return {
        "time_step": 0.1,
        "vehicle_vel": 30 / 3.6,  # m/s
        "cyclist_vel": 15 / 3.6,  # m/s
        "obs_start_time": 50,     # seconds
    }


def setup_ego_vehicle_movement(params):
    """Setup ego vehicle movement simulation."""
    vehicle_start_x = 0
    vehicle_end_x = params["vehicle_vel"] * 100
    
    return MovingSimulatorLine(
        dt=params["time_step"],
        init_state=TP_State(x=vehicle_start_x, y=3.75-2.75, theta=0, vel=params["vehicle_vel"]),
        moving_line=StraightLine(
            start=Point(vehicle_start_x, 3.75-2.75), 
            goal=Point(vehicle_end_x, 3.75-2.75)
        ),
        desired_vel=params["vehicle_vel"],
        desired_accel=1,
        desired_decel=1,
    )


def setup_cyclist_movement(params):
    """Setup cyclist movement simulation."""
    vehicle_start_x = 0
    obs_start_x = params["vehicle_vel"] * params["obs_start_time"] - vehicle_start_x
    cyclist_start_x = obs_start_x + params["cyclist_vel"] * params["obs_start_time"] + 50
    cyclist_end_x = cyclist_start_x - params["cyclist_vel"] * 100
    
    return MovingSimulatorLine(
        dt=params["time_step"],
        init_state=TP_State(x=cyclist_start_x, y=1.3-2.75, theta=0, vel=params["cyclist_vel"]),
        moving_line=StraightLine(
            start=Point(cyclist_start_x, 1.3-2.75), 
            goal=Point(cyclist_end_x, 1.3-2.75)
        ),
        desired_vel=params["cyclist_vel"],
        desired_accel=1,
        desired_decel=1,
    )


def setup_ego_sensors_and_noise():
    """Setup ego vehicle sensors and noise parameters."""
    # CEP 5.0m = σ 4.245m
    gnss = GNSS(pos_std=4.25, fps=1, do_calc_vel=False, do_clac_theta=False)
    # 1km/h = 0.277...m/s
    wheel_vel = WheelVel(vel_std=0.28, fps=10)
    infos = [gnss, wheel_vel]
    
    # AU7684N2000E10

    # 0.5 deg/s = 0.00873 rad/s
    ego_sigma_ww = 0.01
    ego_sigma_aa = 0.10
    
    return infos, ego_sigma_ww, ego_sigma_aa


def setup_radar_parameters():
    """Setup radar sensor parameters."""
    # 0.25m noise and 5% bias from ESR 2.5 datasheet
    range_noise = 0.25
    range_bias = 0.05
    # 1.0 deg = 0.0175 rad from ESR 2.5 datasheet
    azimuth_noise = 0.02
    # 0.12 m/s from ESR 2.5 datasheet
    rel_sigma_vel = 0.12
    # EIYA 0.20 m/s^2
    tp_sigma_aa = 0.20
    # 30 deg/s = 0.5236 rad/s
    tp_sigma_ww = 0.52
    
    return range_noise, range_bias, azimuth_noise, rel_sigma_vel, tp_sigma_ww, tp_sigma_aa


def run_all_sections_integrated(
    ego_movement: MovingSimulatorLine,
    cyclist_movement: MovingSimulatorLine,
    infos: list,
    ego_sigma_ww: float,
    ego_sigma_aa: float,
    radar_params: tuple,
    time_step: float,
    base_fig_dir: Path
) -> tuple[KalmanFilter, KalmanFilter, KalmanFilter]:
    """
    Run all sections (3, 4, 5) in integrated mode through section 5.
    This generates results for all three sections simultaneously.
    """
    print("=== Running Section 3: Self-localization ===")
    
    # Run Section 3: Self-localization
    ego_filter = KalmanFilter()
    ego_filter = self_localization(
        ego_movement, infos, ego_filter, time_step, ego_sigma_ww, ego_sigma_aa
    )
    
    # Plot Section 3 results
    section3_fig_dir = base_fig_dir / "section3"
    section3_fig_dir.mkdir(exist_ok=True, parents=True)
    
    plot_states(ego_movement.state_list, ego_filter, "ego", section3_fig_dir, section="section3", time_offset=-50)
    plot_xy2d_sum(ego_movement.state_list, 0, len(ego_movement)-1, ego_filter, section3_fig_dir, section="section3")
    calculate_confidence_interval_statistics(ego_movement.state_list, ego_filter, "ego")
    
    # Save Section 3 sigma values
    _, P_50s = ego_filter.get_state(500)
    last_sigma_x = np.sqrt(P_50s[0, 0])
    last_sigma_y = np.sqrt(P_50s[1, 1])
    last_sigma_theta = np.sqrt(P_50s[2, 2])
    last_sigma_vel = np.sqrt(P_50s[3, 3])
    
    with open(section3_fig_dir / "last_sigma.csv", "w") as f:
        f.write("last_sigma_x, last_sigma_y, last_sigma_theta, last_sigma_vel\n")
        f.write(f"{last_sigma_x}, {last_sigma_y}, {last_sigma_theta}, {last_sigma_vel}\n")
    
    dill.dump_session(section3_fig_dir / "session.pkl")
    
    print("\n=== Running Section 4: Relative localization ===")
    
    # Run Section 4: Relative localization
    range_noise, range_bias, azimuth_noise, rel_sigma_vel, tp_sigma_ww, tp_sigma_aa = radar_params
    
    rel_cyclist_filter = KalmanFilter()
    (
        rel_cyclist_filter,
        rel_cyclist_state_list,
        first_obs_idx,
        last_obs_idx,
        max_iter
    ) = relative_localization(
        ego_movement,
        cyclist_movement,
        rel_cyclist_filter,
        time_step,
        range_noise,
        range_bias,
        azimuth_noise,
        rel_sigma_vel,
        tp_sigma_ww,
        tp_sigma_aa,
    )
    
    # Plot Section 4 results
    section4_fig_dir = base_fig_dir / "section4"
    section4_fig_dir.mkdir(exist_ok=True, parents=True)
    
    plot_states(
        rel_cyclist_state_list[first_obs_idx:last_obs_idx], 
        rel_cyclist_filter, 
        "rel", 
        section4_fig_dir, 
        section="section4", 
    )
    plot_xy2d_sum(
        rel_cyclist_state_list[first_obs_idx:last_obs_idx], 
        first_obs_idx, 
        last_obs_idx, 
        rel_cyclist_filter, 
        section4_fig_dir, 
        section="section4", 
        state_list_all=rel_cyclist_state_list[max_iter:]
    )
    
    
    # Save Section 4 sigma values
    last_sigma_x = np.sqrt(rel_cyclist_filter.Pp_memory[last_obs_idx - first_obs_idx][0, 0])
    last_sigma_y = np.sqrt(rel_cyclist_filter.Pp_memory[last_obs_idx - first_obs_idx][1, 1])
    last_sigma_theta = np.sqrt(rel_cyclist_filter.Pp_memory[last_obs_idx - first_obs_idx][2, 2])
    last_sigma_vel = np.sqrt(rel_cyclist_filter.Pp_memory[last_obs_idx - first_obs_idx][3, 3])
    
    with open(section4_fig_dir / "last_sigma.csv", "w") as f:
        f.write("last_sigma_x, last_sigma_y, last_sigma_theta, last_sigma_vel\n")
        f.write(f"{last_sigma_x}, {last_sigma_y}, {last_sigma_theta}, {last_sigma_vel}\n")
    
    dill.dump_session(section4_fig_dir / "session.pkl")
    
    print("\n=== Running Section 5: Integrated localization ===")
    
    # Run Section 5: Integrated localization
    abs_cyclist_filter = KalmanFilter()
    abs_cyclist_filter = integrated_localization(
        ego_movement,
        ego_filter,
        cyclist_movement,
        rel_cyclist_filter,
        abs_cyclist_filter,
        time_step,
        tp_sigma_ww,
        tp_sigma_aa,
    )
    
    # Assertions
    assert len(ego_movement) == len(ego_filter)
    assert len(rel_cyclist_state_list[first_obs_idx:max_iter]) == len(rel_cyclist_filter)
    assert len(cyclist_movement.state_list[first_obs_idx:max_iter]) == len(abs_cyclist_filter)
    
    # Plot Section 5 results
    section5_fig_dir = base_fig_dir / "section5"
    section5_fig_dir.mkdir(exist_ok=True, parents=True)
    
    plot_states(
        cyclist_movement.state_list[first_obs_idx:max_iter], 
        abs_cyclist_filter, 
        "abs", 
        section5_fig_dir, 
        section="section5"
    )
    plot_xy2d_sum(
        cyclist_movement.state_list, 
        first_obs_idx, 
        last_obs_idx, 
        abs_cyclist_filter, 
        section5_fig_dir, 
        section="section5"
    )
    
    plot_x_uncertainty(
        first_obs_idx=first_obs_idx,
        last_obs_idx=last_obs_idx,
        section3_filter=ego_filter,
        section4_filter=rel_cyclist_filter,
        section5_filter=abs_cyclist_filter,
        base_fig_dir=section5_fig_dir
    )

    # Save Section 5 results and detailed output
    save_results(abs_cyclist_filter, first_obs_idx, last_obs_idx, section5_fig_dir)
    dill.dump_session(section5_fig_dir / "session.pkl")

    # Calculate Mahalanobis ellipse coverage
    xy_mahas = []
    theta_mahas = []
    vel_mahas = []

    for idx in range(last_obs_idx, last_obs_idx + 101):  # 10 seconds after last observation
        est = abs_cyclist_filter.Xp_memory[idx - first_obs_idx].flatten()
        P = abs_cyclist_filter.Pp_memory[idx - first_obs_idx]
        true = cyclist_movement.state_list[idx]
        # xy: Mahalanobis ellipse
        xy_diff = np.array([est[0] - true.x, est[1] - true.y])
        P_xy = P[:2, :2]
        xy_maha = np.sqrt(xy_diff.T @ np.linalg.inv(P_xy) @ xy_diff)
        xy_mahas.append(xy_maha)

        theta_diff = est[2] - true.theta
        theta_diff = (theta_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        theta_maha = abs(theta_diff) / np.sqrt(P[2, 2])
        theta_mahas.append(theta_maha)

        vel_diff = est[3] - true.vel
        vel_maha = abs(vel_diff) / np.sqrt(P[3, 3])
        vel_mahas.append(vel_maha)

    print("Mahalanobis ellipse coverage statistics:")
    print(f"XY: {np.mean(xy_mahas):.2f} ± {np.std(xy_mahas):.2f}")
    print(f"Theta: {np.mean(theta_mahas):.2f} ± {np.std(theta_mahas):.2f}")
    print(f"Velocity: {np.mean(vel_mahas):.2f} ± {np.std(vel_mahas):.2f}")

    # --- Coverage rate based on chi2 distribution ---
    import scipy.stats
    alpha = 0.95
    chi2_xy = scipy.stats.chi2.ppf(alpha, df=2)
    chi2_1d = scipy.stats.chi2.ppf(alpha, df=1)
    sqrt_chi2_xy = np.sqrt(chi2_xy)
    sqrt_chi2_1d = np.sqrt(chi2_1d)
    xy_coverage = np.mean(np.array(xy_mahas) <= sqrt_chi2_xy)
    theta_coverage = np.mean(np.array(theta_mahas) <= sqrt_chi2_1d)
    vel_coverage = np.mean(np.array(vel_mahas) <= sqrt_chi2_1d)
    print(f"Coverage within {int(alpha*100)}% chi2 region (distance threshold):")
    print(f"  XY: {xy_coverage*100:.1f}% (sqrt(chi2)={sqrt_chi2_xy:.2f})")
    print(f"  Theta: {theta_coverage*100:.1f}% (sqrt(chi2)={sqrt_chi2_1d:.2f})")
    print(f"  Velocity: {vel_coverage*100:.1f}% (sqrt(chi2)={sqrt_chi2_1d:.2f})")

    return ego_filter, rel_cyclist_filter, abs_cyclist_filter


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run all sections (3, 4, 5) simulation")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(seed=args.seed)
    
    # Setup matplotlib
    setup_matplotlib()
    
    file_name = os.path.basename(__file__).replace(".py", "")
    
    # Create base figure directory
    base_fig_dir = Path("figs") / file_name / str(args.seed)
    base_fig_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup common parameters
    params = setup_common_parameters()
    
    # Setup movements
    ego_movement = setup_ego_vehicle_movement(params)
    cyclist_movement = setup_cyclist_movement(params)
    
    # Setup sensors and parameters
    infos, ego_sigma_ww, ego_sigma_aa = setup_ego_sensors_and_noise()
    radar_params = setup_radar_parameters()
    
    print("Running integrated mode: All sections (3, 4, 5) via section 5")
    print(f"Random seed: {args.seed}")
    print(f"Results will be saved to: {base_fig_dir}")
    
    # Run all sections through section 5
    run_all_sections_integrated(
        ego_movement, cyclist_movement, infos, ego_sigma_ww, ego_sigma_aa,
        radar_params, params["time_step"], base_fig_dir
    )
    print("\n=== All requested sections completed ===")
    print(f"Results saved to: {base_fig_dir}")


if __name__ == "__main__":
    main()
