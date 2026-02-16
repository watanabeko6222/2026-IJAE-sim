import os
import warnings
import argparse
from pathlib import Path
import numpy as np

# Import simulation components
from src.kalman_filter import KalmanFilter
from src.moving_generator import MovingSimulatorLine

# Import section-specific functions
from src.self_localization import self_localization
from src.relative_localization import relative_localization
from src.integrated_localization import integrated_localization_ablation

from straight_passing import (
    setup_common_parameters,
    setup_ego_vehicle_movement,
    setup_cyclist_movement,
    setup_ego_sensors_and_noise,
    setup_radar_parameters
)

warnings.filterwarnings("ignore")


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
    
    # Run Section 3: Self-localization
    ego_filter = KalmanFilter()
    ego_filter = self_localization(
        ego_movement, infos, ego_filter, time_step, ego_sigma_ww, ego_sigma_aa
    )
    
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
    
    # Run Section 5: Integrated localization
    abs_cyclist_filter = KalmanFilter()
    abs_cyclist_filter = integrated_localization_ablation(
        ego_movement,
        ego_filter,
        cyclist_movement,
        rel_cyclist_filter,
        abs_cyclist_filter,
        time_step,
        tp_sigma_ww,
        tp_sigma_aa,
    )

    # Calculate Mahalanobis ellipse coverage
    xy_mahas = []
    theta_mahas = []
    vel_mahas = []

    # for idx in range(last_obs_idx, last_obs_idx + 101):  # 10 seconds after last observation
    for idx in range(last_obs_idx, last_obs_idx + 1):  # from first to last observation
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

    return xy_coverage, theta_coverage, vel_coverage


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run all sections (3, 4, 5) simulation")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--iter", type=int, default=1000, help="Number of iterations for the simulation")
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(seed=args.seed)
    
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
    
    xy_coverage_list = []
    theta_coverage_list = []
    vel_coverage_list = []
    for i in range(args.iter):
        print(f"\n=== Iteration {i + 1}/{args.iter} ===")
        
        # Run all sections through section 5
        xy_coverage, theta_coverage, vel_coverage = run_all_sections_integrated(
            ego_movement, cyclist_movement, infos, ego_sigma_ww, ego_sigma_aa,
            radar_params, params["time_step"], base_fig_dir
        )
        
        xy_coverage_list.append(xy_coverage)
        theta_coverage_list.append(theta_coverage)
        vel_coverage_list.append(vel_coverage)
    
    # Calculate and print statistics
    print("\n=== Coverage Statistics ===")
    xy_mean = np.mean(xy_coverage_list)
    xy_std = np.std(xy_coverage_list)
    theta_mean = np.mean(theta_coverage_list)
    theta_std = np.std(theta_coverage_list)
    vel_mean = np.mean(vel_coverage_list)
    vel_std = np.std(vel_coverage_list) 
    
    print(f"XY Coverage: {xy_mean:.5f} ± {xy_std:.5f}")
    print(f"Theta Coverage: {theta_mean:.5f} ± {theta_std:.5f}")
    print(f"Velocity Coverage: {vel_mean:.5f} ± {vel_std:.5f}")
    
    # save results to a text file
    results_file = base_fig_dir / "coverage_results.txt"
    with open(results_file, "w") as f:
        f.write(f"XY Coverage: {xy_mean:.5f} ± {xy_std:.5f}\n")
        f.write(f"Theta Coverage: {theta_mean:.5f} ± {theta_std:.5f}\n")
        f.write(f"Velocity Coverage: {vel_mean:.5f} ± {vel_std:.5f}\n")


if __name__ == "__main__":
    main()
