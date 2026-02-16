"""
Plotting utilities for visualization of localization results.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .kalman_filter import KalmanFilter
from .moving_generator import TP_State


def setup_matplotlib():
    """Setup matplotlib configuration for consistent plotting style."""
    plt.style.use("fast")
    plt.rcParams["figure.figsize"] = [6.4, 4.0]
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.subplot.left"] = 0.14
    plt.rcParams["figure.subplot.bottom"] = 0.14
    plt.rcParams["figure.subplot.right"] = 0.90
    plt.rcParams["figure.subplot.top"] = 0.91
    plt.rcParams["figure.subplot.wspace"] = 0.20
    plt.rcParams["figure.subplot.hspace"] = 0.20

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 1
    plt.rcParams["grid.color"] = "black"

    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 1
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.fancybox"] = False



def plot_states(
    state_list: list[TP_State],
    filter: KalmanFilter,
    type: str,
    base_fig_dir: Path,
    section: str = "section5",
    time_offset: float = 0.0,
):
    """Plot estimated states (theta and velocity) with confidence intervals.
    
    Args:
        state_list: List of true states
        filter: Kalman filter with estimation results
        type: Plot type identifier (e.g., "ego", "rel", "abs")
        base_fig_dir: Directory to save plots
        section: Section identifier to determine plot settings ("section3", "section4", "section5")
        time_offset: Time offset to add to time axis (e.g., 50.0 for section4)
    """
    time = np.arange(0, len(state_list), 1) * 0.1 + time_offset

    # Plot theta
    _plot_theta(state_list, filter, type, base_fig_dir, time, section)
    
    # Plot velocity
    _plot_velocity(state_list, filter, type, base_fig_dir, time, section)


def _plot_theta(state_list: list[TP_State], filter: KalmanFilter, type: str, base_fig_dir: Path, time: np.ndarray, section: str):
    """Plot theta estimation results."""
    theta_std_list = [np.sqrt(filter.get_state(i)[1][2, 2]) for i in range(len(state_list))]
    theta_std_list = [np.arctan2(np.sin(theta), np.cos(theta)) for theta in theta_std_list]
    
    theta_list = [filter.get_state(i)[0][2] for i in range(len(state_list))]
    theta_list = [np.arctan2(np.sin(theta), np.cos(theta)) for theta in theta_list]
    
    # Handle angle wrapping for section4 and section5
    if section in ["section4", "section5"]:
        for i in range(len(theta_list) - 1):
            if theta_list[i + 1] - theta_list[i] > np.pi:
                theta_list[i + 1] -= 2 * np.pi
            elif theta_list[i + 1] - theta_list[i] < -np.pi:
                theta_list[i + 1] += 2 * np.pi
    
    plt.figure()
    plt.plot(time, theta_list, label="Estimated")
    
    # 95% confidence interval
    plt.fill_between(
        time,
        [float(theta + 1.96 * std) for theta, std in zip(theta_list, theta_std_list)],
        [float(theta - 1.96 * std) for theta, std in zip(theta_list, theta_std_list)],
        alpha=0.3,
        label="95% confidence interval",
    )
    
    plt.plot(
        time,
        [np.arctan2(np.sin(state.theta + 1e-5), np.cos(state.theta + 1e-5)) for state in state_list],
        label="Ground Truth",
        color="tab:orange",
        linestyle="--",
    )
    
    plt.xlabel("Time [s]")
    plt.ylabel("Yaw angle [rad]")
    
    # Set plot limits based on section
    if section == "section3":
        plt.xlim(-50, 10)
        plt.ylim(-0.75, 0.75)
    else:
        if section == "section4":
            plt.xlim(0, 4)
            plt.xticks(np.arange(0, 5, 1))
        else:
            plt.xlim(0, 16)
            plt.xticks(np.arange(0, 17, 2))
        plt.ylim(-np.pi - 0.75, -np.pi + 0.75)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

    plt.legend()
    plt.grid()
    plt.savefig(base_fig_dir / f"{type}_theta.png")
    plt.close()


def _plot_velocity(state_list: list[TP_State], filter: KalmanFilter, type: str, base_fig_dir: Path, time: np.ndarray, section: str):
    """Plot velocity estimation results."""
    vel_std_list = [np.sqrt(filter.get_state(i)[1][3, 3]) for i in range(len(state_list))]
    vel_list = [filter.get_state(i)[0][3] for i in range(len(state_list))]
    
    plt.figure()
    plt.plot(time, vel_list, label="Estimated")
    
    # 95% confidence interval
    plt.fill_between(
        time,
        [float(vel - 1.96 * std) for vel, std in zip(vel_list, vel_std_list)],
        [float(vel + 1.96 * std) for vel, std in zip(vel_list, vel_std_list)],
        alpha=0.3,
        label="95% confidence interval",
    )
    
    plt.plot(
        time,
        [state.vel for state in state_list],
        label="Ground Truth",
        color="tab:orange",
        linestyle="--",
    )
    
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    
    # Set plot limits and legend position based on section
    if section == "section3":
        plt.xlim(-50, 10)
        plt.ylim(7, 10)
    else:
        if section == "section4":
            plt.xlim(0, 4)
            plt.ylim(11, 16)
            plt.xticks(np.arange(0, 5, 1))
        else:  # section5
            plt.xlim(0, 16)
            plt.ylim(3, 6)
            plt.xticks(np.arange(0, 17, 2))
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

        plt.legend(loc="upper right")
    
    if section != "section5":
        plt.legend()
    
    plt.grid()
    plt.savefig(base_fig_dir / f"{type}_vel.png")
    plt.close()


def plot_xy2d_sum(
    state_list: list[TP_State],
    first_obs_idx: int,
    last_obs_idx: int,
    cyclist_filter: KalmanFilter,
    base_fig_dir: Path,
    section: str = "section5",
    state_list_all: list[TP_State] = None,
):
    """Plot 2D trajectory with uncertainty ellipses.
    
    Args:
        state_list: List of states for the observed period
        first_obs_idx: First observation index
        last_obs_idx: Last observation index  
        cyclist_filter: Filter with estimation results
        base_fig_dir: Directory to save plots
        section: Section identifier to determine plot settings
        state_list_all: Complete state list (used for section4)
    """
    chi2_val = 5.991
    X_memory = np.array([cyclist_filter.get_state(i)[0] for i in range(len(cyclist_filter))])
    P_memory = np.array([cyclist_filter.get_state(i)[1] for i in range(len(cyclist_filter))])

    ax = plt.gca()
    # Add road background
    if section == "section4":
        ax.add_patch(
            patches.Rectangle(
                (-200, -3.75),
                1000,
                5.5,
                facecolor="gray",
                alpha=0.3,
            )
        )
    else:
        ax.add_patch(
            patches.Rectangle(
                (-200, -2.75),
                1000,
                5.5,
                facecolor="gray",
                alpha=0.3,
            )
        )

    plt.gca().set_aspect("equal")

    # Plot ground truth trajectory - different sampling based on section
    if section == "section3":
        # Section3: plot ego vehicle trajectory
        x = [state.x for state in state_list[::20]]
        y = [state.y for state in state_list[::20]]
        u = np.cos([state.theta for state in state_list[::20]])
        v = np.sin([state.theta for state in state_list[::20]])
    elif section == "section4":
        # Section4: plot both observed and unobserved periods
        x = [state.x for state in state_list[::10]]
        y = [state.y for state in state_list[::10]]
        u = np.cos([state.theta for state in state_list[::10]])
        v = np.sin([state.theta for state in state_list[::10]])
        
        # if state_list_all is not None:
        #     x += [state.x for state in state_list_all[::10]]
        #     y += [state.y for state in state_list_all[::10]]
        #     u = np.append(u, np.cos([state.theta for state in state_list_all[::10]]))
        #     v = np.append(v, np.sin([state.theta for state in state_list_all[::10]]))
    else:  # section5
        # Section5: plot cyclist trajectory
        x = [state.x for state in state_list[::20]]
        y = [state.y for state in state_list[::20]]
        u = np.cos([state.theta for state in state_list[::20]])
        v = np.sin([state.theta for state in state_list[::20]])
    
    # plt.quiver(x, y, u, v, scale=5, scale_units='inches', color='tab:orange', label="Ground Truth")
    plt.scatter(x, y, color='tab:orange', label="Ground Truth", s=20, marker="*")

    # Plot uncertainty ellipses - different sampling based on section
    sample_rate = 10 if section == "section4" else 20
    if section == "section4":
        X_memory = X_memory[:len(state_list)]
        P_memory = P_memory[:len(state_list)]
    is_legend_first = True
    for X, Sigma in zip(X_memory[::sample_rate], P_memory[::sample_rate]):
        P = Sigma[:2, :2]
        w, v = np.linalg.eig(P)
        angle = np.arctan2(v[1, 0], v[0, 0])
        label = "95% confidence interval" if is_legend_first else None
        is_legend_first = False
        
        ellipse = plt.matplotlib.patches.Ellipse(
            xy=(X[0], X[1]),
            width=2 * np.sqrt(chi2_val * w[0]),
            height=2 * np.sqrt(chi2_val * w[1]),
            angle=np.rad2deg(angle),
            edgecolor="tab:blue",
            facecolor="none",
            label=label,
        )
        plt.gca().add_patch(ellipse)
    
    # Plot estimated trajectory
    x = X_memory[:, 0][::sample_rate]
    y = X_memory[:, 1][::sample_rate]
    u = np.cos(X_memory[:, 2][::sample_rate])
    v = np.sin(X_memory[:, 2][::sample_rate])
    # plt.quiver(x, y, u, v, scale=5, scale_units='inches', color='tab:blue', label="Estimated")
    plt.scatter(
        x, y, color='tab:blue', label="Estimated", s=20, marker="^"
    )

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    
    # Set plot limits based on section
    if section == "section3":
        plt.xlim(400, 500)
        plt.ylim(-20, 40)
    elif section == "section4":
        plt.xlim(-40, 60)
        plt.ylim(-20, 40)
    else:  # section5
        plt.xlim(400, 500)
        plt.ylim(-20, 40)
    
    
    plt.legend()
    plt.grid()
        
    if section == "section4":
        fig = plt.gcf()
        ax_inset = fig.add_axes([0.1, 0.2, 0.4, 0.25])
        ax_inset.set_xlim(10, 25)
        ax_inset.set_ylim(-5, 5) 
        
        # 透過背景の矩形を追加
        ax.add_patch(
            patches.Rectangle(
                (10, -5),
                15, 10,
                alpha=1.0,
                facecolor="none",
                edgecolor="black"
            )
        )

        ax_inset.scatter(x, y, color='tab:orange', s=20, marker="*")
        ax_inset.scatter(x, y, color='tab:blue', s=20, marker="^")

        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

        for X, Sigma in zip(X_memory[::sample_rate], P_memory[::sample_rate]):
            P = Sigma[:2, :2]
            w, v = np.linalg.eig(P)
            angle = np.arctan2(v[1, 0], v[0, 0])
            label = "95% confidence interval" if is_legend_first else None
            is_legend_first = False
            
            ellipse = plt.matplotlib.patches.Ellipse(
                xy=(X[0], X[1]),
                width=2 * np.sqrt(chi2_val * w[0]),
                height=2 * np.sqrt(chi2_val * w[1]),
                angle=np.rad2deg(angle),
                edgecolor="tab:blue",
                facecolor="none",
                label=label,
            )
            plt.gca().add_patch(ellipse)
        
        ax_inset.add_patch(
        patches.Rectangle(
            (-200, -3.75),
            1000,
            5.5,
            facecolor="gray",
            alpha=0.3,
            )
        )
        ax_inset.set_aspect("equal")

    plt.savefig(base_fig_dir / "sum_xy_2d_2.png")
    plt.close()


def plot_x_uncertainty(
    first_obs_idx: int,
    last_obs_idx: int,
    section3_filter: KalmanFilter,
    section4_filter: KalmanFilter,
    section5_filter: KalmanFilter,
    base_fig_dir: Path
) -> None:
    """Plot X uncertainty over time."""

    # Plot uncertainty for each section
    for filter, label in zip(
        [section3_filter, section4_filter, section5_filter],
        ["EKF-1", "EKF-2", "Integrated"]
    ):
        # if label == "EKF-1":
        #     x_uncertainty = [
        #         np.sqrt(filter.Pp_memory[i][0, 0]) for i in range(first_obs_idx, last_obs_idx + 200)
        #     ]
        #     time = np.arange(0, len(x_uncertainty)) * 0.1
        #     plt.plot(time, x_uncertainty, label=label)
        if label == "Integrated":
            x_uncertainty = [
                np.sqrt(filter.Pp_memory[i][0, 0]) for i in range(len(filter.Pp_memory))
            ]
            time = np.arange(0, len(x_uncertainty)) * 0.1
            plt.plot(time, x_uncertainty, label=label,color='tab:green')

    plt.xlabel("Time [s]")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    plt.ylabel("Uncertainty [m]")
    plt.xlim(0, 16)
    plt.ylim(0, 3)
    plt.legend()
    plt.grid()
    plt.savefig(base_fig_dir / "x_uncertainty.png")
    plt.close()
    
    scale = 1.96 * 2

    # Plot uncertainty for each section
    for filter, label in zip(
        [section3_filter, section4_filter, section5_filter],
        ["EKF-1", "EKF-2", "Integrated"]
    ):
        if label == "EKF-1":
            x_uncertainty = [
                np.sqrt(filter.Pp_memory[i][0, 0]) * scale for i in range(first_obs_idx, last_obs_idx + 1)
            ]
            time = np.arange(0, len(x_uncertainty)) * 0.1
            plt.plot(time, x_uncertainty, label=label)
        elif label == "EKF-2":
            x_uncertainty = [
                np.sqrt(filter.Pp_memory[i][0, 0]) * scale for i in range(0, last_obs_idx - first_obs_idx + 1)
            ]
            time = np.arange(0, len(x_uncertainty)) * 0.1
            plt.plot(time, x_uncertainty, label=label)
        elif label == "Integrated":
            x_uncertainty = [
                np.sqrt(filter.Pp_memory[i][0, 0]) * scale for i in range(0, last_obs_idx - first_obs_idx + 141)
            ]
            time = np.arange(0, len(x_uncertainty)) * 0.1
            plt.plot(time, x_uncertainty, label=label,color='tab:green')

    plt.vlines(x=(last_obs_idx - first_obs_idx) * 0.1, ymin=0, ymax=10, colors='black')
    plt.xlabel("Time [s]")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    plt.ylabel("95% confidence interval range [m]")
    plt.xlim(0, 16)
    plt.xticks(np.arange(0, 17, 2))
    plt.ylim(0, 10)
    # top center
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95))
    plt.grid()
    plt.savefig(base_fig_dir / "x_uncertainty2.png")
    plt.close()


def save_results(
    abs_cyclist_filter: KalmanFilter,
    first_obs_idx: int,
    last_obs_idx: int,
    base_fig_dir: Path,
):
    """Save final sigma values to CSV file."""
    last_sigma_x = np.sqrt(abs_cyclist_filter.Pp_memory[last_obs_idx - first_obs_idx][0, 0])
    last_sigma_y = np.sqrt(abs_cyclist_filter.Pp_memory[last_obs_idx - first_obs_idx][1, 1])
    last_sigma_theta = np.sqrt(abs_cyclist_filter.Pp_memory[last_obs_idx - first_obs_idx][2, 2])
    last_sigma_vel = np.sqrt(abs_cyclist_filter.Pp_memory[last_obs_idx - first_obs_idx][3, 3])
    
    # Print results
    print(f"last_sigma_x: {last_sigma_x}")
    print(f"last_sigma_y: {last_sigma_y}")
    print(f"last_sigma_theta: {last_sigma_theta}")
    print(f"last_sigma_vel: {last_sigma_vel}")
    
    # Save to CSV
    with open(base_fig_dir / "last_sigma.csv", "w") as f:
        f.write("last_sigma_x, last_sigma_y, last_sigma_theta, last_sigma_vel\n")
        f.write(f"{last_sigma_x}, {last_sigma_y}, {last_sigma_theta}, {last_sigma_vel}\n")
    
    return last_sigma_x, last_sigma_y, last_sigma_theta, last_sigma_vel


def calculate_confidence_interval_statistics(
    state_list: list[TP_State],
    filter: KalmanFilter,
    type: str,
):
    """Calculate percentage of values inside 95% confidence interval (for Section3)."""
    # Theta statistics
    theta_std_list = [np.sqrt(filter.get_state(i)[1][2, 2]) for i in range(len(state_list))]
    theta_list = [filter.get_state(i)[0][2] for i in range(len(state_list))]
    theta_error = np.array([theta - state.theta for theta, state in zip(theta_list, state_list)])
    theta_std = np.array(theta_std_list)
    
    theta_inside_count = sum(1 for error, std in zip(theta_error, theta_std) if abs(error) < 1.96 * std)
    theta_percentage = theta_inside_count / len(theta_error)
    
    # Velocity statistics
    vel_std_list = [np.sqrt(filter.get_state(i)[1][3, 3]) for i in range(len(state_list))]
    vel_list = [filter.get_state(i)[0][3] for i in range(len(state_list))]
    vel_error = np.array([vel - state.vel for vel, state in zip(vel_list, state_list)])
    vel_std = np.array(vel_std_list)
    
    vel_inside_count = sum(1 for error, std in zip(vel_error, vel_std) if abs(error) < 1.96 * std)
    vel_percentage = vel_inside_count / len(vel_error)
    
    print(f"{type}_theta_percentage: {theta_percentage}")
    print(f"{type}_vel_percentage: {vel_percentage}")
    
    return theta_percentage, vel_percentage
