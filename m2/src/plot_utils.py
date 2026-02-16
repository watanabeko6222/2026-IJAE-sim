from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from .filter.integrated_localization import Hypothesis
from .filter.kalman_filter import KalmanFilter
from .simulator.map_constructor import BaseMap
from .simulator.moving_generator import TimeseriesTPState

if TYPE_CHECKING:
    from src.controller.collision_avoidance import ControlCommand
from .utils import normalize_angle

cmap = plt.get_cmap("viridis")
N = 8
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


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
    plt.rcParams["axes.axisbelow"] = True

    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 1
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.fancybox"] = False


def plot_states(
    state_series: TimeseriesTPState,
    input: KalmanFilter | list[Hypothesis] | list[KalmanFilter],
    base_fig_dir: Path,
    step: Literal["step1-rsu", "step2-ego", "step3-rel", "step4-abs", "step5-itg"],
    map: BaseMap,
):
    """Plot estimated states (theta and velocity) with confidence intervals.

    Args:
        state_series: Ground truth state time series
        input: Kalman input with estimation results
        base_fig_dir: Directory to save plots
        step: step identifier to determine plot settings
        map: Base map used to render positional figures
    """
    if isinstance(input, KalmanFilter):
        filter_list = [input]
        hypothesis_list = None
    elif isinstance(input, list) and all(isinstance(item, KalmanFilter) for item in input):
        filter_list = input
        hypothesis_list = None
    else:
        filter_list = [item.filter for item in input]
        hypothesis_list = input
    # Plot theta
    _plot_theta(state_series, filter_list, base_fig_dir, step)

    # Plot velocity
    _plot_velocity(state_series, filter_list, base_fig_dir, step)

    # plot 2d position
    if hypothesis_list is not None:
        _plot_position(state_series, hypothesis_list, base_fig_dir, step, map=map)
    else:
        _plot_position(state_series, filter_list, base_fig_dir, step, map=map)

    # plot 2d rmse
    _plot_2d_rmse(state_series, filter_list, base_fig_dir, step)

    # plot 2d mahalanobis
    _plot_2d_mahalanobis(state_series, filter_list, base_fig_dir, step)

    # plot position uncertainty
    if hypothesis_list is not None:
        _plot_pos_uncertainty(hypothesis_list, base_fig_dir)
    else:
        _plot_pos_uncertainty(filter_list, base_fig_dir)

    # plot hypothesis branching
    if hypothesis_list is not None:
        _plot_hypothesis_branching(hypothesis_list, base_fig_dir)
        _plot_node_existence_scores(state_series, hypothesis_list, base_fig_dir, map)


def _plot_theta(
    state_series: TimeseriesTPState,
    filter_list: list[KalmanFilter],
    base_fig_dir: Path,
    step: str,
):
    """Plot theta estimation results."""
    for idx, filter in enumerate(filter_list):
        color = COLORS[idx % (N - 2)]
        step_offset = filter.step_offset
        time = np.arange(step_offset, len(filter)) * 0.1 - 50
        theta_std_list = [np.sqrt(filter.get_pred(i).P[2, 2]) for i in range(step_offset, len(filter))]
        theta_std_list = [np.arctan2(np.sin(theta), np.cos(theta)) for theta in theta_std_list]

        theta_list = [filter.get_pred(i).X[2] for i in range(step_offset, len(filter))]
        theta_list = [np.arctan2(np.sin(theta), np.cos(theta)) for theta in theta_list]

        # Handle angle wrapping
        for i in range(len(theta_list) - 1):
            theta_list[i + 1] = normalize_angle(theta_list[i + 1] - theta_list[i]) + theta_list[i]

        plt.plot(time, theta_list, label="Estimated", color=color)

        # 95% confidence interval
        plt.fill_between(
            time,
            [float(theta + 1.96 * std) for theta, std in zip(theta_list, theta_std_list)],
            [float(theta - 1.96 * std) for theta, std in zip(theta_list, theta_std_list)],
            alpha=0.3,
            label="95% confidence interval",
            color=color,
        )

    try:
        min_step_offset = min(filter.step_offset for filter in filter_list)
    except ValueError:
        min_step_offset = 0
    min_step_offset = min(min_step_offset, state_series.step_offset)
    time = np.arange(min_step_offset, len(state_series)) * 0.1 - 50
    gt_theta_list = [
        np.arctan2(
            np.sin(state_series.get_state(i).theta),
            np.cos(state_series.get_state(i).theta),
        )
        for i in range(min_step_offset, len(state_series))
    ]
    for i in range(len(gt_theta_list) - 1):
        gt_theta_list[i + 1] = normalize_angle(gt_theta_list[i + 1] - gt_theta_list[i]) + gt_theta_list[i]
    plt.plot(
        time,
        gt_theta_list,
        label="Ground Truth",
        color="tab:orange",
        linestyle="--",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Yaw angle [rad]")
    plt.xlim(-50, 10)
    plt.grid()
    plt.ylim(top=0.75)
    plt.savefig(base_fig_dir / "theta.png")
    
    plt.xlim(-10, 10)
    plt.ylim(-0.75, 0.75)
    plt.legend()
    plt.savefig(base_fig_dir / "theta_zoomed.png")
    plt.close()


def _plot_velocity(
    state_series: TimeseriesTPState,
    filter_list: list[KalmanFilter],
    base_fig_dir: Path,
    step: str,
):
    """Plot velocity estimation results."""
    plt.figure()
    for idx, filter in enumerate(filter_list):
        color = COLORS[idx % (N - 2)]
        step_offset = filter.step_offset
        time = np.arange(step_offset, len(filter)) * 0.1 - 50
        vel_std_list = [np.sqrt(filter.get_pred(i).P[3, 3]) for i in range(step_offset, len(filter))]
        vel_list = [filter.get_pred(i).X[3] for i in range(step_offset, len(filter))]

        plt.plot(time, vel_list, label="Estimated", color=color)

        # 95% confidence interval
        plt.fill_between(
            time,
            [float(vel - 1.96 * std) for vel, std in zip(vel_list, vel_std_list)],
            [float(vel + 1.96 * std) for vel, std in zip(vel_list, vel_std_list)],
            alpha=0.3,
            label="95% confidence interval",
            color=color,
        )

    try:
        min_step_offset = min(filter.step_offset for filter in filter_list)
    except ValueError:
        min_step_offset = 0
    min_step_offset = min(min_step_offset, state_series.step_offset)
    time = np.arange(min_step_offset, len(state_series)) * 0.1 - 50
    plt.plot(
        time,
        [state_series.get_state(i).vel for i in range(min_step_offset, len(state_series))],
        label="Ground Truth",
        color="tab:orange",
        linestyle="--",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.ylim(7.0, 10.0)
    plt.xlim(-50, 10)
    plt.grid()
    plt.savefig(base_fig_dir / "vel.png")
    plt.close()


def _plot_2d_rmse(
    state_series: TimeseriesTPState,
    filter_list: list[KalmanFilter],
    base_fig_dir: Path,
    step: str,
):
    """Plot 2D RMSE over time."""
    plt.figure()
    for idx, filter in enumerate(filter_list):
        color = COLORS[idx % (N - 2)]
        step_offset = filter.step_offset
        time = np.arange(step_offset, len(filter)) * 0.1
        rmse_list = []
        for i in range(step_offset, len(filter)):
            est_state = filter.get_pred(i).X
            gt_state = state_series.get_state(i)
            rmse = np.sqrt((est_state[0] - gt_state.x) ** 2 + (est_state[1] - gt_state.y) ** 2)
            rmse_list.append(rmse)

        plt.plot(time, rmse_list, label="2D RMSE", color=color)

    plt.ylim(bottom=0)
    plt.xlabel("Time [s]")
    plt.ylabel("2D RMSE [m]")
    plt.grid()
    plt.savefig(base_fig_dir / "2d_rmse.png")
    plt.close()


def _plot_2d_mahalanobis(
    state_series: TimeseriesTPState,
    filter_list: list[KalmanFilter],
    base_fig_dir: Path,
    step: str,
):
    """Plot 2D Mahalanobis distance over time."""
    plt.figure()
    min_time = float("inf")
    max_time = float("-inf")
    for idx, filter in enumerate(filter_list):
        color = COLORS[idx % (N - 2)]
        step_offset = filter.step_offset
        time = np.arange(step_offset, len(filter)) * 0.1
        maha_list = []
        for i in range(step_offset, len(filter)):
            pred_pos = filter.get_pred(i).X[:2, 0]
            pred_P = filter.get_pred(i).P[:2, :2]
            # print(f"pred_P at step {i}: {pred_P}")
            gt_state = state_series.get_state(i)
            gt_pos = np.array([gt_state.x, gt_state.y])
            mahalanobis_dist = np.sqrt((pred_pos - gt_pos).T @ np.linalg.inv(pred_P) @ (pred_pos - gt_pos))
            maha_list.append(mahalanobis_dist)
        min_time = min(min_time, time[0])
        max_time = max(max_time, time[-1])

        plt.plot(time, maha_list, label="2D Mahalanobis Distance", color=color)

    plt.ylim(bottom=0, top=10)
    plt.hlines(
        np.sqrt(chi2.ppf(0.95, df=2)),
        xmin=min_time,
        xmax=max_time,
        colors="r",
        linestyles="--",
        label="95% threshold",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("2D Mahalanobis Distance")
    plt.grid()
    plt.savefig(base_fig_dir / "2d_mahalanobis.png")
    plt.close()


def _plot_position(
    state_series: TimeseriesTPState,
    input: list[Hypothesis] | list[KalmanFilter],
    base_fig_dir: Path,
    step: str,
    map: BaseMap | None = None,
):
    """Plot 2D trajectory with uncertainty ellipses.

    Args:
        state_series: Ground truth state history for the tracked vehicle
        input: Hypothesis or Kalman filter list containing predicted states
        base_fig_dir: Directory to save plots
        step: step identifier to determine plot settings ("step1", "step2", "step3")
        map: Map object for adding background information when available
    """
    chi2_val = chi2.ppf(0.95, df=2)
    state_step_offset = state_series.step_offset
    sample_rate = 10
    ax = plt.gca()
    # Add road background
    if step != "step3-rel":
        ax = map.viz_background_map(ax)

    plt.gca().set_aspect("equal")

    is_legend_first = True

    # Plot ground truth trajectory - different sampling based on step
    x = [state_series.get_state(i).x for i in range(state_step_offset, len(state_series))][::sample_rate]
    y = [state_series.get_state(i).y for i in range(state_step_offset, len(state_series))][::sample_rate]
    plt.scatter(
        x, y, color="tab:orange", marker="*", s=20, label="Ground Truth"
    )
    # u = np.cos([state_series.get_state(i).theta for i in range(state_step_offset, len(state_series))][::sample_rate])
    # v = np.sin([state_series.get_state(i).theta for i in range(state_step_offset, len(state_series))][::sample_rate])
    # plt.quiver(
    #     x,
    #     y,
    #     u,
    #     v,
    #     scale=10,
    #     scale_units="inches",
    #     color="tab:orange",
    #     label="Ground Truth",
    # )

    for idx, input_item in enumerate(input):
        if isinstance(input_item, KalmanFilter):
            filter = input_item
            color = COLORS[idx % (N - 2)]
        else:
            filter = input_item.filter
            color = COLORS[input_item.generation % (N - 2)]
        step_offset = filter.step_offset
        step_offset += (state_step_offset - step_offset) % sample_rate
        X_memory = np.array([filter.get_pred(i).X for i in range(step_offset, len(filter))])
        P_memory = np.array([filter.get_pred(i).P for i in range(step_offset, len(filter))])
        if len(X_memory) == 0:
            continue

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
                edgecolor=color,
                facecolor="none",
                label=label,
            )
            plt.gca().add_patch(ellipse)

        # Plot estimated trajectory
        x = X_memory[:, 0][::sample_rate]
        y = X_memory[:, 1][::sample_rate]
        plt.scatter(x, y, color=color, marker="^", s=20, label="Estimated")
        #u = np.cos(X_memory[:, 2][::sample_rate])
        #v = np.sin(X_memory[:, 2][::sample_rate])
        #plt.quiver(x, y, u, v, scale=10, scale_units="inches", color=color, label="Estimated")

    # plt.xlim(-25, 100)
    plt.ylim(-20, 220)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid()
    plt.savefig(base_fig_dir / "position.png")

    plt.xlim(100, 200)
    plt.ylim(-20, 40)
    plt.legend()
    plt.savefig(base_fig_dir / "position_zoomed.png")
    plt.close()


def _plot_pos_uncertainty(input: list[KalmanFilter] | list[Hypothesis], base_fig_dir: Path) -> None:
    """Plot longitudinal and lateral uncertainty of position over time."""
    chi2_val = chi2.ppf(0.95, df=2)

    # Create a figure with 2 subplots (vertical layout)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_long, ax_short = axes

    for idx, input_item in enumerate(input):
        if isinstance(input_item, KalmanFilter):
            filter = input_item
            color = COLORS[idx % (N - 2)]
        else:
            filter = input_item.filter
            color = COLORS[input_item.generation % (N - 2)]

        lon_uncertainty_list = []
        lat_uncertainty_list = []
        step_offset = filter.step_offset

        for i in range(step_offset, len(filter)):
            P = filter.get_pred(i).P
            theta = filter.get_pred(i).X[2]
            # Rotation matrix to align with vehicle heading
            u = np.array([np.cos(theta), np.sin(theta)])
            v = np.array([-np.sin(theta), np.cos(theta)])
            maha_u = u.T @ np.linalg.inv(P[:2, :2]) @ u
            maha_v = v.T @ np.linalg.inv(P[:2, :2]) @ v
            lon_uncertainty = np.sqrt(chi2_val / maha_u).item()
            lat_uncertainty = np.sqrt(chi2_val / maha_v).item()
            lon_uncertainty_list.append(lon_uncertainty)
            lat_uncertainty_list.append(lat_uncertainty)

        time = np.arange(step_offset, len(filter)) * 0.1
        ax_long.plot(time, lon_uncertainty_list, label=f"Filter {idx}", color=color)
        ax_short.plot(time, lat_uncertainty_list, label=f"Filter {idx}", color=color)

    # Format first subplot
    ax_long.set_ylabel("one-side 95 CI long [m]")
    ax_long.grid(True)
    ax_long.set_ylim(ymin=0)

    # Format second subplot
    ax_short.set_xlabel("Time [s]")
    ax_short.set_ylabel("one-side 95 CI lat [m]")
    ax_short.grid(True)
    ax_short.set_ylim(ymin=0)

    # Common formatting
    for ax in axes:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))
        # ax.legend()

    # if len(filter_list) > 1:
    #     import ipdb; ipdb.set_trace()

    plt.tight_layout()
    plt.savefig(base_fig_dir / "pos_uncertainty.png")
    plt.close(fig)


@dataclass
class BranchRecord:
    """Timeline metadata for a single hypothesis instance.

    Attributes:
        id: Hypothesis identifier.
        parent_id: Parent hypothesis identifier if any.
        start_time: Hypothesis birth time in seconds.
        end_time: Hypothesis termination time in seconds.
        generation: Hierarchy depth (root hypothesis starts at 1).
        is_active: Whether the hypothesis survived until simulation end.
        weight_history: Logged weight history for the hypothesis.
    """

    id: int
    parent_id: int | None
    start_time: float
    end_time: float
    generation: int
    is_active: bool
    weight_history: list[tuple[int, float]]


def _plot_hypothesis_branching(input: list[Hypothesis], base_fig_dir: Path) -> None:
    """Plot hypothesis branching tree with a time-aligned horizontal axis.

    Args:
        input: Hypotheses produced by integrated localization.
        base_fig_dir: Directory where the branching figure will be persisted.

    Raises:
        ValueError: Raised when hypotheses are missing or if the genealogy is inconsistent.
    """
    if not input:
        raise ValueError("hypothesis list must not be empty when drawing branching timeline")

    time_step = 0.1
    records: dict[int, BranchRecord] = {}
    for hypothesis in input:
        filter_ref = hypothesis.filter
        start_step = filter_ref.step_offset
        end_step = len(filter_ref) - 1
        if end_step < start_step:
            end_step = start_step
        records[hypothesis.id] = BranchRecord(
            id=hypothesis.id,
            parent_id=hypothesis.senior_hypothesis_id,
            start_time=float(start_step) * time_step,
            end_time=float(end_step) * time_step,
            generation=hypothesis.generation,
            is_active=hypothesis.is_active,
            weight_history=sorted(hypothesis.weight_history, key=lambda item: item[0]),
        )

    child_map: dict[int, list[int]] = {record_id: [] for record_id in records}
    for hypothesis in input:
        for child_id in hypothesis.junior_hypothesis_ids:
            if child_id not in records:
                raise ValueError(f"child hypothesis {child_id} is missing from the provided list")
            child_map[hypothesis.id].append(child_id)

    def sort_children(ids: list[int]) -> list[int]:
        return sorted(ids, key=lambda idx: records[idx].start_time)

    y_positions: dict[int, float] = {}
    next_y = 0.0

    def assign_position(hypothesis_id: int) -> float:
        nonlocal next_y
        children = sort_children(child_map[hypothesis_id])
        if not children:
            y_positions[hypothesis_id] = next_y
            next_y += 1.0
            return y_positions[hypothesis_id]
        child_positions = [assign_position(child_id) for child_id in children]
        y_positions[hypothesis_id] = sum(child_positions) / len(child_positions)
        return y_positions[hypothesis_id]

    root_ids = [record_id for record_id, record in records.items() if record.parent_id is None]
    if not root_ids:
        raise ValueError("branching tree must contain at least one root hypothesis")
    for root_id in sorted(root_ids, key=lambda idx: records[idx].start_time):
        if root_id not in y_positions:
            assign_position(root_id)

    min_time = min(record.start_time for record in records.values())
    max_time = max(record.end_time for record in records.values())
    fig_height = len(records) * 0.5 + 1.0
    fig, ax = plt.subplots(figsize=(10, fig_height))

    for record in sorted(records.values(), key=lambda item: item.start_time):
        y_pos = y_positions[record.id]
        color = COLORS[record.generation % N]
        ax.hlines(y_pos, record.start_time, record.end_time, color=color, linewidth=2.5, zorder=2)
        ax.scatter(record.start_time, y_pos, color=color, marker="o", s=30, zorder=3)
        if record.is_active:
            ax.scatter(record.end_time, y_pos, facecolors="white", edgecolors=color, marker="s", s=45, zorder=3)
        else:
            ax.scatter(record.end_time, y_pos, color=color, marker="x", s=45, zorder=3)
        ax.text(
            record.start_time,
            y_pos + 0.1,
            f"H{record.id}",
            fontsize=14,
            color=color,
            ha="left",
            va="bottom",
        )
        change_points: list[tuple[float, float]] = []
        prev_weight: float | None = None
        for step_point, weight_val in record.weight_history:
            if prev_weight is not None and isclose(prev_weight, weight_val, rel_tol=1e-3, abs_tol=1e-4):
                continue
            time_point = step_point * time_step
            time_clamped = min(max(time_point, record.start_time), record.end_time)
            change_points.append((time_clamped, weight_val))
            prev_weight = weight_val
        for change_idx, (time_point, weight_val) in enumerate(change_points):
            text_offset = 0.18 + 0.15 * change_idx
            ax.scatter(time_point, y_pos, color=color, marker="d", s=35, zorder=3)
            ax.text(
                time_point,
                y_pos - text_offset,
                f"w={weight_val:.3f}",
                fontsize=12,
                color=color,
                ha="center",
                va="top",
            )
        if record.parent_id is not None:
            parent_y = y_positions[record.parent_id]
            ax.plot(
                [record.start_time, record.start_time],
                [parent_y, y_pos],
                color="gray",
                linestyle="--",
                linewidth=1.0,
                zorder=1,
            )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Hypothesis hierarchy")
    ax.set_xlim(min_time - time_step, max_time + time_step)
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.savefig(base_fig_dir / "hypothesis_branching.png")
    plt.close(fig)


def _plot_node_existence_scores(
    state_series: TimeseriesTPState,
    hypothesis_list: list[Hypothesis],
    base_fig_dir: Path,
    map: BaseMap,
) -> None:
    """Plot node existence scores with ground truth occupancy signals.

    Args:
        state_series: Ground truth state time series.
        hypothesis_list: Hypotheses containing node existence score histories.
        base_fig_dir: Directory to store generated figures.
        map: Base map instance for resolving node metadata.
    """
    time_step = 0.1
    # {node_id: list of (hypothesis_id, time_idx, scores)}
    node_series: dict[int, list[tuple[int, np.ndarray, list[float]]]] = {}
    for hypothesis in hypothesis_list:
        score_series = hypothesis.node_exsistence_score
        if score_series is None:
            continue
        time_idx = score_series.step_offset + np.arange(len(score_series.scores))
        node_series.setdefault(score_series.node_id, []).append((hypothesis.id, time_idx, score_series.scores))
    if not node_series:
        return

    start_idx = state_series.step_offset
    end_idx = len(state_series)
    if end_idx <= start_idx:
        return
    gt_time = np.arange(start_idx, end_idx) * time_step
    fig, ax = plt.subplots(figsize=(10, 4))
    aggregated_scores: dict[int, np.ndarray] = {}
    node_id_list = set(
        [
            state_series.get_state(i).in_node_id
            for i in range(start_idx, end_idx)
            if state_series.get_state(i) is not None
        ]
    )
    for node_id in node_id_list:
        if node_id is None:
            continue
        gt_signal: list[float] = []
        for step_idx in range(start_idx, end_idx):
            state = state_series.get_state(step_idx)
            gt_signal.append(1.0 if (state is not None and node_id in state.intersect_node_ids) else 0.0)

        if not any(value == 1.0 for value in gt_signal):
            continue
        # import ipdb; ipdb.set_trace()
        ax.step(gt_time, gt_signal, where="post", color="tab:orange", linewidth=2.0, label="Ground truth occupancy")
        ax.fill_between(gt_time, 0, gt_signal, step="post", color="tab:orange", alpha=0.2)
        # put text label for node id
        pos = gt_signal.index(1.0)
        ax.text(
            (pos + start_idx) * time_step,
            1.05,
            f"Node {node_id}",
            fontsize=14,
            color="tab:orange",
            ha="right",
            va="center",
        )

        score_entries = node_series.get(node_id, [])
        for hypothesis_id, time_idx, scores in score_entries:
            # Accumulate scores on the shared time axis anchored to the ground-truth window
            agg_buffer = aggregated_scores.setdefault(node_id, np.zeros(end_idx - start_idx))
            for step_idx, score in zip(time_idx, scores):
                if start_idx <= step_idx < end_idx:
                    agg_buffer[step_idx - start_idx] += score

        if node_id in aggregated_scores:
            summed_scores = aggregated_scores[node_id]
            color = COLORS[node_id % N]
            if np.any(summed_scores):
                ax.plot(
                    gt_time,
                    summed_scores,
                    linewidth=2.5,
                    color=color,
                    label=f"Sum score (node {node_id})",
                )

    ax.hlines(
        0.05,
        xmin=gt_time[0],
        xmax=gt_time[-1],
        colors="red",
        linestyles="--",
        label="Existence threshold",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Probability / Indicator")
    ax.set_ylim(0.0, 1.2)
    ax.set_xlim(gt_time[0], gt_time[-1])
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(base_fig_dir / "node_existence_node.png")
    plt.close(fig)


def _plot_node_existence_scores_with_cv(
    cyc_state_series: TimeseriesTPState,
    hypothesis_list: list[Hypothesis],
    old_cv_state_series: TimeseriesTPState,
    new_cv_state_series: TimeseriesTPState,
    node_id: int,
    base_fig_dir: Path,
) -> None:
    """Plot node existence scores with ground truth occupancy signals.

    Args:
        cyc_state_series: Ground truth state time series for the cyclist
        hypothesis_list: Hypotheses containing node existence score histories.
        old_cv_state_series: Ground truth state time series for the old control vehicle
        new_cv_state_series: Ground truth state time series for the new control vehicle
        node_id: Node ID to visualize
        base_fig_dir: Directory to store generated figures.
    """
    time_step = 0.1
    # {node_id: list of (hypothesis_id, time_idx, scores)}
    node_series: dict[int, list[tuple[int, np.ndarray, list[float]]]] = {}
    for hypothesis in hypothesis_list:
        score_series = hypothesis.node_exsistence_score
        if score_series is None:
            continue
        time_idx = score_series.step_offset + np.arange(len(score_series.scores))
        node_series.setdefault(score_series.node_id, []).append((hypothesis.id, time_idx, score_series.scores))
    if not node_series:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, state_series in zip(["Cyclist", "Old Control Vehicle", "New Control Vehicle"], [cyc_state_series, old_cv_state_series, new_cv_state_series]):
        start_idx = state_series.step_offset
        end_idx = len(state_series)
        if end_idx <= start_idx:
            return
        gt_time = np.arange(start_idx, end_idx) * time_step
        aggregated_scores: dict[int, np.ndarray] = {}
        gt_signal: list[float] = []
        for step_idx in range(start_idx, end_idx):
            state = state_series.get_state(step_idx)
            gt_signal.append(1.0 if (state is not None and node_id in state.intersect_node_ids) else 0.0)
        if not any(value == 1.0 for value in gt_signal):
            print(f"Skipping node {node_id} as it is never occupied in the given state series")
            continue
        ax.step(gt_time, gt_signal, where="post", linewidth=2.0, label=label)
        ax.fill_between(gt_time, 0, gt_signal, step="post", alpha=0.2)

    score_entries = node_series.get(node_id, [])
    for hypothesis_id, time_idx, scores in score_entries:
        # Accumulate scores on the shared time axis anchored to the ground-truth window
        agg_buffer = aggregated_scores.setdefault(node_id, np.zeros(end_idx - start_idx))
        for step_idx, score in zip(time_idx, scores):
            if start_idx <= step_idx < end_idx:
                agg_buffer[step_idx - start_idx] += score

    if node_id in aggregated_scores:
        summed_scores = aggregated_scores[node_id]
        color = COLORS[node_id % N]
        if np.any(summed_scores):
            ax.plot(
                gt_time,
                summed_scores,
                linewidth=2.5,
                color=color,
                label=f"Sum score (node {node_id})",
            )

    ax.hlines(
        0.05,
        xmin=gt_time[0],
        xmax=gt_time[-1],
        colors="red",
        linestyles="--",
        label="Existence threshold",
    )
    ax.legend()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Probability / Indicator")
    ax.set_ylim(0.0, 1.2)
    ax.set_xlim(gt_time[0], gt_time[-1])
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(base_fig_dir / "node_existence_node.png")
    plt.close(fig)


def plot_collision_avoidance(
    time_step: float,
    cyclist_scores: dict[int, dict[int, float]],
    control_scores: dict[int, dict[int, float]],
    velocity_profile: tuple[np.ndarray, np.ndarray],
    command: "ControlCommand | None",
    base_fig_dir: Path,
    focus_node_id: int | None = None,
) -> None:
    """Visualize collision-avoidance probabilities and control-car velocity."""
    if not cyclist_scores:
        return
    if focus_node_id is None:
        focus_node_id = command.node_id if command else next(iter(cyclist_scores))

    def _timeline_to_arrays(score_map: dict[int, dict[int, float]], node_id: int) -> tuple[np.ndarray, np.ndarray]:
        timeline = score_map.get(node_id, {})
        if not timeline:
            return np.array([]), np.array([])
        steps = sorted(timeline.keys())
        times = np.array(steps, dtype=float) * time_step
        values = np.array([timeline[step] for step in steps], dtype=float)
        return times, values

    cyclist_time, cyclist_vals = _timeline_to_arrays(cyclist_scores, focus_node_id)
    control_time, control_vals = _timeline_to_arrays(control_scores, focus_node_id)
    vel_time, velocities = velocity_profile

    fig, (ax_prob, ax_vel) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    if cyclist_time.size > 0:
        ax_prob.plot(cyclist_time, cyclist_vals, label="Cyclist prob.", color="tab:blue", linewidth=2.0)
    if control_time.size > 0:
        ax_prob.step(control_time, control_vals, where="post", label="Control car prob.", color="tab:orange")
    if command is not None:
        ax_prob.axvline(command.time_sec, color="red", linestyle="--", label="Decel command")
    ax_prob.set_ylabel("Node occupancy prob.")
    ax_prob.set_title(f"Node {focus_node_id} occupancy probabilities")
    ax_prob.set_ylim(0.0, 1.2)
    ax_prob.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_prob.legend()

    if vel_time.size > 0:
        ax_vel.plot(vel_time, velocities, label="Control car speed", color="tab:green")
    if command is not None:
        ax_vel.axvline(command.time_sec, color="red", linestyle="--")
    ax_vel.set_xlabel("Time [s]")
    ax_vel.set_ylabel("Velocity [m/s]")
    ax_vel.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_vel.legend()

    plt.tight_layout()
    base_fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(base_fig_dir / "collision_avoidance.png")
    plt.close(fig)
