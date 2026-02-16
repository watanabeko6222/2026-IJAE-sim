from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.stats import chi2

from .filter.integrated_localization import Hypothesis
from .simulator.moving_generator import TimeseriesTPState
from .utils import normalize_angle


@dataclass
class MetricPerTrial:
    mean_ci95_pos_longitudinal: float
    mean_ci95_pos_lateral: float
    mean_ci95_theta: float
    mean_ci95_vel: float
    conservertive_ratio_within_ci95_pos: float
    conservertive_ratio_within_ci95_theta: float
    conservertive_ratio_within_ci95_vel: float
    weighted_ratio_within_ci95_pos: float
    weighted_ratio_within_ci95_theta: float
    weighted_ratio_within_ci95_vel: float

    score_error: float
    false_alarm_num: int
    miss_alarm_num: int
    existence_tp_ratio: float
    existence_fn_ratio: float


def calc_metrics(
    cyclist_state_series: TimeseriesTPState,
    hypothesis_list: list[Hypothesis],
    exclude_node_ids: list[int] = [],
) -> MetricPerTrial:
    within_ci95_pos_flags: list[bool] = []
    within_ci95_theta_flags: list[bool] = []
    within_ci95_vel_flags: list[bool] = []
    within_ci95_pos_scores: list[float] = []
    within_ci95_theta_scores: list[float] = []
    within_ci95_vel_scores: list[float] = []

    ci95_pos_longitudinals = []
    ci95_pos_laterals = []
    ci95_thetas = []
    ci95_vels = []

    score_errors = []
    gt_in_node_ids = set()
    warned_node_ids = set()
    existence_tp_count = 0
    existence_fn_count = 0

    chi2_2d_value = chi2.ppf(0.95, df=2)  # 95% threshold for 2 DOF
    chi2_1d_value = chi2.ppf(0.95, df=1)  # 95% threshold for 1 DOF

    # {node_id: list of (hypothesis_id, time_idx, scores)}
    node_series: dict[int, list[tuple[int, np.ndarray, list[float]]]] = {}
    for hypothesis in hypothesis_list:
        score_series = hypothesis.node_exsistence_score
        if score_series is None:
            continue
        time_idx = score_series.step_offset + np.arange(len(score_series.scores))
        node_series.setdefault(score_series.node_id, []).append((hypothesis.id, time_idx, score_series.scores))

    # {node_id;: score ndarray}
    aggregated_scores: dict[int, np.ndarray] = {}
    for node_id in node_series.keys():
        score_entries = node_series.get(node_id, [])
        for hypothesis_id, time_idx, scores in score_entries:
            # Accumulate scores on the shared time axis anchored to the ground-truth window
            agg_buffer = aggregated_scores.setdefault(node_id, np.zeros(len(cyclist_state_series)))
            for step_idx, score in zip(time_idx, scores):
                if cyclist_state_series.step_offset <= step_idx < len(cyclist_state_series):
                    agg_buffer[step_idx - cyclist_state_series.step_offset] += score

    for i in range(cyclist_state_series.step_offset, len(cyclist_state_series)):
        if i >= len(cyclist_state_series):
            break
        gt_state = cyclist_state_series.get_state(i)
        if gt_state is None:
            continue
        if gt_state.can_be_observed is False:
            continue
        within_ci95_pos_flag = False
        within_ci95_pos_score = 0.0
        within_ci95_theta_flag = False
        within_ci95_theta_score = 0.0
        within_ci95_vel_flag = False
        within_ci95_vel_score = 0.0

        ci95_pos_longitudinal = 0.0
        ci95_pos_lateral = 0.0
        ci95_theta = 0.0
        ci95_vel = 0.0
        pos_gt = np.array([[gt_state.x], [gt_state.y]]).flatten()
        vel_gt = gt_state.vel
        theta_gt = gt_state.theta

        debug_weight_list = []
        for hypothesis in hypothesis_list:
            cyclist_filter = hypothesis.filter
            if cyclist_filter.step_offset > i:  # filter not yet initialized
                continue
            if len(cyclist_filter) <= i:  # filter already ended
                continue
            pred = cyclist_filter.get_pred(i)
            weight = hypothesis.get_weight(i)
            debug_weight_list.append(weight)
            if pred is None:
                continue
            X, P = pred.X, pred.P
            pos_mean = X[0:2, 0]
            theta_mean = X[2, 0]
            vel_mean = X[3, 0]
            pos_cov = P[0:2, 0:2]
            theta_var = P[2, 2]
            vel_var = P[3, 3]

            # check position
            diff_pos = pos_gt - pos_mean
            mahalanobis_pos = diff_pos.T @ np.linalg.inv(pos_cov) @ diff_pos
            if mahalanobis_pos <= chi2_2d_value:
                within_ci95_pos_flag = True
                within_ci95_pos_score += weight
            else:
                within_ci95_pos_flag = within_ci95_pos_flag or False  # keep previous True

            # Calculate 95% P size for position
            u = np.array([np.cos(theta_mean), np.sin(theta_mean)])
            v = np.array([-np.sin(theta_mean), np.cos(theta_mean)])
            maha_u = u.T @ np.linalg.inv(pos_cov) @ u
            maha_v = v.T @ np.linalg.inv(pos_cov) @ v
            _ci95_pos_longitudinal = np.sqrt(chi2_2d_value / maha_u).item()
            _ci95_pos_lateral = np.sqrt(chi2_2d_value / maha_v).item()

            ci95_pos_longitudinal += weight * _ci95_pos_longitudinal
            ci95_pos_lateral += weight * _ci95_pos_lateral

            # check theta
            diff_theta = theta_gt - theta_mean
            diff_theta = normalize_angle(diff_theta)
            mahalanobis_theta = (diff_theta**2) / theta_var
            if mahalanobis_theta <= chi2_1d_value:
                within_ci95_theta_flag = True
                within_ci95_theta_score += weight
            else:
                within_ci95_theta_flag = within_ci95_theta_flag or False  # keep previous True

            # Calculate 95% P size for theta
            _ci95_theta = float(np.sqrt(theta_var) * np.sqrt(chi2_1d_value))
            ci95_theta += weight * _ci95_theta

            # check velocity
            diff_vel = vel_gt - vel_mean
            mahalanobis_vel = (diff_vel**2) / vel_var
            if mahalanobis_vel <= chi2_1d_value:
                within_ci95_vel_flag = True
                within_ci95_vel_score += weight
            else:
                within_ci95_vel_flag = within_ci95_vel_flag or False  # keep previous True
            # Calculate 95% P size for velocity
            _ci95_vel = float(np.sqrt(vel_var) * np.sqrt(chi2_1d_value))
            ci95_vel += weight * _ci95_vel

        assert within_ci95_pos_score - 1.0 < 1e-6, "Hypothesis weights must sum to 1."
        assert within_ci95_theta_score - 1.0 < 1e-6, "Hypothesis weights must sum to 1."
        assert within_ci95_vel_score - 1.0 < 1e-6, "Hypothesis weights must sum to 1."

        within_ci95_pos_flags.append(within_ci95_pos_flag)
        within_ci95_pos_scores.append(within_ci95_pos_score)
        within_ci95_theta_flags.append(within_ci95_theta_flag)
        within_ci95_theta_scores.append(within_ci95_theta_score)
        within_ci95_vel_flags.append(within_ci95_vel_flag)
        within_ci95_vel_scores.append(within_ci95_vel_score)

        ci95_pos_longitudinals.append(ci95_pos_longitudinal)
        ci95_pos_laterals.append(ci95_pos_lateral)
        ci95_thetas.append(ci95_theta)
        ci95_vels.append(ci95_vel)

        # Calculate existence metrics
        gt_node_ids = gt_state.intersect_node_ids
        assert len(gt_node_ids) < 2, "Cyclist is assumed to be on a single node."
        if len(gt_node_ids) == 1:
            target_node_id = gt_node_ids[0]
            if target_node_id in exclude_node_ids:
                continue
            gt_in_node_ids.add(target_node_id)
            agg_scores = aggregated_scores.get(target_node_id, np.zeros(len(cyclist_state_series)))
            agg_score = agg_scores[i - cyclist_state_series.step_offset]
            threshold = 0.05  # Example threshold
            score_errors.append(1.0 - agg_score)
            if agg_score >= threshold:
                existence_tp_count += 1
            else:
                existence_fn_count += 1
                logger.debug(f"Existence FN at step {i}, node {target_node_id}, score {agg_score}, node_id {gt_node_ids}")

        for node_id, scores in aggregated_scores.items():
            if node_id in exclude_node_ids:
                continue
            agg_score = scores[i - cyclist_state_series.step_offset]
            threshold = 0.05  # Example threshold
            if agg_score >= threshold:
                warned_node_ids.add(node_id)

    miss_alarmed_node_ids = gt_in_node_ids - warned_node_ids
    false_alarmed_node_ids = warned_node_ids - gt_in_node_ids
    
    logger.debug(f"GT in nodes: {gt_in_node_ids}, Warned nodes: {warned_node_ids}")
    logger.debug(f"False alarms: {false_alarmed_node_ids}, Miss alarms: {miss_alarmed_node_ids}")

    result = MetricPerTrial(
        mean_ci95_pos_longitudinal=float(np.mean(ci95_pos_longitudinals)),
        mean_ci95_pos_lateral=float(np.mean(ci95_pos_laterals)),
        mean_ci95_theta=float(np.mean(ci95_thetas)),
        mean_ci95_vel=float(np.mean(ci95_vels)),
        conservertive_ratio_within_ci95_pos=float(np.mean(within_ci95_pos_flags)),
        conservertive_ratio_within_ci95_theta=float(np.mean(within_ci95_theta_flags)),
        conservertive_ratio_within_ci95_vel=float(np.mean(within_ci95_vel_flags)),
        weighted_ratio_within_ci95_pos=float(np.mean(within_ci95_pos_scores)),
        weighted_ratio_within_ci95_theta=float(np.mean(within_ci95_theta_scores)),
        weighted_ratio_within_ci95_vel=float(np.mean(within_ci95_vel_scores)),
        score_error=float(np.mean(score_errors)) if score_errors else 0.0,
        false_alarm_num=len(false_alarmed_node_ids),
        miss_alarm_num=len(miss_alarmed_node_ids),
        existence_tp_ratio=existence_tp_count / (existence_tp_count + existence_fn_count),
        existence_fn_ratio=existence_fn_count / (existence_tp_count + existence_fn_count),
    )
    return result
