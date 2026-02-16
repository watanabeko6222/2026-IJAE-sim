from __future__ import annotations

from copy import deepcopy

import numpy as np

from src.filter.kalman_filter import KalmanFilter
from src.filter.integrated_localization import AllNodeExsistenceScore, Hypothesis
from src.simulator.map_constructor import BaseMap
from src.simulator.moving_generator import ControlCommandSeries, MovingSimulator, TimeseriesTPState


def calc_gt_node_existence(state_series: TimeseriesTPState, map: BaseMap) -> AllNodeExsistenceScore:
    """Return deterministic node-occupancy indicators derived from ground-truth states."""

    gt_existence = AllNodeExsistenceScore(step_offset=state_series.step_offset, node_scores={})

    for node in map.nodes:
        gt_existence.node_scores[node.id] = [0.0] * (len(state_series.state_list))

    for local_idx, state in enumerate(state_series.state_list):
        for node_id in state.intersect_node_ids:
            gt_existence.node_scores[node_id][local_idx] = 1.0

    return gt_existence


def pred_node_existence(
    state_series: TimeseriesTPState, hypothesis_list: list[Hypothesis], map: BaseMap
) -> AllNodeExsistenceScore:
    """Aggregate node existence scores across hypotheses keyed by global step index."""
    aggregated_scores = AllNodeExsistenceScore(step_offset=state_series.step_offset, node_scores={})

    for node in map.nodes:
        aggregated_scores.node_scores[node.id] = [0.0] * (len(state_series.state_list))
    for hypothesis in hypothesis_list:
        series = hypothesis.node_exsistence_score
        if series is None:
            continue
        for step_idx, score in enumerate(series.scores):
            aggregated_scores.node_scores[series.node_id][step_idx] += score
    return aggregated_scores


class ControlCarPlanner:
    """Derive slowdown commands by comparing control-car and cyclist node scores."""

    def __init__(
        self,
        map: BaseMap,
        hypothesis_list: list[Hypothesis],
        cv_movement: MovingSimulator,
        time_step: float = 0.1,
        threshold: float = 0.05,
        target_speed: float = 30.0 / 3.6,
        ref_accel: float = 0.1 * 9.81,
        future_steps: int = 50,
    ) -> None:
        self.hypothesis_list = hypothesis_list
        self.cv_movement = cv_movement
        self.time_step = time_step
        self.threshold = threshold
        self.target_speed = target_speed
        self.map = map
        self.future_steps = future_steps
        self.ref_accel = ref_accel

    def pred_node_existence(
        self, hypothesis_list: list[Hypothesis], map: BaseMap
    ) -> AllNodeExsistenceScore:
        """Aggregate node existence scores across hypotheses keyed by global step index."""
        step_offset = self.cv_movement.step_offset
        aggregated_scores = AllNodeExsistenceScore(step_offset=step_offset, node_scores={})

        for node in map.nodes:
            aggregated_scores.node_scores[node.id] = [0.0] * (len(self.cv_movement) - step_offset)
        for hypothesis in hypothesis_list:
            series = hypothesis.node_exsistence_score
            if series is None:
                continue
            for step_idx in range(step_offset, len(self.cv_movement)):
                score = series.get_score(step_idx)
                if score is None:
                    aggregated_scores.node_scores[series.node_id][step_idx - step_offset] += 0.0
                else:
                    aggregated_scores.node_scores[series.node_id][step_idx - step_offset] += score

        return aggregated_scores

    def get_control_command(
        self,
        now_step: int,
        future_horizon_steps: int,
        cyclist_scores: AllNodeExsistenceScore,
    ) -> ControlCommandSeries:
        """Return a slowdown command if a potential collision is detected."""
        now_state = self.cv_movement.get_state(now_step)
        accel_rate = min(self.ref_accel, (self.target_speed - now_state.vel) / 10)
        node_id = None
        cv_scores = []
        for plus_step in range(now_step, now_step + future_horizon_steps + 1):
            future_cv_state = self.cv_movement.get_state(plus_step)
            if future_cv_state is None:
                cv_scores.append(0.0)
                continue
            if future_cv_state.intersect_node_ids:
                node_id = future_cv_state.intersect_node_ids[0]  # FIXME: only consider first node
                cv_scores.append(1.0)
            else:
                cv_scores.append(0.0)
        if node_id is None:
            return ControlCommandSeries(step_offset=now_step, commands=[accel_rate] * 10)
        cv_scores = np.array(cv_scores)

        cyc_scores = []
        for plus_step in range(now_step, now_step + future_horizon_steps + 1):
            pred_idx = plus_step - cyclist_scores.step_offset
            if pred_idx >= len(cyclist_scores.node_scores[node_id]) - 1:
                cyc_scores.append(0.0)
                continue
            cyc_score = cyclist_scores.node_scores[node_id][pred_idx]
            cyc_scores.append(cyc_score)
        cyc_scores = np.array(cyc_scores)

        joint_scores = cv_scores * cyc_scores
        collision_indices = np.where(joint_scores >= self.threshold)[0]
        if len(collision_indices) == 0:
            return ControlCommandSeries(step_offset=now_step, commands=[accel_rate] * 10)

        decel_rate = - now_state.vel / (collision_indices[0] * self.time_step)
        return ControlCommandSeries(step_offset=now_step, commands=[decel_rate] * 10)

    def run(self):
        """Run the planner over the entire control-car movement timeline."""
        command_series_list = []
        start_step = self.cv_movement.step_offset
        step_idx = start_step
        done_flag = False

        while done_flag is False:
            command_series = self.get_control_command(
                now_step=step_idx,
                future_horizon_steps=self.future_steps,
                cyclist_scores=self.pred_node_existence(self.hypothesis_list, self.map),
            )
            done_flag = self.cv_movement.regenerate_state_with_control(command_series, steps_ahead=self.future_steps)
            command_series_list.append(deepcopy(command_series))
            step_idx += 10

        return command_series_list, self.cv_movement


class ControlCarPlannerWithEKF:
    """Derive slowdown commands by comparing control-car and cyclist node scores."""

    def __init__(
        self,
        map: BaseMap,
        hypothesis_list: list[Hypothesis],
        cv_movement: MovingSimulator,
        cv_ego_filter: KalmanFilter,
        time_step: float = 0.1,
        threshold: float = 0.05,
        target_speed: float = 30.0 / 3.6,
        ref_accel: float = 0.1 * 9.81,
        future_steps: int = 50,
    ) -> None:
        self.hypothesis_list = hypothesis_list
        self.cv_movement = cv_movement
        self.cv_ego_filter = cv_ego_filter
        self.time_step = time_step
        self.threshold = threshold
        self.target_speed = target_speed
        self.map = map
        self.future_steps = future_steps
        self.ref_accel = ref_accel

    def pred_node_existence(
        self, hypothesis_list: list[Hypothesis], map: BaseMap
    ) -> AllNodeExsistenceScore:
        """Aggregate node existence scores across hypotheses keyed by global step index."""
        step_offset = self.cv_movement.step_offset
        aggregated_scores = AllNodeExsistenceScore(step_offset=step_offset, node_scores={})

        for node in map.nodes:
            aggregated_scores.node_scores[node.id] = [0.0] * (len(self.cv_movement) - step_offset)
        for hypothesis in hypothesis_list:
            series = hypothesis.node_exsistence_score
            if series is None:
                continue
            for step_idx in range(step_offset, len(self.cv_movement)):
                score = series.get_score(step_idx)
                if score is None:
                    aggregated_scores.node_scores[series.node_id][step_idx - step_offset] += 0.0
                else:
                    aggregated_scores.node_scores[series.node_id][step_idx - step_offset] += score

        return aggregated_scores

    def get_control_command(
        self,
        now_step: int,
        future_horizon_steps: int,
        cyclist_scores: AllNodeExsistenceScore,
    ) -> ControlCommandSeries:
        """Return a slowdown command if a potential collision is detected."""
        now_state = self.cv_movement.get_state(now_step)
        accel_rate = min(self.ref_accel, (self.target_speed - now_state.vel) / 10)
        node_id = None
        cv_scores = []
        for plus_step in range(now_step, now_step + future_horizon_steps + 1):
            future_cv_state = self.cv_movement.get_state(plus_step)
            if future_cv_state is None:
                cv_scores.append(0.0)
                continue
            if future_cv_state.intersect_node_ids:
                node_id = future_cv_state.intersect_node_ids[0]  # FIXME: only consider first node
                cv_scores.append(1.0)
            else:
                cv_scores.append(0.0)
        if node_id is None:
            return ControlCommandSeries(step_offset=now_step, commands=[accel_rate] * 10)
        cv_scores = np.array(cv_scores)

        cyc_scores = []
        for plus_step in range(now_step, now_step + future_horizon_steps + 1):
            pred_idx = plus_step - cyclist_scores.step_offset
            if pred_idx >= len(cyclist_scores.node_scores[node_id]) - 1:
                cyc_scores.append(0.0)
                continue
            cyc_score = cyclist_scores.node_scores[node_id][pred_idx]
            cyc_scores.append(cyc_score)
        cyc_scores = np.array(cyc_scores)

        joint_scores = cv_scores * cyc_scores
        collision_indices = np.where(joint_scores >= self.threshold)[0]
        if len(collision_indices) == 0:
            return ControlCommandSeries(step_offset=now_step, commands=[accel_rate] * 10)

        decel_rate = - now_state.vel / (collision_indices[0] * self.time_step)
        return ControlCommandSeries(step_offset=now_step, commands=[decel_rate] * 10)

    def run(self):
        """Run the planner over the entire control-car movement timeline."""
        command_series_list = []
        start_step = self.cv_movement.step_offset
        step_idx = start_step

        while True:
            if step_idx + self.future_steps >= len(self.cv_movement):
                break
            command_series = self.get_control_command(
                now_step=step_idx,
                future_horizon_steps=self.future_steps,
                cyclist_scores=self.pred_node_existence(self.hypothesis_list, self.map),
            )
            self.cv_movement.regenerate_state_with_control(command_series)
            command_series_list.append(deepcopy(command_series))
            step_idx += 10

        return command_series_list, self.cv_movement
