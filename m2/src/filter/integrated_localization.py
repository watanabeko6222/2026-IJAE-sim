from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from scipy.stats import norm

from ..simulator.map_constructor import BaseMap
from ..simulator.moving_generator import TimeseriesTPState
from ..utils import chi2_95_1d, chi2_95_4d, normalize_angle
from .kalman_filter import KalmanFilter, KalmanPrediction
from .sensor import LinkInfoSensor


@dataclass
class NodeExsistenceScore:
    node_id: int
    step_offset: int
    scores: list[float] = field(default_factory=list)

    def __str__(self):
        return f"NodeExsistenceScore(node_id={self.node_id}, step_offset={self.step_offset}, len={len(self.scores)})"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.scores) + self.step_offset

    def get_score(self, step_idx: int) -> float | None:
        local_idx = step_idx - self.step_offset
        if local_idx < 0 or local_idx >= len(self.scores):
            return None
        return self.scores[local_idx]


@dataclass
class AllNodeExsistenceScore:
    step_offset: int
    node_scores: dict[int, list[float]]  # node_id -> list of scores


@dataclass
class Hypothesis:
    # Immutable
    id: int
    senior_hypothesis_id: int | None
    junior_hypothesis_ids: list[int]
    generated_node_id: int | None

    # Mutable
    filter: KalmanFilter
    is_active: bool
    generation: int
    senior2junior: bool | None
    active_link_id: tuple[int, int] | None
    weight: float = field(default=0.0)
    weight_history: list[tuple[float, float]] = field(default_factory=list)
    node_exsistence_score: NodeExsistenceScore | None = field(default=None)

    def get_weight(self, step_idx: int) -> float | None:
        for t, w in reversed(self.weight_history):
            if t <= step_idx:
                return w
        return None


def ex_repetitive_predict(
    X: np.ndarray,
    P: np.ndarray,
    dt: float,
    Us: list[np.ndarray],
    Qs: list[np.ndarray],
    link_info_sensor: LinkInfoSensor | None,
    link_id: tuple[int, int],
    map: BaseMap,
    steps: int,
) -> KalmanPrediction:
    """Predict next state multiple times
    Args:
        X: state vector; x, y, theta, velocity
        P: state covariance matrix
        dt: time step
        U: control vector; acceleration, angular velocity
        Q: control covariance matrix
        steps: number of prediction steps
    Returns:
        X: predicted state vector
        P: predicted state covariance matrix
    """
    assert len(Us) == len(Qs) == steps, "Length of Us and Qs must be equal to steps"
    for i in range(steps):
        U = Us[i]
        Q = Qs[i]
        F = np.array(
            [
                [1, 0, -(X[3, 0]) * np.sin(X[2, 0]) * dt, np.cos(X[2, 0]) * dt],
                [0, 1, (X[3, 0]) * np.cos(X[2, 0]) * dt, np.sin(X[2, 0]) * dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        B = np.array(
            [
                [0, (np.cos(X[2, 0]) * dt**2) / 2],
                [0, (np.sin(X[2, 0]) * dt**2) / 2],
                [dt, 0],
                [0, dt],
            ]
        )
        X = X + np.array(
            [
                [X[3, 0] * np.cos(X[2, 0]) * dt + U[0, 0] * dt**2 / 2],
                [X[3, 0] * np.sin(X[2, 0]) * dt + U[1, 0] * dt**2 / 2],
                [U[0, 0] * dt],
                [U[1, 0] * dt],
            ]
        )
        P = F @ P @ F.T + B @ Q @ B.T

        if link_info_sensor is not None and i % int(10 / link_info_sensor.fps) == 0:
            now_link = map.id2link[link_id]
            link_theta = now_link.theta
            p0 = np.array([now_link.entrance_point.x, now_link.entrance_point.y])

            # along road t = [cosθ, sinθ]
            # normal n = [-sinθ, cosθ]
            n = np.array([-np.sin(link_theta), np.cos(link_theta)])
            c = -np.dot(n, p0)  # ax + by + c = 0

            H = np.array([[n[0], n[1], 0, 0]])
            Z = np.array([[-c]])
            R = np.array([[link_info_sensor.pos_std**2]])

            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
            X = X + K @ (Z - H @ X)
            P = (np.eye(4) - K @ H) @ P @ (np.eye(4) - K @ H).T + K @ R @ K.T

    return KalmanPrediction(X, P)


class IDIncrementer:
    def __init__(self) -> None:
        self.current_id = 1

    def get_next_id(self) -> int:
        next_id = self.current_id
        self.current_id += 1
        return next_id


@dataclass
class ProvidedInformation:
    X: np.ndarray
    P: np.ndarray
    link_id: tuple[int, int] | None


class IntegrationTPFilter:
    def __init__(
        self,
        map: BaseMap,
        time_step: float,
        tp_const_sigma_ww: float,
        tp_const_sigma_aa: float,
        link_info_sensor: LinkInfoSensor | None = None,
        future_pred_steps: int = 50,
    ):
        self.map = map
        self.time_step = time_step
        self.tp_const_sigma_ww = tp_const_sigma_ww
        self.tp_const_sigma_aa = tp_const_sigma_aa
        self.link_info_sensor = link_info_sensor
        self.future_pred_steps = future_pred_steps
        self.hypothesis_list: list[Hypothesis] = []
        self.id_incrementer = IDIncrementer()

    def normalize_weights(self, time_idx: int) -> None:
        """Normalize active hypothesis weights and record the new values."""
        if time_idx < 0:
            raise ValueError("time_idx should be non-negative when normalizing weights")
        total_weight = sum(hypothesis.weight if hypothesis.is_active else 0 for hypothesis in self.hypothesis_list)
        assert total_weight > 0, "Total weight of active hypotheses is zero during normalization"
        assert total_weight <= 1.0, "Total weight of active hypotheses is unreasonably large during normalization"

        if abs(total_weight - 1.0) < 1e-6:
            return self.hypothesis_list

        normalized_list: list[Hypothesis] = []
        for hypothesis in self.hypothesis_list:
            if not hypothesis.is_active:
                normalized_list.append(hypothesis)
                continue
            hypothesis.weight /= total_weight
            assert hypothesis.weight >= 0.0, (
                f"Hypothesis weight became negative after normalization: {hypothesis.weight}"
            )
            assert hypothesis.weight <= 1.0, f"Hypothesis weight exceeded 1.0 after normalization: {hypothesis.weight}"
            hypothesis.weight_history.append((time_idx, hypothesis.weight))
            normalized_list.append(hypothesis)

        assert abs(sum(h.weight for h in normalized_list if h.is_active) - 1.0) < 1e-6, (
            "Normalized weights do not sum to 1"
        )
        self.hypothesis_list = normalized_list

    def _collect_provided_information(
        self, provided_filters: list[KalmanFilter], t: int
    ) -> list["ProvidedInformation"]:
        infos: list[ProvidedInformation] = []
        for filter in provided_filters:
            # Align bounds with each observer trajectory/filter length.
            if t >= len(filter):
                continue
            if t < filter.step_offset:
                continue
            if (t - filter.step_offset) % 10 != 0:
                continue

            pred = filter.get_pred(t)
            X, P = pred.X, pred.P
            mean = X[:2, 0]
            cov = P[:2, :2]
            in_link_id = self.map.pred_in_only_link(mean, cov)
            infos.append(ProvidedInformation(X=X, P=P, link_id=in_link_id))
        return infos

    def _initialize_first_hypothesis(self, t: int, infos: list["ProvidedInformation"]) -> None:
        if len([hypothesis for hypothesis in self.hypothesis_list if hypothesis.is_active]) != 0:
            return
        if not infos:
            return
        first_filter = KalmanFilter(step_offset=t)

        first_pred = infos[0]
        in_link_id, first_X, first_P = (
            first_pred.link_id,
            first_pred.X,
            first_pred.P,
        )
        first_filter.X = first_X.copy()
        first_filter.P = first_P.copy()
        first_filter.X_init = first_X.copy()
        first_filter.P_init = first_P.copy()

        first_hypothesis = Hypothesis(
            id=self.id_incrementer.get_next_id(),
            senior_hypothesis_id=None,
            junior_hypothesis_ids=[],
            filter=first_filter,
            active_link_id=in_link_id,
            generated_node_id=None,
            is_active=True,
            generation=1,
            senior2junior=None,
            weight=1.0,
        )
        first_hypothesis.weight_history.append((t, first_hypothesis.weight))
        self.hypothesis_list.append(first_hypothesis)
        logger.trace(f"Time {t}: Hypothesis {first_hypothesis.id} initialized on link {in_link_id}")

    def _update_hypotheses_with_measurements(
        self, infos: list["ProvidedInformation"], H: np.ndarray, t: int
    ) -> list[int]:
        updated_hypothesis_ids: list[int] = []
        for info in infos:
            info_X, info_P, in_link_id = info.X, info.P, info.link_id
            for hypothesis in self.hypothesis_list:
                if not hypothesis.is_active:
                    continue
                if hypothesis.junior_hypothesis_ids:
                    continue
                if hypothesis.active_link_id != in_link_id:
                    continue
                filter = hypothesis.filter
                diff_angle = normalize_angle(info_X[2, 0] - filter.X[2, 0])
                info_X[2, 0] = filter.X[2, 0] + diff_angle
                innovation = info_X - filter.X
                S = filter.P + info_P
                maha_distance = (innovation.T @ np.linalg.inv(S) @ innovation)[0, 0]

                in_95ci_flag = maha_distance < chi2_95_4d
                link_consistent_flag = hypothesis.active_link_id == in_link_id
                if not in_95ci_flag and not link_consistent_flag:
                    continue

                # wrapping measurement angle to be close to prediction
                pred_theta = hypothesis.filter.X[2, 0]
                info_theta = info_X[2, 0]
                diff_theta = normalize_angle(info_theta - pred_theta)
                info_X[2, 0] = pred_theta + diff_theta

                hypothesis.filter.update_CI(info_X, info_P, H)
                updated_hypothesis_ids.append(hypothesis.id)
        return updated_hypothesis_ids

    def _deactivate_unupdated_hypotheses(self, updated_hypothesis_ids: list[int], t: int) -> None:
        for hypothesis in self.hypothesis_list:
            if not hypothesis.is_active:
                continue
            if hypothesis.id in updated_hypothesis_ids:
                continue
            hypothesis.weight = 0.0
            hypothesis.weight_history.append((t, hypothesis.weight))
            hypothesis.is_active = False
            logger.trace((f"Time {t}: Hypothesis {hypothesis.id} deactivated: "))
        self.normalize_weights(time_idx=t)

    def _apply_link_info_observation(self, t: int) -> None:
        if not self.link_info_sensor:
            return
        if t % int(10 / self.link_info_sensor.fps) != 0:
            return
        for hypothesis in self.hypothesis_list:
            if hypothesis.is_active and hypothesis.active_link_id is not None:
                update_flag = True
                if update_flag:
                    self.link_info_sensor.observe(
                        X=None,
                        filter=hypothesis.filter,
                        link_idx=hypothesis.active_link_id,
                    )
                    logger.trace(
                        (
                            f"Time {t}: Hypothesis {hypothesis.id} link info sensor "
                            f"update on link {hypothesis.active_link_id}"
                        )
                    )

    def _update_future_node_scores(
        self,
        t: int,
        U: np.ndarray,
        Q_const: np.ndarray,
        dt: float,
    ) -> None:
        active_hypothesis_list = [h for h in self.hypothesis_list if h.is_active]
        active_link_matched_hypothesis_list = [
            h for h in active_hypothesis_list if h.active_link_id is not None and h.senior2junior is not None
        ]
        if len(active_link_matched_hypothesis_list) == 0:
            return
        for hypothesis in active_link_matched_hypothesis_list:
            pred = hypothesis.filter.get_pred(t)
            X, P = pred.X.copy(), pred.P.copy()
            link_id = hypothesis.active_link_id
            pred_result = ex_repetitive_predict(
                X,
                P,
                dt,
                [U] * self.future_pred_steps,
                [Q_const] * self.future_pred_steps,
                self.link_info_sensor,
                link_id,
                self.map,
                self.future_pred_steps,
            )
            future_X, future_P = pred_result.X, pred_result.P

            # Projection to 1d link direction
            link = self.map.get_link_by_id(link_id)
            if hypothesis.senior2junior:
                target_node = link.junior_node
                target_start_point = link.exit_point
                ref_point = link.entrance_point
                link_theta = link.theta
            else:
                target_node = link.senior_node
                target_start_point = link.entrance_point
                ref_point = link.exit_point
                link_theta = link.theta + np.pi
                link_theta = normalize_angle(link_theta)

            g_x, g_y = future_X[0, 0], future_X[1, 0]
            Rot = np.array([[np.cos(link_theta), np.sin(link_theta)], [-np.sin(link_theta), np.cos(link_theta)]])
            s, _ = Rot @ np.array([g_x - ref_point.x, g_y - ref_point.y])
            sl_cov = Rot @ future_P[:2, :2] @ Rot.T
            s_std = np.sqrt(sl_cov[0, 0])

            target_start_sl = Rot @ np.array(
                [
                    [target_start_point.x - ref_point.x],
                    [target_start_point.y - ref_point.y],
                ]
            )
            target_start_s = target_start_sl[0, 0]
            target_start_dist = target_start_s - s
            target_end_dist = target_start_dist + self.map.link_width

            # Consider cyclist length: 1.9 m
            target_start_dist -= 1.9 / 2
            target_end_dist += 1.9 / 2
            pdf = norm(loc=0, scale=s_std)
            integral = pdf.cdf(target_end_dist) - pdf.cdf(target_start_dist)
            score = integral * hypothesis.weight

            if hypothesis.node_exsistence_score is None:
                hypothesis.node_exsistence_score = NodeExsistenceScore(
                    node_id=target_node.id,
                    scores=[score],
                    step_offset=t + self.future_pred_steps,
                )
            else:
                hypothesis.node_exsistence_score.scores.append(score)

    def _predict_next_step(
        self,
        t: int,
        end_t: int,
        U: np.ndarray,
        Q_const: np.ndarray,
        dt: float,
    ) -> tuple[list[Hypothesis], bool]:
        active_hypothesis_list = [h for h in self.hypothesis_list if h.is_active]
        if len(active_hypothesis_list) == 0:
            return [], False
        if t >= end_t - 1:
            return active_hypothesis_list, True
        for hypothesis in active_hypothesis_list:
            Q = Q_const
            hypothesis.filter.predict(dt, U, Q)

            if hypothesis.active_link_id is None:
                mean = hypothesis.filter.X[:2, 0]
                cov = hypothesis.filter.P[:2, :2]
                in_link_id = self.map.pred_in_only_link(mean, cov)
                if in_link_id is not None:
                    hypothesis.active_link_id = in_link_id

            if hypothesis.active_link_id is not None and hypothesis.senior2junior is None:
                theta_95_ci = np.sqrt(hypothesis.filter.P[2, 2] * chi2_95_1d)
                if theta_95_ci < np.deg2rad(30):
                    pred_theta = hypothesis.filter.X[2, 0]
                    link_theta = self.map.get_link_by_id(hypothesis.active_link_id).theta
                    inner_product = np.cos(pred_theta) * np.cos(link_theta) + np.sin(pred_theta) * np.sin(link_theta)
                    if inner_product >= 0:
                        hypothesis.senior2junior = True
                    else:
                        hypothesis.senior2junior = False
        return active_hypothesis_list, False

    def _branch_hypotheses_at_gateways(
        self,
        active_hypothesis_list: list[Hypothesis],
        t: int,
    ) -> None:
        for hypothesis in active_hypothesis_list:
            intersected_node_id = self.map.pred_in_only_node(hypothesis.filter.X[:2, 0], hypothesis.filter.P[:2, :2])
            if intersected_node_id is not None:
                main_filter = hypothesis.filter
                mean_now = main_filter.get_pred(t + 1).X[:2, 0]
                mean_prev = main_filter.get_pred(t).X[:2, 0]
                node = self.map.get_node_by_id(intersected_node_id)
                node_x, node_y = node.center.x, node.center.y

                dist_now = np.sqrt((mean_now[0] - node_x) ** 2 + (mean_now[1] - node_y) ** 2)
                dist_prev = np.sqrt((mean_prev[0] - node_x) ** 2 + (mean_prev[1] - node_y) ** 2)
                flag = dist_prev <= dist_now

                if not flag:
                    continue

                new_filter_list: list[Hypothesis] = []
                # Skip if the node has already generated a hypothesis from this senior hypothesis
                if node.id == hypothesis.generated_node_id:
                    continue
                new_hypothesys_num = len(
                    [gateway for gateway in node.gateways if gateway.link_id != hypothesis.active_link_id]
                )
                for gateway in node.gateways:
                    if gateway.link_id == hypothesis.active_link_id:
                        continue

                    new_filter = KalmanFilter(step_offset=t + 1)
                    new_filter.X = hypothesis.filter.X.copy()
                    new_filter.P = hypothesis.filter.P.copy()

                    # Adjust angle according to the gateway direction
                    target_link = self.map.get_link_by_id(gateway.link_id)
                    theta = target_link.theta
                    if not gateway.is_entrance:
                        theta += np.pi
                        theta = normalize_angle(theta)
                    diff_theta = theta - new_filter.X[2, 0]
                    new_filter.X[2, 0] = theta

                    Jacobian = np.eye(4)
                    Jacobian[:2, :2] = np.array(
                        [
                            [np.cos(diff_theta), -np.sin(diff_theta)],
                            [np.sin(diff_theta), np.cos(diff_theta)],
                        ]
                    )
                    _P = Jacobian @ new_filter.P @ Jacobian.T
                    new_filter.P = _P
                    new_filter.X_init = new_filter.X.copy()
                    new_filter.P_init = new_filter.P.copy()
                    new_filter_list.append(new_filter)

                    new_hypothesis = Hypothesis(
                        id=self.id_incrementer.get_next_id(),
                        senior_hypothesis_id=hypothesis.id,
                        junior_hypothesis_ids=[],
                        filter=new_filter,
                        active_link_id=gateway.link_id,
                        generated_node_id=node.id,
                        is_active=True,
                        generation=hypothesis.generation + 1,
                        senior2junior=gateway.is_entrance,
                        weight=hypothesis.weight / new_hypothesys_num,
                    )
                    self.hypothesis_list.append(new_hypothesis)
                    new_hypothesis.weight_history.append((t + 1, new_hypothesis.weight))
                    hypothesis.junior_hypothesis_ids.append(new_hypothesis.id)
                    logger.trace(
                        (
                            f"Time {t}: Hypothesis {hypothesis.id} branched to {new_hypothesis.id} "
                            f"at node {node.id} towards link {gateway.link_id}"
                        )
                    )

                hypothesis.weight = 0.0
                hypothesis.weight_history.append((t + 1, hypothesis.weight))
                hypothesis.is_active = False

    def _deactivate_hypotheses_outside_links(self, active_hypothesis_list: list[Hypothesis], t: int) -> None:
        for hypothesis in active_hypothesis_list:
            if hypothesis.active_link_id is None:
                continue
            pred = hypothesis.filter.get_pred(t + 1)
            mean, cov = pred.X[:2, 0], pred.P[:2, :2]
            is_in_active_link = self.map.check_ellipse_link_intersection(
                link_id=hypothesis.active_link_id, mean=mean, cov=cov
            )
            link = self.map.get_link_by_id(hypothesis.active_link_id)
            junior_node = link.junior_node
            senior_node = link.senior_node
            if not is_in_active_link:
                if not senior_node.corner.check_ellipse_intersection(
                    mean, cov
                ) and not junior_node.corner.check_ellipse_intersection(mean, cov):
                    hypothesis.weight = 0.0
                    hypothesis.weight_history.append((t + 1, hypothesis.weight))
                    hypothesis.is_active = False
                    logger.trace(
                        (
                            f"Time {t}: Hypothesis {hypothesis.id} deactivated: left link {link.id} "
                            "with unknown direction"
                        )
                    )

    def localization(
        self,
        provided_filters: list[KalmanFilter],
        cyclist_state_series: TimeseriesTPState,
    ) -> list[Hypothesis]:
        """Fuse multiple observer vehicles' relative estimates into a single track.

        Args:
            provided_filters: List of Kalman filters from observer vehicles
            cyclist_state_series: Ground truth trajectory of the cyclist

        Returns:
            List of Hypothesis representing the multiple hypotheses of the cyclist's state
        """
        if not provided_filters:
            raise ValueError("provided_filters must contain at least one observer")

        U = np.array([[0], [0]])
        Q_const = np.array([[self.tp_const_sigma_ww**2, 0], [0, self.tp_const_sigma_aa**2]])
        H = np.eye(4)
        dt = self.time_step
        self.hypothesis_list = []
        self.id_incrementer = IDIncrementer()

        candidate_offsets = [f.step_offset for f in provided_filters]
        if not candidate_offsets:
            raise ValueError("At least one observer must have a valid first observation index")
        start_t = min(candidate_offsets)
        end_t = len(cyclist_state_series)

        for t in range(start_t, end_t):
            infos = self._collect_provided_information(provided_filters, t)

            # === Start update at time t ===

            # Initialize first hypothesis if none exists
            self._initialize_first_hypothesis(t, infos)

            # Update existing hypotheses with measurements at time t
            if infos:
                updated_hypothesis_ids = self._update_hypotheses_with_measurements(infos, H, t)
                if updated_hypothesis_ids:
                    self._deactivate_unupdated_hypotheses(updated_hypothesis_ids, t)

            # Link info observation for all active hypotheses at time t
            self._apply_link_info_observation(t)

            # === End update at time t ===

            # === Start future prediction for intersection existance at time t + future_step ===
            self._update_future_node_scores(t, U, Q_const, dt)
            # === End future prediction for intersection existance at time t + future_step ===

            # === Start predict at next time step t + 1 ===

            # Predict all active hypotheses to next time step
            active_hypothesis_list, should_break = self._predict_next_step(t, end_t, U, Q_const, dt)
            if should_break:
                break
            if len(active_hypothesis_list) == 0:
                continue

            # Generate new hypotheses at gateways
            self._branch_hypotheses_at_gateways(active_hypothesis_list, t)

            # Deactivate hypotheses that have left their links
            self._deactivate_hypotheses_outside_links(active_hypothesis_list, t)

            # === End predict at next time step t + 1 ===

        return self.hypothesis_list
