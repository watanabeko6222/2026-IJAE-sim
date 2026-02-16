"""Self-localization using GNSS and wheel velocity sensors."""

import numpy as np

from ..simulator.moving_generator import MovingSimulator
from ..utils import normalize_angle
from .kalman_filter import KalmanFilter
from .sensor import BaseInfo


class SelfLocalizationFilter:
    def __init__(
        self,
        time_step: float,
        sigma_ww: float,
        sigma_aa: float,
    ):
        self.time_step = time_step
        self.sigma_ww = sigma_ww
        self.sigma_aa = sigma_aa

    def localization(
        self,
        ego_movement: MovingSimulator,
        rsu_filters: list[KalmanFilter],
        self_infos: list[BaseInfo],
        first_aggrigate_step: int = 10,
    ) -> KalmanFilter:
        """Perform self-localization using GNSS and wheel velocity sensors.

        Args:
            ego_movement: Ego vehicle movement simulator
            rsu_filters: List of RSU Kalman filters for cooperative localization
            self_infos: List of ego vehicle sensor information
            first_aggrigate_step: Step interval for initial aggregation

        Returns:
            Updated ego filter
        """
        step_offset = ego_movement.step_offset

        dt = self.time_step
        Q = np.array([[self.sigma_ww**2, 0], [0, self.sigma_aa**2]])
        U = np.array([[0], [0]])
        ego_filter: KalmanFilter | None = None

        # Main loop for each time step from step_offset to end
        for i in range(step_offset + first_aggrigate_step + 1, len(ego_movement)):
            # === start t = i update === #
            now_true_state = ego_movement.get_state(i)
            now_true_X = np.array(
                [
                    [now_true_state.x],
                    [now_true_state.y],
                    [now_true_state.theta],
                    [now_true_state.vel],
                ]
            )

            # RSU info update
            for rsu_filter in rsu_filters:
                rsu_pred = rsu_filter.get_pred(i)
                if rsu_pred is not None:
                    if ego_filter is None:
                        ego_filter = KalmanFilter(step_offset=i)
                        ego_filter.X = rsu_pred.X.copy()
                        ego_filter.P = rsu_pred.P.copy()
                        ego_filter.X_init = rsu_pred.X.copy()
                        ego_filter.P_init = rsu_pred.P.copy()

                    else:
                        info_X, info_P = rsu_pred.X, rsu_pred.P
                        pred_theta = ego_filter.X[2, 0]
                        info_theta = info_X[2, 0]
                        diff_theta = normalize_angle(info_theta - pred_theta)
                        info_X[2, 0] = pred_theta + diff_theta
                        H = np.eye(4)
                        ego_filter.update_CI(info_X, info_P, H)

            # sensor update
            if ego_filter is not None:
                for self_info in self_infos:
                    if i % int(10 / self_info.fps) == 0:
                        X = now_true_X.copy()
                        self_info.observe(X, filter=ego_filter, link_idx=now_true_state.in_link_id)
            # === end t = i update === #

            # === start t = i+1 predict === #

            # No prediction at the last time step
            if ego_filter is None:
                continue
            if i == len(ego_movement) - 1:
                continue
            else:
                yaw_rate = now_true_state.yaw_rate
                accel = now_true_state.accel
                U = np.array(
                    [
                        [yaw_rate + np.random.normal(0, self.sigma_ww)],
                        [accel + np.random.normal(0, self.sigma_aa)],
                    ]
                )
                ego_filter.predict(dt, U, Q)
            # === end t = i+1 predict === #

        return ego_filter
