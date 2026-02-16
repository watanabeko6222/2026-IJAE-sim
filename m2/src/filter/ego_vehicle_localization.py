"""Self-localization using GNSS and wheel velocity sensors."""

import numpy as np

from ..simulator.moving_generator import MovingSimulator
from ..utils import normalize_angle
from .kalman_filter import KalmanFilter
from .sensor import GNSS, BaseInfo, WheelVel


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
        first_aggrigate_step: int = 50,
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

        # Initial aggregation of observations to set initial state
        gnss_info = None
        wheel_vel_info = None
        for info in self_infos:
            if isinstance(info, GNSS):
                gnss_info = info
            elif isinstance(info, WheelVel):
                wheel_vel_info = info
        first_true_state = ego_movement.get_state(step_offset)
        first_obs = np.array(
            [
                [first_true_state.x + np.random.normal(0, gnss_info.pos_std)],
                [first_true_state.y + np.random.normal(0, gnss_info.pos_std)],
                [0],
                [first_true_state.vel + np.random.normal(0, wheel_vel_info.vel_std)],
            ]
        )

        second_true_state = ego_movement.get_state(step_offset + first_aggrigate_step)
        second_obs = np.array(
            [
                [second_true_state.x + np.random.normal(0, gnss_info.pos_std)],
                [second_true_state.y + np.random.normal(0, gnss_info.pos_std)],
                [0],
                [second_true_state.vel + np.random.normal(0, wheel_vel_info.vel_std)],
            ]
        )

        aggrigated_theta = np.arctan2(second_obs[1, 0] - first_obs[1, 0], second_obs[0, 0] - first_obs[0, 0])
        X_init = first_obs.copy()
        X_init[2, 0] = aggrigated_theta
        P_init = np.array(
            [
                [gnss_info.pos_std**2, 0, 0, 0],
                [0, gnss_info.pos_std**2, 0, 0],
                [0, 0, wheel_vel_info.vel_std**2, 0],
                [0, 0, 0, wheel_vel_info.vel_std**2],
            ]
        )

        ego_filter = KalmanFilter(step_offset=step_offset)
        ego_filter.X_init = X_init
        ego_filter.P_init = P_init
        ego_filter.X = X_init
        ego_filter.P = P_init
        # initial aggregation done
        yaw_rate = first_true_state.yaw_rate
        accel = first_true_state.accel
        U = np.array(
            [
                [yaw_rate + np.random.normal(0, self.sigma_ww)],
                [accel + np.random.normal(0, self.sigma_aa)],
            ]
        )
        ego_filter.predict(dt, U, Q)

        # Main loop for each time step from step_offset to end
        for i in range(step_offset + 1, len(ego_movement)):
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

            # sensor update
            for self_info in self_infos:
                if i % int(10 / self_info.fps) == 0:
                    X = now_true_X.copy()
                    self_info.observe(X, filter=ego_filter)

            # RSU info update
            for rsu_filter in rsu_filters:
                rsu_pred = rsu_filter.get_pred(i)
                if rsu_pred is not None:
                    info_X, info_P = rsu_pred.X, rsu_pred.P
                    pred_theta = ego_filter.X[2, 0]
                    info_theta = info_X[2, 0]
                    diff_theta = normalize_angle(info_theta - pred_theta)
                    info_X[2, 0] = pred_theta + diff_theta
                    H = np.eye(4)
                    ego_filter.update_CI(info_X, info_P, H)

            # === end t = i update === #

            # === start t = i+1 predict === #

            # No prediction at the last time step
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
