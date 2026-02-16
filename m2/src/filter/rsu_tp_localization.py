"""Self-localization using GNSS and wheel velocity sensors."""

import numpy as np

from ..simulator.map_constructor import BaseMap
from ..simulator.moving_generator import MovingSimulator
from .kalman_filter import KalmanFilter
from .sensor import RoadSideSensor


class RSUFilter:
    def __init__(self, map: BaseMap, time_step: float):
        self.map = map
        self.time_step = time_step

    def localization(
        self,
        tp_movement: MovingSimulator,
        sigma_ww: float,
        sigma_aa: float,
        aggrigate_step: int = 10,
    ) -> list[KalmanFilter]:
        """Perform self-localization using GNSS and wheel velocity sensors.

        Args:
            tp_movement: Ego vehicle movement simulator
            sigma_ww: Angular velocity noise standard deviation
            sigma_aa: Acceleration noise standard deviation
            aggrigate_step: Number of steps to aggrigate for initial observation

        Returns:
            Updated ego filter
        """
        step_offset = tp_movement.step_offset
        first_obs = None
        first_obs_t = None
        second_obs = None
        last_obs_t = None
        tp_filters: list[KalmanFilter] = []

        dt = self.time_step
        U = np.array([[0], [0]])
        Q = np.array([[sigma_ww**2, 0], [0, sigma_aa**2]])

        rsu_infos: list[RoadSideSensor] = []
        for node in self.map.nodes:
            if node.rsu is not None:
                rsu_sensor = RoadSideSensor(pos_std=0.1, fps=10, range=10, x=node.center.x, y=node.center.y)
                rsu_infos.append(rsu_sensor)

        if len(rsu_infos) == 0:
            return tp_filters

        # Main loop for each time step from step_offset to end
        for t in range(step_offset, len(tp_movement)):
            # === start t = i update === #
            now_true_state = tp_movement.get_state(t)
            if now_true_state is None:
                continue
            now_true_X = np.array(
                [
                    [now_true_state.x],
                    [now_true_state.y],
                    [now_true_state.theta],
                    [now_true_state.vel],
                ]
            )

            # sensor update
            for rsu_info in rsu_infos:
                if t % int(10 / rsu_info.fps) == 0:
                    if rsu_info.is_visible(now_true_state.x, now_true_state.y):
                        last_obs_t = t
                        if first_obs is None and second_obs is None:
                            # Initialize filter with first observation
                            first_obs_t = t
                            noise = np.random.randn(2, 1) * rsu_info.pos_std
                            first_obs_xy = (
                                np.array(
                                    [
                                        [now_true_state.x],
                                        [now_true_state.y],
                                    ]
                                )
                                + noise
                            )
                            first_obs = np.vstack([first_obs_xy, [[0.0]], [[0.0]]])
                        elif first_obs is not None and second_obs is None:
                            if t - first_obs_t >= aggrigate_step:
                                noise = np.random.randn(2, 1) * rsu_info.pos_std
                                second_obs_xy = (
                                    np.array(
                                        [
                                            [now_true_state.x],
                                            [now_true_state.y],
                                        ]
                                    )
                                    + noise
                                )
                                second_obs = np.vstack([second_obs_xy, [[0.0]], [[0.0]]])
                                dx = second_obs[0, 0] - first_obs_xy[0, 0]
                                dy = second_obs[1, 0] - first_obs_xy[1, 0]
                                distance = np.hypot(dx, dy)
                                aggrigated_theta = np.arctan2(dy, dx)
                                aggrigated_vel = distance / ((t - first_obs_t) * dt)
                                X_init = second_obs.copy()
                                X_init[2, 0] = aggrigated_theta
                                X_init[3, 0] = aggrigated_vel
                                P_init = np.array(
                                    [
                                        [rsu_info.pos_std**2, 0, 0, 0],
                                        [0, rsu_info.pos_std**2, 0, 0],
                                        [0, 0, 2 * (rsu_info.pos_std / distance) ** 2, 0],
                                        [
                                            0,
                                            0,
                                            0,
                                            2 * (rsu_info.pos_std / ((t - first_obs_t) * dt)) ** 2,
                                        ],
                                    ]
                                )
                                tp_filter = KalmanFilter(step_offset=t)
                                tp_filter.X_init = X_init
                                tp_filter.P_init = P_init
                                tp_filter.X = X_init
                                tp_filter.P = P_init

                        else:
                            X = now_true_X.copy()
                            rsu_info.observe(X, filter=tp_filter)

            # === end t = i update === #

            # === start t = i+1 predict === #
            if not (first_obs is not None and second_obs is not None):
                continue
            # No prediction at the last time step
            if t == len(tp_movement) - 1:
                break
            # Skip prediction if no observation for a long time
            elif t - last_obs_t > 10:
                tp_filters.append(tp_filter)
                first_obs = None
                first_obs_t = None
                second_obs = None
                last_obs_t = None
                tp_filter = None
            else:
                tp_filter.predict(dt, U, Q)
            # === end t = i+1 predict === #

        return tp_filters
