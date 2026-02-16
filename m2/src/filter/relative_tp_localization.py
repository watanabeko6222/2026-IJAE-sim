"""Relative localization using radar observations."""

import numpy as np

from ..simulator.moving_generator import MovingSimulator, TimeseriesTPState, TP_State
from ..utils import normalize_angle
from .coordinate_transformation import transform_G2L
from .kalman_filter import KalmanFilter


class RelativeTPFilter:
    def __init__(
        self,
        time_step: float,
        radar_range_noise: float,
        radar_range_bias: float,
        radar_azimuth_noise: float,
        radar_vel_noise: float,
        radar_fov_deg: float,
        radar_range_limit: float,
        cyclist_sigma_ww: float,
        cyclist_sigma_aa: float,
    ):
        self.time_step = time_step
        self.radar_range_noise = radar_range_noise
        self.radar_range_bias = radar_range_bias
        self.radar_azimuth_noise = radar_azimuth_noise
        self.radar_vel_noise = radar_vel_noise
        self.radar_fov_deg = radar_fov_deg
        self.radar_range_limit = radar_range_limit
        self.cyclist_sigma_ww = cyclist_sigma_ww
        self.cyclist_sigma_aa = cyclist_sigma_aa

    def is_observable(
        self,
        ego_state: TP_State | None,
        cyclist_state: TP_State | None,
    ) -> bool:
        """Check if cyclist is observable from ego vehicle.

        Args:
            ego_state: Ego vehicle state
            cyclist_state: Cyclist state

        Returns:
            True if cyclist is observable
        """
        if ego_state is None or cyclist_state is None:
            return False

        if ego_state.can_do_observe is False or cyclist_state.can_be_observed is False:
            return False

        dx = cyclist_state.x - ego_state.x
        dy = cyclist_state.y - ego_state.y
        dist = np.sqrt(dx**2 + dy**2)

        if dist > self.radar_range_limit:
            return False

        ego2cyclist = np.arctan2(dy, dx)
        ego_theta = ego_state.theta

        fov_theta1 = normalize_angle(ego_theta - np.deg2rad(self.radar_fov_deg) / 2)
        fov_theta2 = normalize_angle(ego_theta + np.deg2rad(self.radar_fov_deg) / 2)

        if fov_theta1 < fov_theta2:
            return fov_theta1 < ego2cyclist < fov_theta2
        else:
            return fov_theta1 < ego2cyclist or ego2cyclist < fov_theta2

    def check_direction(
        self,
        ego_state: TP_State,
        cyclist_state: TP_State,
        angle_threshold: float = 1e-8,
    ) -> bool:
        """Check if cyclist is moving in the same direction as ego vehicle.

        Args:
            ego_state: Ego vehicle state
            cyclist_state: Cyclist state
            angle_threshold: Maximum angle difference in degrees to be considered same direction
        Returns:
            True if moving in the same direction
        """
        angle_diff = np.abs(normalize_angle(ego_state.theta - cyclist_state.theta))

        if angle_diff < angle_threshold:
            return "same"
        elif np.abs(angle_diff - np.pi) < angle_threshold:
            return "opposite"
        else:
            return "different"

    def rel_obs_radar(
        self,
        rel_state: TP_State,
    ) -> tuple[float, float, float, float, float]:
        """Simulate radar observation with noise.

        Args:
            rel_state: Relative state

        Returns:
            Observed x, y, velocity, sigma_x, sigma_y
        """
        true_dist = np.sqrt(rel_state.x**2 + rel_state.y**2)
        true_azimuth = np.arctan2(rel_state.y, rel_state.x)
        obs_dist = true_dist + np.random.normal(0, self.radar_range_noise + self.radar_range_bias * true_dist)
        obs_azimuth = true_azimuth + np.random.normal(0, self.radar_azimuth_noise)

        xx = obs_dist * np.cos(obs_azimuth)
        yy = obs_dist * np.sin(obs_azimuth)
        vel = rel_state.vel + np.random.normal(0, self.radar_vel_noise)

        sigma_xx = np.sqrt(
            np.cos(obs_azimuth) ** 2 * (self.radar_range_noise + self.radar_range_bias * true_dist) ** 2
            + obs_dist**2 * np.sin(obs_azimuth) ** 2 * self.radar_azimuth_noise**2
        )
        sigma_yy = np.sqrt(
            np.sin(obs_azimuth) ** 2 * (self.radar_range_noise + self.radar_range_bias * true_dist) ** 2
            + obs_dist**2 * np.cos(obs_azimuth) ** 2 * self.radar_azimuth_noise**2
        )

        return xx, yy, vel, sigma_xx, sigma_yy

    def localization(
        self,
        ego_movement: MovingSimulator,
        cyclist_movement: MovingSimulator,
    ) -> tuple[KalmanFilter | None, TimeseriesTPState | None]:
        """Perform relative localization using radar observations.

        Args:
            ego_movement: Ego vehicle movement simulator
            cyclist_movement: Cyclist movement simulator

        Returns:
            Updated filter, relative state list, last observation indices, max iterations
        """
        max_iter = min(len(ego_movement), len(cyclist_movement))
        rel_cyclist_state_series: TimeseriesTPState | None = None
        dt = self.time_step
        U = np.array([[0], [0]])
        Q = np.array([[self.cyclist_sigma_ww**2, 0], [0, self.cyclist_sigma_aa**2]])
        is_first_obs_rel = True
        last_obs_i = None

        rel_cyclist_filter: KalmanFilter | None = None

        for i in range(max_iter):
            cyclist_state = cyclist_movement.get_state(i)
            try:
                ego_state = ego_movement.get_state(i)
            # In case ego movement is shorter than cyclist movement
            except IndexError:
                ego_state = None

            if ego_state is None or cyclist_state is None:
                pass
            else:
                if rel_cyclist_state_series is None:
                    rel_cyclist_state_series = TimeseriesTPState(step_offset=i)
                rel_cyclist_state = transform_G2L(cyclist_state, ego_state)
                rel_cyclist_state_series.add_state(rel_cyclist_state)

            # === start t = i update === #
            if self.is_observable(ego_state, cyclist_state):
                direction = self.check_direction(ego_state, cyclist_state)
                if direction != "different":
                    xx, yy, vel, rel_sigma_xx, rel_sigma_yy = self.rel_obs_radar(
                        rel_cyclist_state,
                    )

                    if direction == "same":
                        vel = -vel

                    rel_Z = np.array([[xx], [yy], [vel]])
                    rel_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                    rel_R = np.array(
                        [
                            [rel_sigma_xx**2, 0, 0],
                            [0, rel_sigma_yy**2, 0],
                            [0, 0, self.radar_vel_noise**2],
                        ]
                    )

                    if is_first_obs_rel:
                        rel_cyclist_filter = KalmanFilter(step_offset=i)
                        if direction == "same":
                            rel_cyclist_filter.allow_negative_velocity = True
                            init_angle = 0
                        elif direction == "opposite":
                            init_angle = -np.pi

                        rel_cyclist_filter.X = np.array([[xx], [yy], [init_angle], [vel]])
                        rel_cyclist_filter.P = np.array(
                            [
                                [rel_sigma_xx**2, 0, 0, 0],
                                [0, rel_sigma_yy**2, 0, 0],
                                [0, 0, (np.pi / 4) ** 2, 0],
                                [0, 0, 0, self.radar_vel_noise**2],
                            ]
                        )
                        rel_cyclist_filter.X_init = np.array([[xx], [yy], [init_angle], [vel]])
                        rel_cyclist_filter.P_init = np.array(
                            [
                                [rel_sigma_xx**2, 0, 0, 0],
                                [0, rel_sigma_yy**2, 0, 0],
                                [0, 0, (np.pi / 4) ** 2, 0],
                                [0, 0, 0, self.radar_vel_noise**2],
                            ]
                        )
                        is_first_obs_rel = False
                    else:
                        rel_cyclist_filter.update(rel_Z, rel_R, rel_H)
                        rel_cyclist_filter.Z_memory.append(rel_Z)
                        rel_cyclist_filter.last_Z = rel_Z
                    last_obs_i = i

            # === end t = i update === #

            # === start t = i+1 predict === #
            if not is_first_obs_rel:
                if i == max_iter - 1:
                    break
                elif i - last_obs_i > 10 and not is_first_obs_rel:
                    break

                rel_cyclist_filter.predict(dt, U, Q)
            # === end t = i+1 predict === #

        return rel_cyclist_filter, rel_cyclist_state_series
