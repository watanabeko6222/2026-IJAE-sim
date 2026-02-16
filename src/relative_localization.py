"""
Relative localization using radar observations.
"""
import numpy as np
from .kalman_filter import KalmanFilter
from .moving_generator import MovingSimulatorLine, TP_State


def is_observable(ego_state: TP_State, cyclist_state: TP_State, fov: float = 90, range_limit: float = 50) -> bool:
    """
    Check if cyclist is observable from ego vehicle.
    
    Args:
        ego_state: Ego vehicle state
        cyclist_state: Cyclist state
        fov: Field of view in degrees
        range_limit: Maximum observation range in meters
    
    Returns:
        True if cyclist is observable
    """
    dx = cyclist_state.x - ego_state.x
    dy = cyclist_state.y - ego_state.y
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist > range_limit:
        return False

    ego2cyclist = np.arctan2(dy, dx) % (2 * np.pi)
    ego_theta = ego_state.theta % (2 * np.pi)

    fov_theta1 = (ego_theta - np.deg2rad(fov) / 2) % (2 * np.pi)
    fov_theta2 = (ego_theta + np.deg2rad(fov) / 2) % (2 * np.pi)

    if fov_theta1 < fov_theta2:
        return fov_theta1 < ego2cyclist < fov_theta2
    else:
        return fov_theta1 < ego2cyclist or ego2cyclist < fov_theta2


def rel_obs_radar(
    rel_state: TP_State, 
    range_noise: float, 
    range_bias: float,
    azimuth_noise: float, 
    rel_sigma_vel: float
) -> tuple[float, float, float, float, float]:
    """
    Simulate radar observation with noise.
    
    Args:
        rel_state: Relative state
        range_noise: Range measurement noise std
        range_bias: Range measurement bias
        azimuth_noise: Azimuth measurement noise std
        rel_sigma_vel: Relative velocity measurement noise std
    
    Returns:
        Observed x, y, velocity and their standard deviations
    """
    true_dist = np.sqrt(rel_state.x**2 + rel_state.y**2)
    true_azimuth = np.arctan2(rel_state.y, rel_state.x)
    obs_dist = true_dist + np.random.normal(0, range_noise + range_bias * true_dist)
    obs_azimuth = true_azimuth + np.random.normal(0, azimuth_noise)
    
    xx = obs_dist * np.cos(obs_azimuth)
    yy = obs_dist * np.sin(obs_azimuth)
    vel = rel_state.vel + np.random.normal(0, rel_sigma_vel)
    
    sigma_xx = np.sqrt(np.cos(obs_azimuth)**2 * (range_noise + range_bias * true_dist)**2 + 
                      obs_dist**2 * np.sin(obs_azimuth)**2 * azimuth_noise**2)
    sigma_yy = np.sqrt(np.sin(obs_azimuth)**2 * (range_noise + range_bias * true_dist)**2 + 
                      obs_dist**2 * np.cos(obs_azimuth)**2 * azimuth_noise**2)

    return xx, yy, vel, sigma_xx, sigma_yy


def transform_G2L(G_tp: TP_State, G_ego: TP_State) -> TP_State:
    """Transform from global coordinates to local coordinates."""
    return TP_State(
        x = (G_tp.x - G_ego.x) * np.cos(G_ego.theta) + (G_tp.y - G_ego.y) * np.sin(G_ego.theta),
        y = (G_tp.y - G_ego.y) * np.cos(G_ego.theta) - (G_tp.x - G_ego.x) * np.sin(G_ego.theta),
        theta = G_tp.theta - G_ego.theta,
        vel = np.sqrt(G_tp.vel**2 + G_ego.vel**2 - 2 * G_tp.vel * G_ego.vel * np.cos(G_tp.theta - G_ego.theta)),
    )


def relative_localization(
    ego_movement: MovingSimulatorLine,
    cyclist_movement: MovingSimulatorLine,
    rel_cyclist_filter: KalmanFilter,
    time_step: float,
    range_noise: float,
    range_bias: float,
    azimuth_noise: float,
    rel_sigma_vel: float,
    sigma_ww: float,
    sigma_aa: float,
) -> tuple[KalmanFilter, list[TP_State], int, int, int]:
    """
    Perform relative localization using radar observations.
    
    Args:
        ego_movement: Ego vehicle movement simulator
        cyclist_movement: Cyclist movement simulator
        rel_cyclist_filter: Relative cyclist Kalman filter
        time_step: Time step for prediction
        range_noise: Range measurement noise std
        range_bias: Range measurement bias
        azimuth_noise: Azimuth measurement noise std
        rel_sigma_vel: Relative velocity measurement noise std
        sigma_ww: Angular velocity noise std
        sigma_aa: Acceleration noise std
    
    Returns:
        Updated filter, relative state list, first/last observation indices, max iterations
    """
    max_iter = min(len(ego_movement), len(cyclist_movement))
    rel_cyclist_state_list = []
    first_obs_idx = None
    last_obs_idx = None
    is_first_obs_rel = True

    for i in range(max_iter):
        ego_state = ego_movement.get_state(i)
        cyclist_state = cyclist_movement.state_list[i]
        rel_cyclist_state = transform_G2L(cyclist_state, ego_state)
        rel_cyclist_state_list.append(rel_cyclist_state)

        # === start t = i update === #
        if is_observable(ego_state, cyclist_state):
            xx, yy, vel, rel_sigma_xx, rel_sigma_yy = rel_obs_radar(
                rel_cyclist_state, range_noise, range_bias, azimuth_noise, rel_sigma_vel
            )
            rel_Z = np.array([[xx], [yy], [vel]])
            rel_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            rel_R = np.array(
                [
                    [rel_sigma_xx**2, 0, 0],
                    [0, rel_sigma_yy**2, 0],
                    [0, 0, rel_sigma_vel**2],
                ]
            )

            if is_first_obs_rel:
                rel_cyclist_filter.X = np.array([[xx], [yy], [-np.pi], [vel]])
                rel_cyclist_filter.P = np.array(
                    [
                        [rel_sigma_xx**2, 0, 0, 0],
                        [0, rel_sigma_yy**2, 0, 0],
                        [0, 0, (np.pi/4)**2, 0],
                        [0, 0, 0, rel_sigma_vel**2],
                    ]
                )
                rel_cyclist_filter.X_init = np.array([[xx], [yy], [-np.pi], [vel]])
                rel_cyclist_filter.P_init = np.array(
                    [
                        [rel_sigma_xx**2, 0, 0, 0],
                        [0, rel_sigma_yy**2, 0, 0],
                        [0, 0, (np.pi/4)**2, 0],
                        [0, 0, 0, rel_sigma_vel**2],
                    ]
                )
                is_first_obs_rel = False
                first_obs_idx = i
            else:
                rel_cyclist_filter.update(rel_Z, rel_R, rel_H)
                rel_cyclist_filter.Z_memory.append(rel_Z)
                rel_cyclist_filter.last_Z = rel_Z
            last_obs_idx = i

        # === end t = i update === #

        # === start t = i+1 predict === #
        if is_first_obs_rel:
            continue
        elif i == len(ego_movement) - 1:
            continue
        else:
            dt = time_step
            U = np.array([[0], [0]])
            Q = np.array([[sigma_ww**2, 0], [0, sigma_aa**2]])
            rel_cyclist_filter.predict(dt, U, Q)
        # === end t = i+1 predict === #

    return (
        rel_cyclist_filter,
        rel_cyclist_state_list,
        first_obs_idx,
        last_obs_idx,
        max_iter
    )
