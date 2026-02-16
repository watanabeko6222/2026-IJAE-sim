"""
Self-localization using GNSS and wheel velocity sensors.
"""
import numpy as np
from .kalman_filter import KalmanFilter
from .moving_generator import MovingSimulatorLine
from .sensor import GNSS, WheelVel


def self_localization(
    ego_movement: MovingSimulatorLine,
    infos: list[GNSS | WheelVel],
    ego_filter: KalmanFilter,
    time_step: float,
    sigma_ww: float,
    sigma_aa: float
) -> KalmanFilter:
    """
    Perform self-localization using GNSS and wheel velocity sensors.
    
    Args:
        ego_movement: Ego vehicle movement simulator
        infos: List of sensors (GNSS, WheelVel)
        ego_filter: Ego vehicle Kalman filter
        time_step: Time step for prediction
        sigma_ww: Angular velocity noise standard deviation
        sigma_aa: Acceleration noise standard deviation
    
    Returns:
        Updated ego filter
    """
    is_first_obs = True
    
    for i in range(len(ego_movement)):
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
        if is_first_obs:
            P = np.array(
                [
                    [4.25**2, 0, 0, 0],
                    [0, 4.25**2, 0, 0],
                    [0, 0, 0.28**2, 0],
                    [0, 0, 0, np.pi**2],
                ]
            )
            X = np.array(
                [
                    [now_true_state.x + np.random.normal(0, 4.25)],
                    [now_true_state.y + np.random.normal(0, 4.25)],
                    [0],
                    [now_true_state.vel + np.random.normal(0, 0.28)],
                ]
            )
            ego_filter.X = X
            ego_filter.P = P
            ego_filter.X_init = X
            ego_filter.P_init = P
            is_first_obs = False
        else:
            for info in infos:
                if i % int(10 / info.fps) == 0:
                    X = now_true_X.copy()
                    info.observe(X, filter=ego_filter)
        # === end t = i update === #

        # === start t = i+1 predict === #
        if i == len(ego_movement) - 1:
            continue
        else:
            dt = time_step
            yaw_rate = now_true_state.yaw_rate
            accel = now_true_state.accel
            U = np.array(
                [
                    [yaw_rate + np.random.normal(0, sigma_ww)],
                    [accel + np.random.normal(0, sigma_aa)],
                ]
            )
            Q = np.array([[sigma_ww**2, 0], [0, sigma_aa**2]])
            ego_filter.predict(dt, U, Q)
        # === end t = i+1 predict === #

    return ego_filter
