import numpy as np
from .moving_generator import TP_State, MovingSimulatorLine
from .kalman_filter import KalmanFilter
from .relative_localization import is_observable


def transform_L2G(G_ego: TP_State, L_tp: TP_State) -> TP_State:
    return TP_State(
        x=G_ego.x + L_tp.x * np.cos(G_ego.theta) - L_tp.y * np.sin(G_ego.theta),
        y=G_ego.y + L_tp.x * np.sin(G_ego.theta) + L_tp.y * np.cos(G_ego.theta),
        theta=G_ego.theta + L_tp.theta,
        vel=np.sqrt(G_ego.vel**2 + L_tp.vel**2 + 2 * G_ego.vel * L_tp.vel * np.cos(L_tp.theta)),
    )

def jacoboan_transform_L2G(G_ego: TP_State, L_tp: TP_State) -> np.ndarray:
    G_ego_element_44 = (G_ego.vel + L_tp.vel * np.cos(L_tp.theta)) \
                    / np.sqrt(G_ego.vel**2 + L_tp.vel**2 + 2 * G_ego.vel * L_tp.vel * np.cos(L_tp.theta))
    jacobian_G_ego = np.array(
        [
            [1, 0, -L_tp.x * np.sin(G_ego.theta) - L_tp.y * np.cos(G_ego.theta), 0],
            [0, 1, L_tp.x * np.cos(G_ego.theta)- L_tp.y * np.sin(G_ego.theta), 0],
            [0, 0, 1, 0],
            [0, 0, 0, G_ego_element_44],
        ],
    )

    L_tp_element_34 = -L_tp.vel * G_ego.vel * np.sin(L_tp.theta) \
                    / np.sqrt(G_ego.vel**2 + L_tp.vel**2 + 2 * G_ego.vel * L_tp.vel * np.cos(L_tp.theta))
    L_tp_element_44 = (L_tp.vel + G_ego.vel * np.cos(L_tp.theta)) \
                    / np.sqrt(G_ego.vel**2 + L_tp.vel**2 + 2 * G_ego.vel * L_tp.vel * np.cos(L_tp.theta))
    jacobian_L_tp = np.array(
        [
            [np.cos(G_ego.theta), -np.sin(G_ego.theta), 0, 0],
            [np.sin(G_ego.theta), np.cos(G_ego.theta), 0, 0],
            [0, 0, 1, L_tp_element_34],
            [0, 0, 0, L_tp_element_44],
        ]
    )

    return jacobian_G_ego, jacobian_L_tp


def integrated_localization(
    ego_movement: MovingSimulatorLine,
    ego_filter: KalmanFilter,
    cyclist_movement: MovingSimulatorLine,
    rel_cyclist_filter: KalmanFilter,
    abs_cyclist_filter: KalmanFilter,
    time_step: float,
    sigma_ww: float,
    sigma_aa: float,
) -> KalmanFilter:
    max_iter = min(len(ego_movement), len(cyclist_movement))
    no_more_obserbable = False
    is_first_obs_rel = True
    first_obs_idx = 0

    for i in range(max_iter):
        ego_state = ego_movement.get_state(i)
        ego_pred, Sigma_G_ego = ego_filter.get_state(i)
        ego_pred = TP_State(
            x=ego_pred[0, 0], y=ego_pred[1, 0], theta=ego_pred[2, 0], vel=ego_pred[3, 0]
        )
        cyclist_state = cyclist_movement.state_list[i]
        
        if is_first_obs_rel is False:
            cyclist_pred, Sigma_L_cyclist = rel_cyclist_filter.get_state(i - first_obs_idx)
            cyclist_pred = TP_State(
                x=cyclist_pred[0, 0], y=cyclist_pred[1, 0], theta=cyclist_pred[2, 0], vel=cyclist_pred[3, 0]
            )

        # === start t = i update === #
        if is_observable(ego_state, cyclist_state):
            if is_first_obs_rel:
                is_first_obs_rel = False
                first_obs_idx = i
                
                cyclist_pred, Sigma_L_cyclist = rel_cyclist_filter.get_state(0)
                cyclist_pred = TP_State(
                    x=cyclist_pred[0, 0], y=cyclist_pred[1, 0], theta=cyclist_pred[2, 0], vel=cyclist_pred[3, 0]
                )
                
                abs_cyclist_obs = transform_L2G(G_ego=ego_pred, L_tp=cyclist_pred)
                jacobian_G_ego, jacobian_L_tp = jacoboan_transform_L2G(
                    G_ego=ego_pred, L_tp=cyclist_pred
                )
                Sigma_G_tp = (
                    jacobian_G_ego @ Sigma_G_ego @ jacobian_G_ego.T
                    + jacobian_L_tp @ Sigma_L_cyclist @ jacobian_L_tp.T
                )
                abs_X = np.array(
                    [
                        [abs_cyclist_obs.x],
                        [abs_cyclist_obs.y],
                        [abs_cyclist_obs.theta],
                        [abs_cyclist_obs.vel],
                    ]
                )
                abs_cyclist_filter.X = abs_X
                abs_cyclist_filter.P = Sigma_G_tp
                abs_cyclist_filter.X_init = abs_X
                abs_cyclist_filter.P_init = Sigma_G_tp
        
        elif is_first_obs_rel is False:
            no_more_obserbable = True
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
            abs_cyclist_obs = transform_L2G(G_ego=ego_pred, L_tp=cyclist_pred)
            
            jacobian_G_ego, jacobian_L_tp = jacoboan_transform_L2G(
                G_ego=ego_pred, L_tp=cyclist_pred
            )
            Sigma_G_tp = (
                jacobian_G_ego @ Sigma_G_ego @ jacobian_G_ego.T
                + jacobian_L_tp @ Sigma_L_cyclist @ jacobian_L_tp.T
            )
            abs_X = np.array(
                [
                    [abs_cyclist_obs.x],
                    [abs_cyclist_obs.y],
                    [abs_cyclist_obs.theta],
                    [abs_cyclist_obs.vel],
                ]
            )
            abs_R = Sigma_G_tp
            
            if no_more_obserbable:
                abs_cyclist_filter.predict(dt, U, Q)
            else:
                abs_cyclist_filter.X = abs_X
                abs_cyclist_filter.P = abs_R
                abs_cyclist_filter.Xp_memory.append(abs_X)
                abs_cyclist_filter.Pp_memory.append(abs_R)
        # === end t = i+1 predict === #

    return abs_cyclist_filter

def integrated_localization_ablation(
    ego_movement: MovingSimulatorLine,
    ego_filter: KalmanFilter,
    cyclist_movement: MovingSimulatorLine,
    rel_cyclist_filter: KalmanFilter,
    abs_cyclist_filter: KalmanFilter,
    time_step: float,
    sigma_ww: float,
    sigma_aa: float,
) -> KalmanFilter:
    max_iter = min(len(ego_movement), len(cyclist_movement))
    no_more_obserbable = False
    is_first_obs_rel = True
    first_obs_idx = 0

    for i in range(max_iter):
        ego_state = ego_movement.get_state(i)
        ego_pred, Sigma_G_ego = ego_filter.get_state(i)
        ego_pred = TP_State(
            x=ego_pred[0, 0], y=ego_pred[1, 0], theta=ego_pred[2, 0], vel=ego_pred[3, 0]
        )
        cyclist_state = cyclist_movement.state_list[i]
        
        if is_first_obs_rel is False:
            cyclist_pred, Sigma_L_cyclist = rel_cyclist_filter.get_state(i - first_obs_idx)
            cyclist_pred = TP_State(
                x=cyclist_pred[0, 0], y=cyclist_pred[1, 0], theta=cyclist_pred[2, 0], vel=cyclist_pred[3, 0]
            )

        # === start t = i update === #
        if is_observable(ego_state, cyclist_state):
            if is_first_obs_rel:
                is_first_obs_rel = False
                first_obs_idx = i
                
                cyclist_pred, Sigma_L_cyclist = rel_cyclist_filter.get_state(0)
                cyclist_pred = TP_State(
                    x=cyclist_pred[0, 0], y=cyclist_pred[1, 0], theta=cyclist_pred[2, 0], vel=cyclist_pred[3, 0]
                )
                
                abs_cyclist_obs = transform_L2G(G_ego=ego_pred, L_tp=cyclist_pred)
                jacobian_G_ego, jacobian_L_tp = jacoboan_transform_L2G(
                    G_ego=ego_pred, L_tp=cyclist_pred
                )
                Sigma_G_tp = (
                    # jacobian_G_ego @ Sigma_G_ego @ jacobian_G_ego.T
                    jacobian_L_tp @ Sigma_L_cyclist @ jacobian_L_tp.T
                )
                abs_X = np.array(
                    [
                        [abs_cyclist_obs.x],
                        [abs_cyclist_obs.y],
                        [abs_cyclist_obs.theta],
                        [abs_cyclist_obs.vel],
                    ]
                )
                abs_cyclist_filter.X = abs_X
                abs_cyclist_filter.P = Sigma_G_tp
                abs_cyclist_filter.X_init = abs_X
                abs_cyclist_filter.P_init = Sigma_G_tp
        
        elif is_first_obs_rel is False:
            no_more_obserbable = True
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
            abs_cyclist_obs = transform_L2G(G_ego=ego_pred, L_tp=cyclist_pred)
            
            jacobian_G_ego, jacobian_L_tp = jacoboan_transform_L2G(
                G_ego=ego_pred, L_tp=cyclist_pred
            )
            Sigma_G_tp = (
                # jacobian_G_ego @ Sigma_G_ego @ jacobian_G_ego.T
                jacobian_L_tp @ Sigma_L_cyclist @ jacobian_L_tp.T
            )
            abs_X = np.array(
                [
                    [abs_cyclist_obs.x],
                    [abs_cyclist_obs.y],
                    [abs_cyclist_obs.theta],
                    [abs_cyclist_obs.vel],
                ]
            )
            abs_R = Sigma_G_tp
            
            if no_more_obserbable:
                abs_cyclist_filter.predict(dt, U, Q)
            else:
                abs_cyclist_filter.X = abs_X
                abs_cyclist_filter.P = abs_R
                abs_cyclist_filter.Xp_memory.append(abs_X)
                abs_cyclist_filter.Pp_memory.append(abs_R)
        # === end t = i+1 predict === #

    return abs_cyclist_filter