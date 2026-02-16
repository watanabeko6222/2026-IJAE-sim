import numpy as np

from ..simulator.moving_generator import TP_State
from ..utils import normalize_angle


def transform_G2L(G_tp: TP_State, G_ego: TP_State) -> TP_State:
    """Transform from global coordinates to local coordinates."""
    return TP_State(
        x=(G_tp.x - G_ego.x) * np.cos(G_ego.theta) + (G_tp.y - G_ego.y) * np.sin(G_ego.theta),
        y=(G_tp.y - G_ego.y) * np.cos(G_ego.theta) - (G_tp.x - G_ego.x) * np.sin(G_ego.theta),
        theta=normalize_angle(G_tp.theta - G_ego.theta),
        vel=np.sqrt(G_tp.vel**2 + G_ego.vel**2 - 2 * G_tp.vel * G_ego.vel * np.cos(G_tp.theta - G_ego.theta)),
    )


def transform_L2G(G_ego: TP_State, L_tp: TP_State) -> TP_State:
    return TP_State(
        x=G_ego.x + L_tp.x * np.cos(G_ego.theta) - L_tp.y * np.sin(G_ego.theta),
        y=G_ego.y + L_tp.x * np.sin(G_ego.theta) + L_tp.y * np.cos(G_ego.theta),
        theta=normalize_angle(G_ego.theta + L_tp.theta),
        vel=np.sqrt(G_ego.vel**2 + L_tp.vel**2 + 2 * G_ego.vel * L_tp.vel * np.cos(L_tp.theta)),
    )


def jacobian_transform_L2G(G_ego: TP_State, L_tp: TP_State) -> np.ndarray:
    G_ego_element_44 = (G_ego.vel + L_tp.vel * np.cos(L_tp.theta)) / np.sqrt(
        G_ego.vel**2 + L_tp.vel**2 + 2 * G_ego.vel * L_tp.vel * np.cos(L_tp.theta)
    )
    jacobian_G_ego = np.array(
        [
            [1, 0, -L_tp.x * np.sin(G_ego.theta) - L_tp.y * np.cos(G_ego.theta), 0],
            [0, 1, L_tp.x * np.cos(G_ego.theta) - L_tp.y * np.sin(G_ego.theta), 0],
            [0, 0, 1, 0],
            [0, 0, 0, G_ego_element_44],
        ],
    )

    L_tp_element_34 = (
        -L_tp.vel
        * G_ego.vel
        * np.sin(L_tp.theta)
        / np.sqrt(G_ego.vel**2 + L_tp.vel**2 + 2 * G_ego.vel * L_tp.vel * np.cos(L_tp.theta))
    )
    L_tp_element_44 = (L_tp.vel + G_ego.vel * np.cos(L_tp.theta)) / np.sqrt(
        G_ego.vel**2 + L_tp.vel**2 + 2 * G_ego.vel * L_tp.vel * np.cos(L_tp.theta)
    )
    jacobian_L_tp = np.array(
        [
            [np.cos(G_ego.theta), -np.sin(G_ego.theta), 0, 0],
            [np.sin(G_ego.theta), np.cos(G_ego.theta), 0, 0],
            [0, 0, 1, L_tp_element_34],
            [0, 0, 0, L_tp_element_44],
        ]
    )

    return jacobian_G_ego, jacobian_L_tp
