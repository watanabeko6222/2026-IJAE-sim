import numpy as np

from ..simulator.moving_generator import TP_State
from .coordinate_transformation import jacobian_transform_L2G, transform_L2G
from .kalman_filter import KalmanFilter


class GlobalTPFilter:
    def localization(
        self,
        ego_filter: KalmanFilter,
        rel_filter: KalmanFilter,
        aggregate_step: int = 10,
    ) -> KalmanFilter | None:
        """Fuse ego and relative filters to obtain absolute cyclist localization.

        Args:
            ego_filter: Ego vehicle Kalman filter
            rel_filter: Relative Kalman filter
            aggregate_step: Number of steps to aggregate
        Returns:
            Updated absolute cyclist filter
        """
        integrated_filter: KalmanFilter | None = None
        start_t = max(ego_filter.step_offset, rel_filter.step_offset) + aggregate_step
        end_t = min(len(ego_filter), len(rel_filter))

        if start_t >= end_t:
            return integrated_filter

        for t in range(start_t, end_t):
            rel_pred = rel_filter.get_pred(t)
            rel_state_array, Sigma_L = rel_pred.X, rel_pred.P
            rel_tp = TP_State(
                x=rel_state_array[0, 0],
                y=rel_state_array[1, 0],
                theta=rel_state_array[2, 0],
                vel=rel_state_array[3, 0],
            )

            ego_pred = ego_filter.get_pred(t)
            ego_state_array, Sigma_G = ego_pred.X, ego_pred.P
            ego_tp = TP_State(
                x=ego_state_array[0, 0],
                y=ego_state_array[1, 0],
                theta=ego_state_array[2, 0],
                vel=ego_state_array[3, 0],
            )

            abs_tp = transform_L2G(G_ego=ego_tp, L_tp=rel_tp)
            jacobian_G, jacobian_L = jacobian_transform_L2G(G_ego=ego_tp, L_tp=rel_tp)
            Sigma_abs = jacobian_G @ Sigma_G @ jacobian_G.T + jacobian_L @ Sigma_L @ jacobian_L.T
            measurement = np.array(
                [
                    [abs_tp.x],
                    [abs_tp.y],
                    [abs_tp.theta],
                    [abs_tp.vel],
                ]
            )

            if integrated_filter is None:
                integrated_filter = KalmanFilter(step_offset=t)
                integrated_filter.X = measurement.copy()
                integrated_filter.P = Sigma_abs.copy()
                integrated_filter.X_init = integrated_filter.X.copy()
                integrated_filter.P_init = integrated_filter.P.copy()
            else:
                integrated_filter.Pp_memory.append(Sigma_abs.copy())
                integrated_filter.Xp_memory.append(measurement.copy())

        return integrated_filter
