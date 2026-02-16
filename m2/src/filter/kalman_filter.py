from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from ..utils import normalize_angle


@dataclass
class KalmanPrediction:
    X: np.ndarray
    P: np.ndarray


class KalmanFilter(object):
    """2d KalmanFilter
    Args:
        X: state vector; x, y, theta, velocity
        P: state covariance matrix
    """

    def __init__(self, allow_negative_velocity: bool = False, step_offset: int = 0) -> None:
        self.X: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.X_init: np.ndarray | None = None
        self.P_init: np.ndarray | None = None

        self.Xu_memory: list[np.ndarray] = []
        self.Pu_memory: list[np.ndarray] = []
        self.Xp_memory: list[np.ndarray] = []
        self.Pp_memory: list[np.ndarray] = []

        self.Z_memory: list[np.ndarray] = []
        self.R_memory: list[np.ndarray] = []
        self.U_memory: list[np.ndarray] = []
        self.Q_memory: list[np.ndarray] = []

        self.allow_negative_velocity = allow_negative_velocity
        self.step_offset = step_offset

    def predict(self, dt: float, U: np.ndarray, Q: np.ndarray) -> None:
        """Predict next state
        Args:
            dt: time step
            U: control vector; acceleration, angular velocity
            Q: control covariance matrix
            Cs: error in motion model
        """
        F = np.array(
            [
                [
                    1,
                    0,
                    -(self.X[3, 0]) * np.sin(self.X[2, 0]) * dt,
                    np.cos(self.X[2, 0]) * dt,
                ],
                [
                    0,
                    1,
                    (self.X[3, 0]) * np.cos(self.X[2, 0]) * dt,
                    np.sin(self.X[2, 0]) * dt,
                ],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        B = np.array(
            [
                [0, (np.cos(self.X[2, 0]) * dt**2) / 2],
                [0, (np.sin(self.X[2, 0]) * dt**2) / 2],
                [dt, 0],
                [0, dt],
            ]
        )
        self.X = self.X + np.array(
            [
                [self.X[3, 0] * np.cos(self.X[2, 0]) * dt + U[0, 0] * dt**2 / 2],
                [self.X[3, 0] * np.sin(self.X[2, 0]) * dt + U[1, 0] * dt**2 / 2],
                [U[0, 0] * dt],
                [U[1, 0] * dt],
            ]
        )
        self.X[2, 0] = normalize_angle(self.X[2, 0])
        if self.X[3, 0] < 0 and not self.allow_negative_velocity:
            self.X[3, 0] = 0
        self.P = F @ self.P @ F.T + B @ Q @ B.T

        self.Xp_memory.append(self.X)
        self.Pp_memory.append(self.P)
        self.U_memory.append(U)
        self.Q_memory.append(Q)

    def update(self, Z: np.ndarray, R: np.ndarray, H: np.ndarray) -> None:
        """Update state
        Args:
            Z: measurement vector
            R: measurement covariance matrix
            H: measurement matrix
        """
        if Z.shape[0] > 1:
            self.last_Z = Z
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        self.X = self.X + K @ (Z - H @ self.X)
        self.X[2, 0] = normalize_angle(self.X[2, 0])
        if self.X[3, 0] < 0 and not self.allow_negative_velocity:
            self.X[3, 0] = 0
        self.P = (np.eye(4) - K @ H) @ self.P @ (np.eye(4) - K @ H).T + K @ R @ K.T

        self.Z_memory.append(Z)
        self.R_memory.append(R)
        self.Xu_memory.append(self.X)
        self.Pu_memory.append(self.P)

    def _covariance_intersection1(
        self,
        x1: np.ndarray,
        P1: np.ndarray,
        x2: np.ndarray,
        P2: np.ndarray,
        step: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Covariance intersection
        Args:
            x1: state vector 1
            P1: state covariance matrix 1
            x2: state vector 2
            P2: state covariance matrix 2
        Returns:
            x: fused state vector
            P: fused state covariance matrix
            omega: weight
        """
        for omega in np.arange(0, 1 + step, step):
            P_inv = omega * np.linalg.inv(P1) + (1 - omega) * np.linalg.inv(P2)
            P = np.linalg.inv(P_inv)
            x = P @ (omega * np.linalg.inv(P1) @ x1 + (1 - omega) * np.linalg.inv(P2) @ x2)
            trace = np.trace(P)
            if omega == 0:
                min_trace = trace
                best_x = x
                best_P = P
                best_omega = omega
            elif trace < min_trace:
                min_trace = trace
                best_x = x
                best_P = P
                best_omega = omega

        return best_x, best_P, best_omega

    def _covariance_intersection2(
        self,
        x1: np.ndarray,
        P1: np.ndarray,
        x2: np.ndarray,
        P2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Covariance intersection using scalar optimization (falls back to bounds if needed)."""
        inv_P1 = np.linalg.inv(P1)
        inv_P2 = np.linalg.inv(P2)

        def fuse(omega: float) -> tuple[np.ndarray, np.ndarray, float]:
            P_inv = omega * inv_P1 + (1 - omega) * inv_P2
            P = np.linalg.inv(P_inv)
            x = P @ (omega * inv_P1 @ x1 + (1 - omega) * inv_P2 @ x2)
            return x, P, float(omega)

        def trace_objective(omega: float) -> float:
            _, P, _ = fuse(omega)
            return np.trace(P)

        result = minimize_scalar(trace_objective, bounds=(0.0, 1.0), method="bounded")
        candidate_omegas = [0.0, 1.0]
        candidate_omegas.append(result.x if result.success else 0.5)

        best_x, best_P, best_omega = None, None, None
        min_trace = float("inf")
        for omega in candidate_omegas:
            x, P, omega_val = fuse(omega)
            trace = np.trace(P)
            if trace < min_trace:
                min_trace = trace
                best_x, best_P, best_omega = x, P, omega_val

        return best_x, best_P, best_omega

    def update_CI(self, Z: np.ndarray, R: np.ndarray, H: np.ndarray) -> None:
        """Update state with covariance intersection
        Args:
            Z: measurement vector
            R: measurement covariance matrix
            H: measurement matrix
        """
        x1 = self.X
        P1 = self.P
        x2 = np.linalg.inv(H) @ Z
        P2 = np.linalg.inv(H) @ R @ np.linalg.inv(H).T
        x, P, omega = self._covariance_intersection2(x1, P1, x2, P2)
        self.X = x
        if self.X[3, 0] < 0 and not self.allow_negative_velocity:
            self.X[3, 0] = 0
        self.P = P

        self.Z_memory.append(Z)
        self.R_memory.append(R)
        self.Xu_memory.append(self.X)
        self.Pu_memory.append(self.P)

    def __len__(self) -> int:
        # add initial state
        return len(self.Xp_memory) + self.step_offset + 1

    def get_pred(self, idx: int) -> KalmanPrediction | None:
        if idx - self.step_offset < 0:
            return None
        elif idx - self.step_offset == 0:
            return KalmanPrediction(self.X_init, self.P_init)
        elif idx - self.step_offset > len(self.Xp_memory):
            return None
        else:
            return KalmanPrediction(
                self.Xp_memory[idx - self.step_offset - 1],
                self.Pp_memory[idx - self.step_offset - 1],
            )

    def save_csv(self, file_path: str) -> None:
        """Save state history to CSV file
        Args:
            file_path: path to save the CSV file
        """
        import pandas as pd

        data = {
            "step": list(range(self.step_offset, self.step_offset + len(self.Xp_memory) + 1)),
            "x": [self.X_init[0, 0]] + [x[0, 0] for x in self.Xp_memory],
            "y": [self.X_init[1, 0]] + [x[1, 0] for x in self.Xp_memory],
            "theta": [self.X_init[2, 0]] + [x[2, 0] for x in self.Xp_memory],
            "vel": [self.X_init[3, 0]] + [x[3, 0] for x in self.Xp_memory],
            "P00": [self.P_init[0, 0]] + [p[0, 0] for p in self.Pp_memory],
            "P11": [self.P_init[1, 1]] + [p[1, 1] for p in self.Pp_memory],
            "P22": [self.P_init[2, 2]] + [p[2, 2] for p in self.Pp_memory],
            "P33": [self.P_init[3, 3]] + [p[3, 3] for p in self.Pp_memory],
            "P01": [self.P_init[0, 1]] + [p[0, 1] for p in self.Pp_memory],
            "P02": [self.P_init[0, 2]] + [p[0, 2] for p in self.Pp_memory],
            "P03": [self.P_init[0, 3]] + [p[0, 3] for p in self.Pp_memory],
            "P12": [self.P_init[1, 2]] + [p[1, 2] for p in self.Pp_memory],
            "P13": [self.P_init[1, 3]] + [p[1, 3] for p in self.Pp_memory],
            "P23": [self.P_init[2, 3]] + [p[2, 3] for p in self.Pp_memory],
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

    def __str__(self):
        return f"KalmanPrediction(len={len(self)}, step_offset={self.step_offset})"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    ekf = KalmanFilter()
    import time

    # debug covariance intersection
    for _ in range(10):
        X1 = np.random.uniform(-10, 10, (4, 1))
        P1 = np.random.uniform(0.1, 5, (4, 4))
        P1 = P1 @ P1.T  # make it positive definite
        X2 = np.random.uniform(-10, 10, (4, 1))
        P2 = np.random.uniform(0.1, 5, (4, 4))
        P2 = P2 @ P2.T  # make it positive definite
        start1 = time.time()
        x, P, omega = ekf._covariance_intersection1(X1, P1, X2, P2)
        print(f"Method1 Result: trace_P={np.trace(P)}, omega={omega}")
        end1 = time.time()
        print(f"Method1 Time: {end1 - start1:.6f} sec")
        start2 = time.time()
        x, P, omega = ekf._covariance_intersection2(X1, P1, X2, P2)
        print(f"Method2 Result: trace_P={np.trace(P)}, omega={omega}")
        end2 = time.time()
        print(f"Method2 Time: {end2 - start2:.6f} sec")
