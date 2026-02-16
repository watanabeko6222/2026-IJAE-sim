import numpy as np


class KalmanFilter(object):
    """2d KalmanFilter
    Args:
        X: state vector; x, y, theta, velocity
        P: state covariance matrix
    """

    def __init__(self) -> None:
        self.X = None
        self.P = None
        self.X_init = None
        self.P_init = None
        
        self.Xu_memory = []
        self.Pu_memory = []
        self.Xp_memory = []
        self.Pp_memory = []

        self.Z_memory = []
        self.U_memory = []
        self.Q_memory = []
        
        self.smoothed = False
    
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
                [1, 0, -(self.X[3, 0]) * np.sin(self.X[2, 0]) * dt, np.cos(self.X[2, 0]) * dt],
                [0, 1, (self.X[3, 0]) * np.cos(self.X[2, 0]) * dt, np.sin(self.X[2, 0]) * dt],
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
        self.X = self.X + \
            np.array(
                [
                    [self.X[3, 0] * np.cos(self.X[2, 0]) * dt + U[0, 0] * dt**2 / 2],
                    [self.X[3, 0] * np.sin(self.X[2, 0]) * dt + U[1, 0] * dt**2 / 2],
                    [U[0, 0] * dt],
                    [U[1, 0] * dt],
                ]
            )
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
        if self.X[3, 0] < 0:
            self.X[3, 0] = 0
        self.P = (np.eye(4) - K @ H) @ self.P @ (np.eye(4) - K @ H).T + K @ R @ K.T
        
        self.Z_memory.append(Z)
        self.Xu_memory.append(self.X)
        self.Pu_memory.append(self.P)
    
    def get_state(self, idx:int) -> np.ndarray:
        if self.smoothed:
            if idx == 0:
                return self.X_init_smooth, self.P_init_smooth
            else:
                return self.X_smooths[idx - 1], self.P_smooths[idx - 1]
        else:
            if idx == 0:
                return self.X_init, self.P_init
            else:
                return self.Xp_memory[idx - 1], self.Pp_memory[idx - 1]
    
    def __len__(self) -> int:
        return len(self.Xp_memory) + 1
    
    def smooth(self):
        """RTS smoothing
        """
        N = len(self.Xp_memory)
        if N == 0:
            raise ValueError("No prediction")

        X_smooths = [None]*N
        P_smooths = [None]*N

        X_smooths[-1] = self.Xp_memory[-1].copy()
        P_smooths[-1] = self.Pp_memory[-1].copy()
        
        dt = 0.1 # FIXME: hard coding

        for k in range(N-2, -1, -1):
            F = np.array(
                [
                    [1, 0, -(self.Xp_memory[k][3, 0]) * np.sin(self.Xp_memory[k][2, 0]) * dt, np.cos(self.Xp_memory[k][2, 0]) * dt],
                    [0, 1, (self.Xp_memory[k][3, 0]) * np.cos(self.Xp_memory[k][2, 0]) * dt, np.sin(self.Xp_memory[k][2, 0]) * dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            B = np.array(
                [
                    [0, (np.cos(self.Xp_memory[k][2, 0]) * dt**2) / 2],
                    [0, (np.sin(self.Xp_memory[k][2, 0]) * dt**2) / 2],
                    [dt, 0],
                    [0, dt],
                ]
            )

            Pp = F @ self.Pp_memory[k] @ F.T + B @ self.Q_memory[k] @ B.T
            xp = self.Xp_memory[k] + \
                np.array(
                    [
                        [self.Xp_memory[k][3, 0] * np.cos(self.Xp_memory[k][2, 0]) * dt + self.U_memory[k][0, 0] * dt**2 / 2],
                        [self.Xp_memory[k][3, 0] * np.sin(self.Xp_memory[k][2, 0]) * dt + self.U_memory[k][1, 0] * dt**2 / 2],
                        [self.U_memory[k][0, 0] * dt],
                        [self.U_memory[k][1, 0] * dt],
                    ]
                )
            K = self.Pp_memory[k] @ F.T @ np.linalg.inv(Pp)
            x = self.Xp_memory[k] + K @ (X_smooths[k+1] - xp)
            P = self.Pp_memory[k] + K @ (P_smooths[k+1] - Pp) @ K.T
            
            X_smooths[k] = x
            P_smooths[k] = P
        
        # idx == 0
        F = np.array(
            [
                [1, 0, -(self.X_init[3, 0]) * np.sin(self.X_init[2, 0]) * dt, np.cos(self.X_init[2, 0]) * dt],
                [0, 1, (self.X_init[3, 0]) * np.cos(self.X_init[2, 0]) * dt, np.sin(self.X_init[2, 0]) * dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        B = np.array(
            [
                [0, (np.cos(self.X_init[2, 0]) * dt**2) / 2],
                [0, (np.sin(self.X_init[2, 0]) * dt**2) / 2],
                [dt, 0],
                [0, dt],
            ]
        )

        Pp = F @ self.P_init @ F.T # + B @ self.Q_memory[0] @ B.T
        xp = self.X_init + \
            np.array(
                [
                    [self.X_init[3, 0] * np.cos(self.X_init[2, 0]) * dt + 0 * dt**2 / 2],
                    [self.X_init[3, 0] * np.sin(self.X_init[2, 0]) * dt + 0 * dt**2 / 2],
                    [0 * dt],
                    [0 * dt],
                ]
            )
        K = self.P_init @ F.T @ np.linalg.inv(Pp)
        x = self.X_init + K @ (X_smooths[0] - xp)
        P = self.P_init + K @ (P_smooths[0] - Pp) @ K.T
        
        X_init_smooth = x
        P_init_smooth = P
        
        self.X_smooths = X_smooths
        self.P_smooths = P_smooths
        self.X_init_smooth = X_init_smooth
        self.P_init_smooth = P_init_smooth
        
        self.smoothed = True
    
    def get_state_smooth(self, idx:int) -> np.ndarray:
        if idx == 0:
            return self.X_init_smooth, self.P_init_smooth
        else:
            return self.X_smooths[idx - 1], self.P_smooths[idx - 1]
            


if __name__ == "__main__":
    ekf = KalmanFilter(
        X=np.array([[0],
                    [0],
                    [0],
                    [0]]),
        P=np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    )
    Z = np.array([[0.5],
                  [0.5]])
    R = np.array([[0.5, 0],
                  [0, 0.5]])
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    ekf.update(Z, R, H)
    print(ekf.X)
    print(ekf.P)
    U = np.array([[0.5],
                  [0.5]])
    Q = np.array([[0.5, 0],
                  [0, 0.5]])
    ekf.predict(0.1, U, Q)
    print(ekf.X)
    print(ekf.P)

    ekf.update(Z, R, H)
    ekf.predict(0.1, U, Q)
    ekf.update(Z, R, H)
    ekf.predict(0.1, U, Q)
    
    ekf.smooth()
    