import numpy as np
from src.kalman_filter import KalmanFilter


class BaseInfo:
    def __init__(self) -> None:
        pass

    def is_visible(self, x, y):
        raise NotImplementedError

    def observe(self, X, filter: KalmanFilter):
        raise NotImplementedError


class BasePosSensor(BaseInfo):
    def __init__(self, pos_std, fps, do_calc_vel=True, do_clac_theta=True) -> None:
        self.pos_std = pos_std
        self.do_calc_vel = do_calc_vel
        self.do_calc_theta = do_clac_theta
        self.fps = fps
        self.last_Z = None

    def is_visible(self, x, y):
        raise NotImplementedError

    def observe(self, X, filter: KalmanFilter):
        if self.last_Z is None:
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            R = np.array([[self.pos_std**2, 0], [0, self.pos_std**2]])
            Z = np.array([[X[0, 0] + np.random.randn() * self.pos_std], [X[1, 0] + np.random.randn() * self.pos_std]])
        
        else:
            calc_vel = np.linalg.norm([X[0, 0] - self.last_Z[0, 0], X[1, 0] - self.last_Z[1, 0]]) / (1 / self.fps)
            calc_theta = np.arctan2(X[1, 0] - self.last_Z[1, 0], X[0, 0] - self.last_Z[0, 0])
            if self.do_calc_vel and self.do_calc_theta:
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                R = np.array(
                    [
                        [self.pos_std**2, 0, 0, 0],
                        [0, self.pos_std**2, 0, 0],
                        # [0, 0, (np.sqrt(2) * self.pos_std * (calc_vel / self.fps)) ** 2, 0],
                        [0, 0, (np.sqrt(2) * self.pos_std * (6.9 / self.fps)) ** 2, 0],
                        [0, 0, 0, (np.sqrt(2) * self.pos_std * self.fps) ** 2],
                    ]
                )
                Z = np.array(
                    [
                        [X[0, 0] + np.random.randn() * self.pos_std],
                        [X[1, 0] + np.random.randn() * self.pos_std],
                        [calc_theta],
                        [calc_vel],
                    ]
                )
            elif self.do_calc_vel:
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                R = np.array(
                    [[self.pos_std**2, 0, 0, ], [0, self.pos_std**2, 0], [0, 0, (self.pos_std * self.fps) ** 2]]
                )
                Z = np.array(
                    [[X[0, 0] + np.random.randn() * self.pos_std], [X[1, 0] + np.random.randn() * self.pos_std], [calc_vel]]
                )
            elif self.do_calc_theta:
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
                R = np.array([[self.pos_std**2, 0, 0], [0, self.pos_std**2, 0], [0, 0, 10**2]])
                Z = np.array(
                    [
                        [X[0, 0] + np.random.randn() * self.pos_std],
                        [X[1, 0] + np.random.randn() * self.pos_std],
                        [calc_theta],
                    ]
                )
            else:
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
                R = np.array([[self.pos_std**2, 0], [0, self.pos_std**2]])
                Z = np.array([[X[0, 0] + np.random.randn() * self.pos_std], [X[1, 0] + np.random.randn() * self.pos_std]])

        self.last_Z = Z
        filter.update(Z, R, H)


class BaseVelSensor(BaseInfo):
    def __init__(self, vel_std, fps) -> None:
        self.vel_std = vel_std
        self.fps = fps
        self.last_Z = None

    def is_visible(self, x, y):
        raise NotImplementedError

    def observe(self, X, filter: KalmanFilter):
        if self.last_Z is None:
            H = np.array([[0, 0, 0, 1]])
            R = np.array([[self.vel_std**2]])
            Z = np.array([[X[3, 0] + np.random.randn() * self.vel_std]])
        else:
            H = np.array([[0, 0, 0, 1]])
            R = np.array([[self.vel_std**2]])
            Z = np.array([[X[3, 0] + np.random.randn() * self.vel_std]])

        self.last_Z = Z
        filter.update(Z, R, H)


class BasePosVelSensor(BaseInfo):
    def __init__(self, pos_std, vel_std, fps, do_calc_theta=True) -> None:
        self.pos_std = pos_std
        self.vel_std = vel_std
        self.do_calc_theta = do_calc_theta
        self.fps = fps
        self.last_Z = None

    def is_visible(self, x, y):
        raise NotImplementedError

    def observe(self, X, filter: KalmanFilter):
        if self.last_Z is None:
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            R = np.array([[self.pos_std**2, 0, 0], [0, self.pos_std**2, 0], [0, 0, self.vel_std**2]])
            Z = np.array(
                [
                    [X[0, 0] + np.random.randn() * self.pos_std],
                    [X[1, 0] + np.random.randn() * self.pos_std],
                    [X[3, 0] + np.random.randn() * self.vel_std],
                ]
            )
        else:
            calc_theta = np.arctan2(X[1, 0] - self.last_Z[1, 0], X[0, 0] - self.last_Z[0, 0])
            if self.do_calc_theta:
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                R = np.array(
                    [
                        [self.pos_std**2, 0, 0, 0],
                        [0, self.pos_std**2, 0, 0],
                        [0, 0, 10**2, 0],
                        [0, 0, 0, self.vel_std**2],
                    ]
                )
                Z = np.array(
                    [
                        [X[0, 0] + np.random.randn() * self.pos_std],
                        [X[1, 0] + np.random.randn() * self.pos_std],
                        [calc_theta],
                        [X[3, 0] + np.random.randn() * self.vel_std],
                    ]
                )
            else:
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                R = np.array([[self.pos_std**2, 0, 0], [0, self.pos_std**2, 0], [0, 0, self.vel_std**2]])
                Z = np.array(
                    [
                        [X[0, 0] + np.random.randn() * self.pos_std],
                        [X[1, 0] + np.random.randn() * self.pos_std],
                        [X[3, 0] + np.random.randn() * self.vel_std],
                    ]
                )

        self.last_Z = Z
        filter.update(Z, R, H)


class GNSS(BasePosSensor):
    def __init__(self, pos_std, fps, do_calc_vel=True, do_clac_theta=True) -> None:
        super().__init__(pos_std, fps, do_calc_vel, do_clac_theta)

    def is_visible(self, x, y):
        return True

    def observe(self, X, filter: KalmanFilter):
        super().observe(X, filter)


class WheelVel(BaseVelSensor):
    def __init__(self, vel_std, fps) -> None:
        super().__init__(vel_std, fps)

    def is_visible(self, x, y):
        return True

    def observe(self, X, filter: KalmanFilter):
        super().observe(X, filter)
