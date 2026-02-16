import numpy as np

from ..simulator.map_constructor import BaseMap
from .kalman_filter import KalmanFilter


class BaseInfo:
    def __init__(self) -> None:
        pass

    def is_visible(self, x: float, y: float):
        raise NotImplementedError

    def observe(self, X: np.ndarray, filter: KalmanFilter):
        raise NotImplementedError

    def __str__(self):
        cls_name = self.__class__.__name__
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{cls_name}({attrs})"


class BasePosSensor(BaseInfo):
    def __init__(self, pos_std: float, fps: int) -> None:
        self.pos_std = pos_std
        self.fps = fps
        self.last_Z = None
        self.last_R = None
        self.last_H = None

    def is_visible(self, x: float, y: float):
        raise NotImplementedError

    def observe(self, X: np.ndarray, filter: KalmanFilter):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.array([[self.pos_std**2, 0], [0, self.pos_std**2]])
        Z = np.array(
            [
                [X[0, 0] + np.random.randn() * self.pos_std],
                [X[1, 0] + np.random.randn() * self.pos_std],
            ]
        )

        self.last_Z = Z
        self.last_R = R
        self.last_H = H
        filter.update(Z, R, H)


class BaseVelSensor(BaseInfo):
    def __init__(self, vel_std: float, fps: int) -> None:
        self.vel_std = vel_std
        self.fps = fps
        self.last_Z = None
        self.last_R = None
        self.last_H = None

    def is_visible(self, x: float, y: float):
        raise NotImplementedError

    def observe(self, X: np.ndarray, filter: KalmanFilter):
        H = np.array([[0, 0, 0, 1]])
        R = np.array([[self.vel_std**2]])
        Z = np.array([[X[3, 0] + np.random.randn() * self.vel_std]])

        self.last_Z = Z
        self.last_R = R
        self.last_H = H
        filter.update(Z, R, H)


class BaseThetaSensor(BaseInfo):
    def __init__(self, theta_std: float, fps: int) -> None:
        self.theta_std = theta_std
        self.fps = fps
        self.last_Z = None
        self.last_R = None
        self.last_H = None

    def is_visible(self, x: float, y: float):
        raise NotImplementedError

    def observe(self, X: np.ndarray, filter: KalmanFilter):
        H = np.array([[0, 0, 1, 0]])
        R = np.array([[self.theta_std**2]])
        Z = np.array([[X[2, 0] + np.random.randn() * self.theta_std]])

        self.last_Z = Z
        self.last_R = R
        self.last_H = H
        filter.update(Z, R, H)


class BasePosVelSensor(BaseInfo):
    def __init__(self, pos_std: float, vel_std: float, fps: int) -> None:
        self.pos_std = pos_std
        self.vel_std = vel_std
        self.fps = fps
        self.last_Z = None
        self.last_R = None
        self.last_H = None

    def is_visible(self, x: float, y: float):
        raise NotImplementedError

    def observe(self, X, filter: KalmanFilter):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        R = np.array(
            [
                [self.pos_std**2, 0, 0],
                [0, self.pos_std**2, 0],
                [0, 0, self.vel_std**2],
            ]
        )
        Z = np.array(
            [
                [X[0, 0] + np.random.randn() * self.pos_std],
                [X[1, 0] + np.random.randn() * self.pos_std],
                [X[3, 0] + np.random.randn() * self.vel_std],
            ]
        )

        self.last_Z = Z
        self.last_R = R
        self.last_H = H
        filter.update(Z, R, H)


class GNSS(BasePosSensor):
    def __init__(self, pos_std: float, fps: int) -> None:
        super().__init__(pos_std, fps)

    def is_visible(self, x: float, y: float):
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


class RoadSideSensor(BasePosSensor):
    def __init__(self, pos_std: float, fps: int, range: float, x: float, y: float) -> None:
        super().__init__(pos_std, fps)
        self.range = range
        self.x = x
        self.y = y

    def is_visible(self, x: float, y: float):
        distance = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        return distance <= self.range

    def observe(self, X, filter: KalmanFilter):
        super().observe(X, filter)


class LinkInfoSensor(BaseInfo):
    def __init__(self, fps: int, map: BaseMap, std_denominator: float) -> None:
        self.fps = fps
        self.map = map
        self.link_width = map.link_width
        assert std_denominator > 1e-6
        self.pos_std = self.link_width / std_denominator
        self.now_link_normal: np.ndarray | None = None  # [a, b]
        self.now_link_c: float | None = None

    def observe(self, X: None, filter: KalmanFilter, link_idx: tuple[int, int] | None = None) -> bool:
        assert self.now_link_normal is None and self.now_link_c is None

        pos_vec = filter.X[:2, 0]
        pos_cov = filter.P[:2, :2]

        if link_idx is None:
            link_idx = self.map.pred_in_only_link(pos_vec, pos_cov)
            if link_idx is None:
                return False

        now_link = self.map.id2link[link_idx]
        link_theta = now_link.theta
        p0 = np.array([now_link.entrance_point.x, now_link.entrance_point.y])

        # along road t = [cosθ, sinθ]
        # normal n = [-sinθ, cosθ]
        n = np.array([-np.sin(link_theta), np.cos(link_theta)])
        c = -np.dot(n, p0)  # ax + by + c = 0

        H = np.array([[n[0], n[1], 0, 0]])
        Z = np.array([[-c]])
        R = np.array([[self.pos_std**2]])

        self.last_Z = Z
        filter.update(Z, R, H)

        self.now_link_normal = None
        self.now_link_c = None
        return True
