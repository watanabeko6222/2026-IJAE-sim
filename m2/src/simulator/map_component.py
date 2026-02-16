from dataclasses import dataclass

import matplotlib.patches as patches
import numpy as np
from scipy.stats import chi2
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union


@dataclass
class Point:
    x: float
    y: float

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)


def check_intersection(p1: Point, p2: Point, p3: Point, p4: Point):
    tc1 = (p1.x - p2.x) * (p3.y - p1.y) + (p1.y - p2.y) * (p1.x - p3.x)
    tc2 = (p1.x - p2.x) * (p4.y - p1.y) + (p1.y - p2.y) * (p1.x - p4.x)
    td1 = (p3.x - p4.x) * (p1.y - p3.y) + (p3.y - p4.y) * (p3.x - p1.x)
    td2 = (p3.x - p4.x) * (p2.y - p3.y) + (p3.y - p4.y) * (p3.x - p2.x)
    return tc1 * tc2 < 0 and td1 * td2 < 0


@dataclass(init=False)
class Corner:
    points: list[Point]

    def __init__(self, points: list[Point]) -> None:
        assert len(points) >= 3, f"ConvexCorner requires at least 3 points, got {points}"
        self.points = points
        pts = np.array([[p.x, p.y] for p in points])
        self.viz_poly = patches.Polygon(pts, color="red", alpha=0.2)

    def check_point_inclusion(self, point: Point) -> bool:
        """Check if the point is in the corner using cross product"""
        num_points = len(self.points)

        has_positive = False
        has_negative = False
        for i in range(num_points):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % num_points]
            cross_product = (p2.x - p1.x) * (point.y - p1.y) - (p2.y - p1.y) * (point.x - p1.x)

            if cross_product > 0:
                has_positive = True
            elif cross_product < 0:
                has_negative = True

            # signs of cross products differ, point is outside
            if has_positive and has_negative:
                return False

        return True

    def check_corner_intersection(self, other: "Corner") -> bool:
        """Return True if the two rectangle nodes intersect."""
        corners1 = self.points
        corners2 = other.points

        # 1) Any rectangle corner inside the other rectangle
        for c in corners1:
            if other.check_point_inclusion(c):
                return True
        for c in corners2:
            if self.check_point_inclusion(c):
                return True

        # 2) Any rectangle edge intersects the other rectangle
        def seg_intersects_rect(p, q, rect: "Corner") -> bool:
            rect_corners = rect.points
            edges = [(rect_corners[i], rect_corners[(i + 1) % len(rect_corners)]) for i in range(len(rect_corners))]
            for r_p, r_q in edges:
                if check_intersection(p, q, r_p, r_q):
                    return True
            return False

        edges1 = [(corners1[i], corners1[(i + 1) % len(corners1)]) for i in range(len(corners1))]
        for p, q in edges1:
            if seg_intersects_rect(p, q, other):
                return True

        edges2 = [(corners2[i], corners2[(i + 1) % len(corners2)]) for i in range(len(corners2))]
        for p, q in edges2:
            if seg_intersects_rect(p, q, self):
                return True

        return False

    def check_ellipse_intersection(self, mean: np.ndarray, cov: np.ndarray, alpha: float = 0.95) -> bool:
        """Return True if the rectangle node intersects the covariance ellipse."""
        corners = self.points
        scale = chi2.ppf(alpha, df=2)

        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-12, None)
        T: np.ndarray = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T  # maps ellipse to unit circle

        def point_in_ellipse(x: float, y: float) -> bool:
            # Why: use Mahalanobis distance <= 1 to test inside
            d = np.array([x - float(mean[0]), y - float(mean[1])], dtype=float)
            u = T @ d
            return float(u @ u) <= scale + 1e-12

        # 1) Any rectangle corner inside the ellipse
        for c in corners:
            if point_in_ellipse(c.x, c.y):
                return True

        # 2) Ellipse center inside the rectangle (covers ellipse fully inside rect)
        center_point = Point(float(mean[0]), float(mean[1]))
        if self.check_point_inclusion(center_point):
            return True

        # 3) Any rectangle edge intersects the ellipse
        def seg_intersects_unit_circle(p, q) -> bool:
            # Why: test intersection in whitened space against unit circle
            P = T @ (np.array([p.x, p.y], dtype=float) - mean[:2].astype(float))
            Q = T @ (np.array([q.x, q.y], dtype=float) - mean[:2].astype(float))
            d = Q - P

            a = float(d @ d)
            b = 2.0 * float(P @ d)
            c = float(P @ P) - 1.0

            # Degenerate segment
            if a <= 1e-15:
                return float(P @ P) <= scale + 1e-12

            # Quick reject/accept via closest approach distance
            t0 = -b / (2.0 * a)
            t = min(1.0, max(0.0, t0))
            closest = P + t * d
            if float(closest @ closest) <= scale + 1e-12:
                return True

            # Exact root check (robust against grazing cases)
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                return False
            sqrt_disc = np.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)

        edges = [(corners[i], corners[(i + 1) % len(corners)]) for i in range(len(corners))]
        for p, q in edges:
            if seg_intersects_unit_circle(p, q):
                return True

        return False


def merge_close_points(points: list[Point], threshold: float = 0.5) -> list[Point]:
    if not points:
        return []

    # list({'sum_x': float, 'sum_y': float, 'count': int, 'center': Point})
    clusters: list[dict[str, object]] = []

    for p in points:
        matched = False
        for cluster in clusters:
            dx = cluster["center"].x - p.x
            dy = cluster["center"].y - p.y
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < threshold:
                cluster["sum_x"] += p.x
                cluster["sum_y"] += p.y
                cluster["count"] += 1

                cluster["center"] = Point(cluster["sum_x"] / cluster["count"], cluster["sum_y"] / cluster["count"])
                matched = True
                break
        if not matched:
            clusters.append({"sum_x": p.x, "sum_y": p.y, "count": 1, "center": p})

    return [c["center"] for c in clusters]


@dataclass
class Gateway:
    link_id: tuple[int, int]
    is_entrance: bool
    point: Point


@dataclass
class RoadSideUnit:
    id: int
    range: float  # in meters


@dataclass
class Node:
    id: int
    center: Point
    corner: Corner
    gateways: list[Gateway]
    connected_node_ids: list[int]
    connected_link_ids: list[tuple[int, int]]
    rsu: RoadSideUnit | None = None


@dataclass(init=False)
class Link:
    id: tuple[int, int]
    senior_node: Node
    entrance_point: Point
    junior_node: Node
    exit_point: Point
    corner: Corner
    theta: float
    length: float
    width: float
    base_waypoints: list[Point]

    def __init__(self, senior_node: Node, junior_node: Node, width: float) -> None:
        self.senior_node = senior_node
        self.junior_node = junior_node
        assert senior_node.id < junior_node.id, "senior_node id must be smaller than junior_node id"
        self.id = (senior_node.id, junior_node.id)
        self.entrance_point = Point(0, 0)
        self.exit_point = Point(0, 0)
        self.width = width

        # get link corner from inward node corner
        tmp_corner = []
        for corner, buf_point in zip(
            [senior_node.corner, junior_node.corner],
            [self.entrance_point, self.exit_point],
        ):
            for i in range(len(corner.points)):
                p1 = corner.points[i]
                p2 = corner.points[(i + 1) % len(corner.points)]
                if check_intersection(p1, p2, senior_node.center, junior_node.center):
                    tmp_corner.append(p1)
                    tmp_corner.append(p2)
                    buf_point.x = (p1.x + p2.x) / 2
                    buf_point.y = (p1.y + p2.y) / 2
                    break
        # sort corners as clockwise
        tmp_corner.sort(key=lambda x: np.arctan2(x.y - junior_node.center.y, x.x - junior_node.center.x))
        self.corner = Corner(points=tmp_corner)
        # append node entrance point
        senior_node.gateways.append(Gateway(link_id=self.id, is_entrance=True, point=self.entrance_point))
        junior_node.gateways.append(Gateway(link_id=self.id, is_entrance=False, point=self.exit_point))
        senior_node.connected_node_ids.append(junior_node.id)
        junior_node.connected_node_ids.append(senior_node.id)
        senior_node.connected_link_ids.append(self.id)
        junior_node.connected_link_ids.append(self.id)
        # calculate link theta from entrance point to exit point
        self.theta = np.arctan2(
            self.exit_point.y - self.entrance_point.y,
            self.exit_point.x - self.entrance_point.x,
        )
        # cache geometry for waypoint generation
        self._link_length = float(
            np.sqrt((self.exit_point.x - self.entrance_point.x) ** 2 + (self.exit_point.y - self.entrance_point.y) ** 2)
        )
        self._unit_vec = (float(np.cos(self.theta)), float(np.sin(self.theta)))
        self._normal_vec = (-self._unit_vec[1], self._unit_vec[0])
        self._waypoint_resolution = 0.1

        # calc base waypoints along the link centerline to keep the previous behaviour
        self.base_waypoints = self.generate_base_waypoints()
        self.length = self._link_length

    def __str__(self):
        return f"Link(id={self.id}, senior_node={self.senior_node.id}, junior_node={self.junior_node.id})"

    def __repr__(self):
        return self.__str__()

    def _resolve_offset(self, distance_from_wall: float | None, side: str) -> float:
        if distance_from_wall is None:
            return 0.0

        if distance_from_wall < 0:
            raise ValueError("distance_from_wall must be non-negative")
        if side not in ("left", "right"):
            raise ValueError("side must be either 'left' or 'right'")

        half_width = self.width / 2
        if distance_from_wall > half_width:
            raise ValueError(f"distance_from_wall={distance_from_wall} is larger than the half width {half_width:.3f}")

        offset_magnitude = max(half_width - distance_from_wall, 0.0)
        return offset_magnitude if side == "left" else -offset_magnitude

    def _offset_point(self, point: Point, offset_from_center: float) -> Point:
        return Point(
            point.x + self._normal_vec[0] * offset_from_center,
            point.y + self._normal_vec[1] * offset_from_center,
        )

    def _generate_waypoints(self, offset_from_center: float = 0.0) -> list[Point]:
        waypoints: list[Point] = []
        start_x = self.entrance_point.x + self._normal_vec[0] * offset_from_center
        start_y = self.entrance_point.y + self._normal_vec[1] * offset_from_center
        num_steps = int(self._link_length / self._waypoint_resolution)
        for i in range(num_steps):
            delta = i * self._waypoint_resolution
            x = start_x + self._unit_vec[0] * delta
            y = start_y + self._unit_vec[1] * delta
            waypoints.append(Point(x, y))
        return waypoints

    def generate_base_waypoints(self, distance_from_wall: float | None = None, side: str = "left") -> list[Point]:
        offset_from_center = self._resolve_offset(distance_from_wall, side)
        return self._generate_waypoints(offset_from_center=offset_from_center)

    def get_gateway_points(
        self,
        *,
        distance_from_wall: float | None = None,
        side: str = "left",
    ) -> tuple[Point, Point]:
        offset_from_center = self._resolve_offset(distance_from_wall, side)
        entrance = self._offset_point(self.entrance_point, offset_from_center)
        exit = self._offset_point(self.exit_point, offset_from_center)
        return entrance, exit

    def get_entrance_point(
        self,
        *,
        distance_from_wall: float | None = None,
        side: str = "left",
    ) -> Point:
        entrance, _ = self.get_gateway_points(distance_from_wall=distance_from_wall, side=side)
        return entrance

    def get_exit_point(
        self,
        *,
        distance_from_wall: float | None = None,
        side: str = "left",
    ) -> Point:
        _, exit = self.get_gateway_points(distance_from_wall=distance_from_wall, side=side)
        return exit


def corner2bbox(corner: Corner) -> tuple[float, float, float, float]:
    xs = [p.x for p in corner.points]
    ys = [p.y for p in corner.points]
    return min(xs), max(xs), min(ys), max(ys)


def clip_node_polygon_by_links(node_points: list[Point], connected_links: list[Link]) -> list[Point]:
    if not node_points or len(node_points) < 3:
        return node_points

    node_poly = ShapelyPolygon([(p.x, p.y) for p in node_points])
    if not node_poly.is_valid:
        node_poly = node_poly.buffer(0)

    link_polys = []
    for link in connected_links:
        c = link.corner
        pts = [(p.x, p.y) for p in c.points]
        link_poly = ShapelyPolygon(pts)
        if link_poly.is_valid:
            link_polys.append(link_poly)

    if not link_polys:
        return node_points

    links_union = unary_union(link_polys)
    trimmed_poly = node_poly.difference(links_union)

    if trimmed_poly.is_empty:
        raise ValueError("Node polygon completely trimmed by connected links.")

    if trimmed_poly.geom_type == "Polygon":
        return [Point(x, y) for x, y in trimmed_poly.exterior.coords[:-1]]

    elif trimmed_poly.geom_type == "MultiPolygon":
        largest_poly = max(trimmed_poly.geoms, key=lambda p: p.area)
        return [Point(x, y) for x, y in largest_poly.exterior.coords[:-1]]


def cross_product(o: Point, a: Point, b: Point) -> float:
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

def get_convex_hull(points: list[Point]) -> list[Point]:
    n = len(points)
    if n <= 2:
        return points
    
    points.sort(key=lambda p: (p.x, p.y))

    upper = []
    for p in points:
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    lower = []
    for p in reversed(points):
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    return upper[:-1] + lower[:-1]