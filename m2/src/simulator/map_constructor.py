import json
import os
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from .enu_transformation import ENU
from .map_component import (
    Corner,
    Link,
    Node,
    Point,
    RoadSideUnit,
    clip_node_polygon_by_links,
    corner2bbox,
    get_convex_hull,
    merge_close_points,
)


@dataclass
class BaseMap:
    links: list[Link]
    nodes: list[Node]
    id2link: dict[tuple[int, int], Link]
    id2node: dict[int, Node]
    base_waypoints: np.ndarray  # shape (N, 2)
    base_waypoints_labels: list[tuple[int, int]]
    node_centers: np.ndarray  # shape (M, 2)
    link_width: float
    _node_corner_cache: list[tuple[int, Corner, tuple[float, float, float, float]]] = field(
        init=False, default_factory=list
    )
    _link_corner_cache: list[tuple[tuple[int, int], Corner, tuple[float, float, float, float]]] = field(
        init=False, default_factory=list
    )

    def build_spatial_index(self) -> None:
        # Precompute bounding boxes for nodes/links to cheaply cull non-overlapping shapes.
        self._node_corner_cache = [(node.id, node.corner, corner2bbox(node.corner)) for node in self.nodes]
        self._link_corner_cache = [(link.id, link.corner, corner2bbox(link.corner)) for link in self.links]

    def is_node_visible_from_node(self, from_node_id: int, to_node_id: int) -> bool:
        raise NotImplementedError

    def is_link_visible_from_node(self, from_node_id: int, to_link_id: tuple[int, int]) -> bool:
        raise NotImplementedError

    def get_node_by_id(self, node_id: int) -> Node:
        return self.id2node[node_id]

    def get_link_by_id(self, link_id: tuple[int, int]) -> Link:
        return self.id2link[link_id]

    def get_neighbor_nodes(self, node_id: int) -> list[Node]:
        neighbor_nodes: list[Node] = []
        for link in self.links:
            link: Link
            if link.senior_node.id == node_id:
                neighbor_nodes.append(link.junior_node)
            elif link.junior_node.id == node_id:
                neighbor_nodes.append(link.senior_node)
        return neighbor_nodes

    def check_ellipse_node_intersection(
        self, node_id: int, mean: np.ndarray, cov: np.ndarray, alpha: float = 0.95
    ) -> bool:
        """Return True if the rectangle node intersects the covariance ellipse."""
        node = self.nodes[node_id]
        corner = node.corner
        return corner.check_ellipse_intersection(mean, cov, alpha)

    def check_ellipse_link_intersection(
        self,
        link_id: tuple[int, int],
        mean: np.ndarray,
        cov: np.ndarray,
        alpha: float = 0.95,
    ) -> bool:
        """Return True if the rectangle link intersects the covariance ellipse."""
        link = self.get_link_by_id(link_id)
        corner = link.corner
        return corner.check_ellipse_intersection(mean, cov, alpha)

    def pred_in_only_link(self, mean: np.ndarray, cov: np.ndarray) -> tuple[int, int] | None:
        """Find the link where the predicted position is in
        Arguments:
            mean: np.ndarray, shape (2,), predicted position mean
            cov: np.ndarray, shape (2, 2), predicted position covariance
        """
        intersected_links: list[Link] = []
        for link in self.links:
            link: Link
            if link.corner.check_ellipse_intersection(mean, cov):
                intersected_links.append(link)
        if len(intersected_links) != 1:
            return None

        return intersected_links[0].id

    def pred_in_only_node(self, mean: np.ndarray, cov: np.ndarray) -> int | None:
        """Find the node where the predicted position is in
        Arguments:
            mean: np.ndarray, shape (2,), predicted position mean
            cov: np.ndarray, shape (2, 2), predicted position covariance
        """
        intersected_nodes: list[Node] = []

        for node in self.nodes:
            if node.corner.check_ellipse_intersection(mean, cov):
                intersected_nodes.append(node)
        if len(intersected_nodes) != 1:
            return None

        return intersected_nodes[0].id

    def viz_background_map(self, ax: plt.Axes, detail_flag: bool = False) -> plt.Axes:
        for link in self.links:
            link: Link
            r = link.corner.viz_poly
            r = copy(r)
            r.set_color("gray")
            ax.add_patch(r)
            if detail_flag:
                exit_points = [
                    link.get_exit_point(distance_from_wall=dist, side=side)
                    for (dist, side) in [(1.3, "left"), (1.3, "right")]
                ]
                entrance_points = [
                    link.get_entrance_point(distance_from_wall=dist, side=side)
                    for (dist, side) in [(1.3, "left"), (1.3, "right")]
                ]
                exit_points += [link.exit_point]
                entrance_points += [link.entrance_point]
                for entrance_point, exit_point in zip(entrance_points, exit_points):
                    plt.plot(
                        [entrance_point.x, exit_point.x],
                        [entrance_point.y, exit_point.y],
                        color="black",
                        marker="o",
                        markersize=1,
                        linestyle="dotted",
                    )
        for node in self.nodes:
            node: Node
            r = node.corner.viz_poly
            r.set_color("gray")
            r = copy(r)
            ax.add_patch(r)

            if detail_flag:
                gateways = node.gateways
                for gateway in gateways:
                    plt.scatter(
                        gateway.point.x,
                        gateway.point.y,
                        color="red",
                        marker="*",
                    )

            if node.rsu is not None:
                plt.scatter(node.center.x, node.center.y, color="tab:green", s=20, marker="s")
                circle = plt.Circle(
                    (node.center.x, node.center.y),
                    node.rsu.range,
                    color="tab:green",
                    fill=True,
                    linestyle="dashed",
                    alpha=0.5,
                )
                ax.add_patch(circle)

        return ax

    def visualize(self, detail_flag: bool = False) -> None:
        plt.figure(figsize=(20, 20))
        ax = plt.axes()
        plt.gca().set_aspect("equal", adjustable="datalim")
        ax = self.viz_background_map(ax, detail_flag=detail_flag)
        # plot base waypoints
        # ax.scatter(self.base_waypoints[:, 0], self.base_waypoints[:, 1], color="green", s=1)
        # put node_id
        for node in self.nodes:
            ax.text(node.center.x, node.center.y, str(node.id), color="blue", fontsize=8)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        plt.savefig(f"{self.__class__.__name__}.png", dpi=300)
        plt.show()
        plt.close()


class GridMap(BaseMap):
    def __init__(
        self,
        link_length: float,
        link_width: float,
        x_num: int,
        y_num: int,
        rsu_node_ids: list[int] = [],
    ) -> None:
        self.link_length = link_length
        self.link_width = link_width
        self.x_num = x_num
        self.y_num = y_num
        self.rsu_node_ids = rsu_node_ids
        self.links: list[Link] = []
        self.nodes: list[Node] = []

        # add nodes
        for i in range(self.x_num):
            for j in range(self.y_num):
                current_index = i * self.y_num + j + 1
                center = Point(i * self.link_length, j * self.link_length)
                corner = Corner(
                    points=[
                        Point(
                            i * self.link_length - self.link_width / 2,
                            j * self.link_length - self.link_width / 2,
                        ),
                        Point(
                            i * self.link_length - self.link_width / 2,
                            j * self.link_length + self.link_width / 2,
                        ),
                        Point(
                            i * self.link_length + self.link_width / 2,
                            j * self.link_length + self.link_width / 2,
                        ),
                        Point(
                            i * self.link_length + self.link_width / 2,
                            j * self.link_length - self.link_width / 2,
                        ),
                    ]
                )
                current_node = Node(
                    id=current_index,
                    center=center,
                    corner=corner,
                    gateways=[],
                    connected_node_ids=[],
                    connected_link_ids=[],
                )
                if current_index in self.rsu_node_ids:
                    current_node.rsu = RoadSideUnit(id=current_index, range=10)
                self.nodes.append(current_node)

        self.id2node = {node.id: node for node in self.nodes}

        # add links
        for i in range(self.x_num):
            for j in range(self.y_num):
                current_index = i * self.y_num + j + 1
                current_node = self.id2node[current_index]
                # add link to the upper node
                if j < self.y_num - 1:
                    upper_node = self.id2node[current_index + 1]
                    self.links.append(Link(current_node, upper_node, width=self.link_width))
                # add link to the right node
                if i < self.x_num - 1:
                    right_node = self.id2node[current_index + self.y_num]
                    self.links.append(
                        Link(
                            current_node,
                            right_node,
                            width=self.link_width,
                        )
                    )

        # generate all base_waypoints
        self.base_waypoints: list[Point] = []
        self.base_waypoints_labels: list[tuple[int, int]] = []
        for link in self.links:
            link: Link
            self.base_waypoints += link.base_waypoints
            self.base_waypoints_labels += [link.id] * len(link.base_waypoints)
        self.base_waypoints = np.array([[point.x, point.y] for point in self.base_waypoints])

        # generate all node centers
        self.node_centers: list[Point] = []
        for node in self.nodes:
            node: Node
            self.node_centers.append(node.center)
        self.node_centers = np.array([[point.x, point.y] for point in self.node_centers])

        self.id2link = {link.id: link for link in self.links}
        self.build_spatial_index()

    def is_node_visible_from_node(self, from_node_id: int, to_node_id: int) -> bool:
        """Check if the node is visible from the node"""
        from_node = self.id2node[from_node_id]
        to_node = self.id2node[to_node_id]
        if from_node.center.x == to_node.center.x or from_node.center.y == to_node.center.y:
            return True
        else:
            return False

    def is_link_visible_from_node(self, from_node_id: int, to_link_id: tuple[int, int]) -> bool:
        """Check if the link is visible from the node"""
        from_node = self.id2node[from_node_id]
        to_link = self.id2link[to_link_id]
        if from_node.center.x == to_link.entrance_point.x or from_node.center.y == to_link.entrance_point.y:
            return True
        else:
            return False


class MipMap(BaseMap):
    def __init__(
        self,
        config_dir_path: str,
        link_width: float = 5.5,
        rsu_node_ids: list[int] = [],
    ) -> None:
        self.config_dir_path = config_dir_path
        self.link_width = link_width
        self.rsu_node_ids = rsu_node_ids

        orig_pts_path = os.path.join(config_dir_path, "origin_point.csv")
        with open(orig_pts_path, "r") as f:
            orig_pts = map(float, f.readline().split(","))  # lon, lat, alt, geoid_height
        enu_transformer = ENU(*orig_pts)

        # add nodes
        self.nodes: list[Node] = []
        node_pts_path = os.path.join(config_dir_path, "node_position.csv")
        with open(node_pts_path, "r") as f:
            for line in f.readlines():
                node_id, lon, lat, alt, other = line.split(",")
                node_id = int(node_id)
                lon = float(lon)
                lat = float(lat)
                alt = float(alt)
                x, y, z = enu_transformer.calc_enu_based_on_altitude_including_geoid_height(lon, lat, alt)
                center = Point(x, y)
                # temporary
                corner = Corner(
                    points=[
                        Point(x - self.link_width / 2, y - self.link_width / 2),
                        Point(x - self.link_width / 2, y + self.link_width / 2),
                        Point(x + self.link_width / 2, y + self.link_width / 2),
                        Point(x + self.link_width / 2, y - self.link_width / 2),
                    ]
                )
                current_node = Node(
                    id=node_id, center=center, corner=corner, gateways=[], connected_node_ids=[], connected_link_ids=[]
                )
                if node_id in self.rsu_node_ids:
                    current_node.rsu = RoadSideUnit(id=node_id, range=10)
                self.nodes.append(current_node)
        self.id2node = {node.id: node for node in self.nodes}

        # add links
        self.links: list[Link] = []
        link_pts_path = os.path.join(config_dir_path, "link_info.csv")
        with open(link_pts_path, "r") as f:
            for line in f.readlines():
                link_id, senior_node_id, junior_node_id, *others = line.split(",")
                link_id = int(link_id)
                senior_node_id = int(senior_node_id)
                junior_node_id = int(junior_node_id)

                tmp_theta = np.arctan2(
                    self.id2node[junior_node_id].center.y - self.id2node[senior_node_id].center.y,
                    self.id2node[junior_node_id].center.x - self.id2node[senior_node_id].center.x,
                )

                for node_id in [senior_node_id, junior_node_id]:
                    old_node = self.id2node[node_id]
                    c1 = Point(
                        old_node.center.x + np.cos(tmp_theta + np.pi / 4) * (self.link_width / 2) * np.sqrt(2),
                        old_node.center.y + np.sin(tmp_theta + np.pi / 4) * (self.link_width / 2) * np.sqrt(2),
                    )
                    c2 = Point(
                        old_node.center.x
                        + np.cos(tmp_theta + np.pi / 2 + np.pi / 4) * (self.link_width / 2) * np.sqrt(2),
                        old_node.center.y
                        + np.sin(tmp_theta + np.pi / 2 + np.pi / 4) * (self.link_width / 2) * np.sqrt(2),
                    )
                    c3 = Point(
                        old_node.center.x + np.cos(tmp_theta + np.pi + np.pi / 4) * (self.link_width / 2) * np.sqrt(2),
                        old_node.center.y + np.sin(tmp_theta + np.pi + np.pi / 4) * (self.link_width / 2) * np.sqrt(2),
                    )
                    c4 = Point(
                        old_node.center.x
                        + np.cos(tmp_theta + np.pi * 3 / 2 + np.pi / 4) * (self.link_width / 2) * np.sqrt(2),
                        old_node.center.y
                        + np.sin(tmp_theta + np.pi * 3 / 2 + np.pi / 4) * (self.link_width / 2) * np.sqrt(2),
                    )

                    tmp_corner = [c1, c2, c3, c4]
                    # sort corners as clockwise
                    tmp_corner.sort(key=lambda x: np.arctan2(x.y - old_node.center.y, x.x - old_node.center.x))
                    # update node corner
                    self.id2node[node_id].corner = Corner(points=tmp_corner)

                self.links.append(
                    Link(
                        self.id2node[senior_node_id],
                        self.id2node[junior_node_id],
                        width=self.link_width,
                    )
                )

        node_points: dict[int, list[Point]] = defaultdict(list)
        for link in self.links:
            senior_node = link.senior_node
            junior_node = link.junior_node
            link_center = Point(
                (senior_node.center.x + junior_node.center.x) / 2,
                (senior_node.center.y + junior_node.center.y) / 2,
            )
            center2seior_vec = np.array(
                [
                    senior_node.center.x - link_center.x,
                    senior_node.center.y - link_center.y,
                ]
            )
            center2junior_vec = np.array(
                [
                    junior_node.center.x - link_center.x,
                    junior_node.center.y - link_center.y,
                ]
            )
            for point in link.corner.points:
                vec = np.array([point.x - link_center.x, point.y - link_center.y])
                if np.dot(vec, center2seior_vec) > 0:
                    node_points[senior_node.id].append(point)
                elif np.dot(vec, center2junior_vec) > 0:
                    node_points[junior_node.id].append(point)

        for node in self.nodes:
            points = node_points[node.id]
            if not points:
                continue

            merged_points = merge_close_points(points, threshold=0.5)
            hull_points = get_convex_hull(merged_points)
            connected_links = []
            for link in self.links:
                if link.senior_node.id == node.id or link.junior_node.id == node.id:
                    connected_links.append(link)
            final_points = clip_node_polygon_by_links(hull_points, connected_links)
            node.corner = Corner(points=final_points)
            self.id2node[node.id] = node

        self.nodes = list(self.id2node.values())
        self.id2link = {link.id: link for link in self.links}

        # read visibility info
        visibility_path = os.path.join(config_dir_path, "visible_info.json")
        with open(visibility_path, "r") as f:
            visibility_info = json.load(f)
            self.visibility_info_list = visibility_info["info"]

        # generate all base_waypoints
        self.base_waypoints: list[Point] = []
        self.base_waypoints_labels: list[tuple[int, int]] = []
        for link in self.links:
            self.base_waypoints += link.base_waypoints
            self.base_waypoints_labels += [link.id] * len(link.base_waypoints)
        self.base_waypoints = np.array([[point.x, point.y] for point in self.base_waypoints])

        # generate all node centers
        self.node_centers: list[Point] = []
        self.node_centers_labels: list[int] = []
        for node in self.nodes:
            self.node_centers.append(node.center)
            self.node_centers_labels.append(node.id)
        self.node_centers = np.array([[point.x, point.y] for point in self.node_centers])

        self.build_spatial_index()

    def is_node_visible_from_node(self, from_node_id: int, to_node_id: int) -> bool:
        """Check if the node is visible from the node"""
        for visible_info in self.visibility_info_list:
            if visible_info["node_id"] == from_node_id:
                return to_node_id in visible_info["visible_node_ids"]

        raise ValueError(f"from_node_id={from_node_id} is not found in visibility_info_list")

    def is_link_visible_from_node(self, from_node_id: int, to_link_id: tuple[int, int]) -> bool:
        """Check if the link is visible from the node"""
        to_link_id = list(to_link_id)
        for visible_info in self.visibility_info_list:
            if visible_info["node_id"] == from_node_id:
                return to_link_id in visible_info["visible_link_ids"]

        raise ValueError(f"from_node_id={from_node_id} is not found in visibility_info_list")


if __name__ == "__main__":
    from ..plot_utils import setup_matplotlib

    setup_matplotlib()

    grid_map = GridMap(100.0, 5.5, 4, 4, rsu_node_ids=[1, 4, 13, 16])

    import ipdb

    ipdb.set_trace()
    grid_map.visualize()

    # kashiwa_map = MipMap(config_dir_path="map_info/kashiwa", link_width=5.5)
    # kashiwa_map.visualize(detail_flag=True)
