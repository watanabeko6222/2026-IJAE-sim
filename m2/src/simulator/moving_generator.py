import bisect
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from .map_constructor import BaseMap, Corner, Link, Node, Point, corner2bbox


@dataclass
class TP_State:
    x: float
    y: float
    theta: float
    vel: float
    accel: float = 0.0
    yaw_rate: float = 0.0
    in_node_id: int | None = None
    in_link_id: tuple[int, int] | None = None
    intersect_node_ids: list[int] = field(default_factory=list)
    intersect_link_ids: list[tuple[int, int]] = field(default_factory=list)
    next_node_id: int | None = None
    can_do_observe: bool = True
    can_be_observed: bool = True


@dataclass(init=False)
class StraightLine:
    start: Point
    goal: Point
    theta: float
    length: float
    in_node_id: int | None = None
    in_link_id: tuple[int, int] | None = None
    next_node_id: int | None = None
    can_do_observe: bool = True
    can_be_observed: bool = True

    def __init__(
        self,
        start: Point,
        goal: Point,
        in_node_id: int | None = None,
        in_link_id: tuple[int, int] | None = None,
        next_node_id: int | None = None,
        can_do_observe: bool = True,
        can_be_observed: bool = True,
    ):
        self.start = start
        self.goal = goal
        self.theta = np.arctan2(goal.y - start.y, goal.x - start.x)
        self.length = np.linalg.norm(np.array([start.x, start.y]) - np.array([goal.x, goal.y]))
        self.in_node_id = in_node_id
        self.in_link_id = in_link_id
        self.next_node_id = next_node_id
        self.can_do_observe = can_do_observe
        self.can_be_observed = can_be_observed

    def calc_point_from_length(self, length: float):
        assert length >= 0, "length should be positive"
        assert length <= self.length, "length should be smaller than line length"
        return TP_State(
            x=self.start.x + length * np.cos(self.theta),
            y=self.start.y + length * np.sin(self.theta),
            theta=self.theta,
            vel=None,
            accel=None,
            yaw_rate=0,
            can_do_observe=self.can_do_observe,
            can_be_observed=self.can_be_observed,
        )


@dataclass(init=False)
class Arc:
    start: Point
    theta1: float
    goal: Point
    theta2: float
    center: Point
    radius: float
    length: float
    in_node_id: int | None = None
    in_link_id: tuple[int, int] | None = None
    next_node_id: int | None = None

    def __init__(
        self,
        start: Point,
        goal: Point,
        center: Point,
        in_node_id: int | None = None,
        in_link_id: tuple[int, int] | None = None,
        next_node_id: int | None = None,
    ):
        self.start = start
        self.goal = goal

        self.center = center
        self.theta1 = np.arctan2(start.y - self.center.y, start.x - self.center.x)
        self.theta2 = np.arctan2(goal.y - self.center.y, goal.x - self.center.x)
        self.radius = np.linalg.norm(np.array([start.x, start.y]) - np.array([self.center.x, self.center.y]))
        self.length = self.radius * (abs(self.theta2 - self.theta1) % np.pi)
        self.in_node_id = in_node_id
        self.in_link_id = in_link_id
        self.next_node_id = next_node_id

    def calc_point_from_angle(self, delta_theta: float):
        assert delta_theta >= 0, "delta_theta should be positive"
        assert delta_theta <= abs(self.theta2 - self.theta1) % np.pi, (
            "delta_theta should be smaller than abs(theta2 - theta1)"
        )
        if abs(self.theta2 - self.theta1) < np.pi:
            if self.theta1 < self.theta2:
                theta = self.theta1 + delta_theta
                tp_theta = theta + np.pi / 2
                return TP_State(
                    self.center.x + self.radius * np.cos(theta),
                    self.center.y + self.radius * np.sin(theta),
                    tp_theta,
                    None,
                    accel=None,
                    yaw_rate=1 / self.radius,
                )
            else:
                theta = self.theta1 - delta_theta
                tp_theta = theta - np.pi / 2
                return TP_State(
                    x=self.center.x + self.radius * np.cos(theta),
                    y=self.center.y + self.radius * np.sin(theta),
                    theta=tp_theta,
                    vel=None,
                    accel=None,
                    yaw_rate=-1 / self.radius,
                )
        else:
            if self.theta1 < self.theta2:
                theta = self.theta1 - delta_theta
                tp_theta = theta - np.pi / 2
                return TP_State(
                    self.center.x + self.radius * np.cos(theta),
                    self.center.y + self.radius * np.sin(theta),
                    tp_theta,
                    None,
                    accel=None,
                    yaw_rate=-1 / self.radius,
                )
            else:
                theta = self.theta1 + delta_theta
                tp_theta = theta + np.pi / 2
                return TP_State(
                    x=self.center.x + self.radius * np.cos(theta),
                    y=self.center.y + self.radius * np.sin(theta),
                    theta=tp_theta,
                    vel=None,
                    accel=None,
                    yaw_rate=1 / self.radius,
                )

    def calc_point_from_length(self, length: float):
        assert length >= 0, "length should be positive"
        assert length <= self.length, "length should be smaller than radius * abs(theta2 - theta1)"
        delta_theta = length / self.radius
        return self.calc_point_from_angle(delta_theta)


class TimeseriesTPState:
    def __init__(self, step_offset: int):
        self.step_offset = step_offset
        self.state_list: list[TP_State] = []

    def add_state(self, state: TP_State):
        self.state_list.append(state)

    def get_state(self, id: int) -> TP_State | None:
        if id - self.step_offset < 0 or id - self.step_offset >= len(self.state_list):
            return None
        else:
            return self.state_list[id - self.step_offset]

    def __len__(self):
        return len(self.state_list) + self.step_offset


@dataclass
class ControlCommandSeries:
    step_offset: int
    commands: list[float]  # list of acceleration commands


class MovingSimulator:
    def __init__(
        self,
        map: BaseMap,
        dt: float,
        node_id_list: list[int],
        tp_type: Literal["cyclist", "automobile"],
        desired_vel: float,
        step_offset: int = 0,
        init_progress_rate: float = 0.0,
        init_point: Point | None = None,
    ):
        self.dt = dt
        self.map = map
        self.node_id_list = node_id_list
        self.tp_type = tp_type
        self.step_offset = step_offset
        self.state_timeseries = TimeseriesTPState(step_offset=step_offset)
        self.offset_list: list[float] = []
        assert 0.0 <= init_progress_rate < 1.0, "init_progress_rate should be in [0.0, 1.0)"
        self.init_progress_rate = init_progress_rate
        self.init_point = init_point
        self._node_corner_cache: list[tuple[int, Corner, tuple[float, float, float, float]]] = []
        self._link_corner_cache: list[tuple[tuple[int, int], Corner, tuple[float, float, float, float]]] = []
        self.desired_vel = desired_vel

        if tp_type == "cyclist":
            self.width = 0.6
            self.length = 1.9

        elif tp_type == "automobile":
            self.width = 1.7
            self.length = 3.9
        else:
            raise ValueError("tp_type should be cyclist or automobile")
        self.generate_trajectory()
        self.generate_state_timestamp()

    def _bbox_intersects(
        self, bbox1: tuple[float, float, float, float], bbox2: tuple[float, float, float, float]
    ) -> bool:
        min_x1, max_x1, min_y1, max_y1 = bbox1
        min_x2, max_x2, min_y2, max_y2 = bbox2
        return not (max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1)

    def generate_trajectory(self):
        self.trajectory_list: list[StraightLine | Arc] = []
        self.trajectory_offset_list: list[float] = [0.0]
        last_goal = None
        last_angle = None
        last_senior2junior = None
        for i in range(len(self.node_id_list) - 1):
            current_node: Node = self.map.get_node_by_id(self.node_id_list[i])
            next_node: Node = self.map.get_node_by_id(self.node_id_list[i + 1])
            senior2junior = current_node.id < next_node.id

            # trajectory in node
            if i == 0:
                if self.init_point is not None:
                    if senior2junior:
                        selected_link: Link = self.map.get_link_by_id((current_node.id, next_node.id))
                        if self.tp_type == "automobile":
                            goal = selected_link.entrance_point
                        elif self.tp_type == "cyclist":
                            goal = selected_link.get_entrance_point(
                                distance_from_wall=1.0 + self.width / 2, side="left"
                            )
                    else:
                        selected_link: Link = self.map.get_link_by_id((next_node.id, current_node.id))
                        if self.tp_type == "automobile":
                            goal = selected_link.exit_point
                        elif self.tp_type == "cyclist":
                            goal = selected_link.get_exit_point(distance_from_wall=1.0 + self.width / 2, side="right")

                    if self.tp_type == "cyclist":
                        trajectory = StraightLine(
                            self.init_point,
                            goal,
                            next_node_id=current_node.id,
                            can_be_observed=False,
                        )
                    else:
                        trajectory = StraightLine(
                            self.init_point,
                            goal,
                            next_node_id=current_node.id,
                        )
                    self.trajectory_list.append(trajectory)
                    self.trajectory_offset_list.append(trajectory.length)
                else:
                    if senior2junior:
                        selected_link: Link = self.map.get_link_by_id((current_node.id, next_node.id))
                        if self.tp_type == "automobile":
                            goal = selected_link.entrance_point
                        elif self.tp_type == "cyclist":
                            goal = selected_link.get_entrance_point(
                                distance_from_wall=1.0 + self.width / 2, side="left"
                            )
                    else:
                        selected_link: Link = self.map.get_link_by_id((next_node.id, current_node.id))
                        if self.tp_type == "automobile":
                            goal = selected_link.exit_point
                        elif self.tp_type == "cyclist":
                            goal = selected_link.get_exit_point(distance_from_wall=1.0 + self.width / 2, side="right")

            else:
                if senior2junior:
                    start = last_goal
                    if self.tp_type == "automobile":
                        goal = self.map.get_link_by_id((current_node.id, next_node.id)).entrance_point
                    elif self.tp_type == "cyclist":
                        goal = self.map.get_link_by_id((current_node.id, next_node.id)).get_entrance_point(
                            distance_from_wall=1.0 + self.width / 2, side="left"
                        )
                else:
                    start = last_goal
                    if self.tp_type == "automobile":
                        goal = self.map.get_link_by_id((next_node.id, current_node.id)).exit_point
                    elif self.tp_type == "cyclist":
                        goal = self.map.get_link_by_id((next_node.id, current_node.id)).get_exit_point(
                            distance_from_wall=1.0 + self.width / 2, side="right"
                        )
                # check straight or curve
                angle = np.arctan2(goal.y - start.y, goal.x - start.x)
                if abs((angle % np.pi) - (last_angle % np.pi)) % np.pi < (10 / 180) * np.pi:
                    trajectory = StraightLine(
                        start,
                        goal,
                        in_node_id=current_node.id,
                        next_node_id=next_node.id,
                    )
                else:
                    center2start = np.array([start.x - current_node.center.x, start.y - current_node.center.y])
                    center2goal = np.array([goal.x - current_node.center.x, goal.y - current_node.center.y])
                    corner = current_node.corner
                    corner_points = corner.points
                    for corner_point in corner_points:
                        center2corner = np.array(
                            [corner_point.x - current_node.center.x, corner_point.y - current_node.center.y]
                        )
                        ip1 = np.dot(center2corner, center2start)
                        ip2 = np.dot(center2corner, center2goal)
                        if ip1 > 0 and ip2 > 0:
                            opposite_corner = corner_point
                            break
                    trajectory = Arc(
                        start,
                        goal,
                        center=opposite_corner,
                        in_node_id=current_node.id,
                        next_node_id=next_node.id,
                    )
                self.trajectory_list.append(trajectory)
                self.trajectory_offset_list.append(self.trajectory_offset_list[-1] + trajectory.length)

            # trajectory in link
            if senior2junior:
                if self.tp_type == "automobile":
                    start = self.map.get_link_by_id((current_node.id, next_node.id)).entrance_point
                    goal = self.map.get_link_by_id((current_node.id, next_node.id)).exit_point
                elif self.tp_type == "cyclist":
                    start = self.map.get_link_by_id((current_node.id, next_node.id)).get_entrance_point(
                        distance_from_wall=1.0 + self.width / 2, side="left"
                    )
                    goal = self.map.get_link_by_id((current_node.id, next_node.id)).get_exit_point(
                        distance_from_wall=1.0 + self.width / 2, side="left"
                    )
                trajectory = StraightLine(
                    start,
                    goal,
                    in_link_id=(current_node.id, next_node.id),
                    next_node_id=next_node.id,
                )
            else:
                if self.tp_type == "automobile":
                    start = self.map.get_link_by_id((next_node.id, current_node.id)).exit_point
                    goal = self.map.get_link_by_id((next_node.id, current_node.id)).entrance_point
                elif self.tp_type == "cyclist":
                    start = self.map.get_link_by_id((next_node.id, current_node.id)).get_exit_point(
                        distance_from_wall=1.0 + self.width / 2, side="right"
                    )
                    goal = self.map.get_link_by_id((next_node.id, current_node.id)).get_entrance_point(
                        distance_from_wall=1.0 + self.width / 2, side="right"
                    )

                trajectory = StraightLine(
                    start,
                    goal,
                    in_link_id=(next_node.id, current_node.id),
                    next_node_id=next_node.id,
                )

            self.trajectory_list.append(trajectory)
            if len(self.trajectory_offset_list) == 0:
                self.trajectory_offset_list.append(trajectory.length)
            else:
                self.trajectory_offset_list.append(self.trajectory_offset_list[-1] + trajectory.length)
            last_goal = goal
            last_angle = trajectory.theta

        # finally, add trajectory in last node
        last_vec = goal - start
        last_vec_norm = np.linalg.norm(np.array([last_vec.x, last_vec.y]))
        start = last_goal
        goal = Point(
            last_goal.x + (last_vec.x / last_vec_norm) * self.map.link_width * 2,
            last_goal.y + (last_vec.y / last_vec_norm) * self.map.link_width * 2,
        )

        trajectory = StraightLine(
            start,
            goal,
            in_node_id=self.node_id_list[-1],
        )
        self.trajectory_list.append(trajectory)
        self.trajectory_offset_list.append(self.trajectory_offset_list[-1] + trajectory.length)

        self.max_offset = self.trajectory_offset_list[-1]

    def generate_state_timestamp(self):
        now_offset = self.init_progress_rate * self.max_offset + 1e-10
        last_state = None

        while now_offset < self.max_offset:
            now_component_id = bisect.bisect_left(self.trajectory_offset_list, now_offset)
            if now_component_id > len(self.trajectory_list):
                break
            now_component = self.trajectory_list[now_component_id - 1]

            now_state = now_component.calc_point_from_length(
                now_offset - self.trajectory_offset_list[now_component_id - 1]
            )
            now_state: TP_State
            now_state.vel = self.desired_vel
            now_state.accel = 0
            if last_state is not None:
                if (now_state.theta - last_state.theta) > np.pi:
                    now_state.yaw_rate = (now_state.theta - last_state.theta - 2 * np.pi) / self.dt
                elif (now_state.theta - last_state.theta) < -np.pi:
                    now_state.yaw_rate = (now_state.theta - last_state.theta + 2 * np.pi) / self.dt
                else:
                    now_state.yaw_rate = (now_state.theta - last_state.theta) / self.dt

            if now_component.in_node_id is not None:
                now_state.in_node_id = now_component.in_node_id
                now_state.next_node_id = now_component.next_node_id
            elif now_component.in_link_id is not None:
                now_state.in_link_id = now_component.in_link_id
                now_state.next_node_id = now_component.next_node_id
            else:
                now_state.next_node_id = self.node_id_list[0]

            cx, cy = now_state.x, now_state.y
            local_vertex = [
                Point(self.length / 2, self.width / 2),
                Point(self.length / 2, -self.width / 2),
                Point(-self.length / 2, -self.width / 2),
                Point(-self.length / 2, self.width / 2),
            ]

            tp_vertex = []
            for l_v in local_vertex:
                x = cx + l_v.x * np.cos(now_state.theta) - l_v.y * np.sin(now_state.theta)
                y = cy + l_v.x * np.sin(now_state.theta) + l_v.y * np.cos(now_state.theta)
                tp_vertex.append(Point(x, y))
            tp_corner = Corner(points=[tp_vertex[2], tp_vertex[3], tp_vertex[0], tp_vertex[1]])
            tp_bbox = corner2bbox(tp_corner)
            for node_id, corner, bbox in self.map._node_corner_cache:
                if not self._bbox_intersects(tp_bbox, bbox):
                    continue
                if corner.check_corner_intersection(tp_corner):
                    now_state.intersect_node_ids.append(node_id)
            for link_id, corner, bbox in self.map._link_corner_cache:
                if not self._bbox_intersects(tp_bbox, bbox):
                    continue
                if corner.check_corner_intersection(tp_corner):
                    now_state.intersect_link_ids.append(link_id)

            self.state_timeseries.add_state(now_state)
            last_state = now_state
            self.offset_list.append(now_offset)
            now_offset += self.dt * self.desired_vel

        assert len(self.offset_list) == len(self.state_timeseries.state_list), (
            "length of offset_list and state_list should be the same"
        )


    def regenerate_state_with_control(self, control_command_series: ControlCommandSeries, steps_ahead: int = 50) -> bool:
        assert self.step_offset <= control_command_series.step_offset < len(self.state_timeseries), (
            "control_command_series.step_offset should be in [self.step_offset, len(self.state_timeseries))"
        )
        restart_idx = control_command_series.step_offset - self.step_offset
        acc_rate_list = control_command_series.commands.copy()
        now_offset = self.offset_list[restart_idx]
        now_state = self.state_timeseries.state_list[restart_idx]
        self.offset_list = self.offset_list[:restart_idx + 1] # contain now_offset
        self.state_timeseries.state_list = self.state_timeseries.state_list[:restart_idx + 1]  # contain now_state
        last_state = now_state

        done_flag = False
        for _ in range(steps_ahead):
            if len(acc_rate_list) == 0:
                accel_command = 0.0
            else:
                accel_command = acc_rate_list.pop(0)

            now_offset += now_state.vel * self.dt + 0.5 * accel_command * self.dt * self.dt
            if now_offset > self.max_offset:
                done_flag = True
                break

            now_component_id = bisect.bisect_left(self.trajectory_offset_list, now_offset)
            if now_component_id > len(self.trajectory_list):
                done_flag = True
                break
            now_component = self.trajectory_list[now_component_id - 1]

            # update position
            now_state = now_component.calc_point_from_length(
                now_offset - self.trajectory_offset_list[now_component_id - 1]
            )
            # update velocity
            now_state.vel = last_state.vel + accel_command * self.dt
            if now_state.vel < 0:
                now_state.vel = 0.0
            now_state.accel = accel_command
            if last_state is not None:
                if (now_state.theta - last_state.theta) > np.pi:
                    now_state.yaw_rate = (now_state.theta - last_state.theta - 2 * np.pi) / self.dt
                elif (now_state.theta - last_state.theta) < -np.pi:
                    now_state.yaw_rate = (now_state.theta - last_state.theta + 2 * np.pi) / self.dt
                else:
                    now_state.yaw_rate = (now_state.theta - last_state.theta) / self.dt

            if now_component.in_node_id is not None:
                now_state.in_node_id = now_component.in_node_id
                now_state.next_node_id = now_component.next_node_id
            elif now_component.in_link_id is not None:
                now_state.in_link_id = now_component.in_link_id
                now_state.next_node_id = now_component.next_node_id
            else:
                now_state.next_node_id = self.node_id_list[0]

            cx, cy = now_state.x, now_state.y
            local_vertex = [
                Point(self.length / 2, self.width / 2),
                Point(self.length / 2, -self.width / 2),
                Point(-self.length / 2, -self.width / 2),
                Point(-self.length / 2, self.width / 2),
            ]

            tp_vertex = []
            for l_v in local_vertex:
                x = cx + l_v.x * np.cos(now_state.theta) - l_v.y * np.sin(now_state.theta)
                y = cy + l_v.x * np.sin(now_state.theta) + l_v.y * np.cos(now_state.theta)
                tp_vertex.append(Point(x, y))
            tp_corner = Corner(tp_vertex[2], tp_vertex[3], tp_vertex[0], tp_vertex[1])
            tp_bbox = corner2bbox(tp_corner)
            for node_id, corner, bbox in self.map._node_corner_cache:
                if not self._bbox_intersects(tp_bbox, bbox):
                    continue
                if corner.check_corner_intersection(tp_corner):
                    now_state.intersect_node_ids.append(node_id)
            for link_id, corner, bbox in self.map._link_corner_cache:
                if not self._bbox_intersects(tp_bbox, bbox):
                    continue
                if corner.check_corner_intersection(tp_corner):
                    now_state.intersect_link_ids.append(link_id)

            self.state_timeseries.state_list.append(now_state)
            self.offset_list.append(now_offset)
            last_state = now_state

        assert len(self.offset_list) == len(self.state_timeseries.state_list), (
            "length of offset_list and state_list should be the same"
        )
        return done_flag

    def __len__(self):
        return len(self.state_timeseries)

    def get_state(self, id: int) -> TP_State | None:
        return self.state_timeseries.get_state(id)

    def get_closest_state(self, point: Point, threshold: float = 2.0) -> tuple[TP_State | None, int | None]:
        closest_state = None
        closest_dist = float("inf")
        closest_id = None
        for id, state in enumerate(self.state_timeseries.state_list):
            if state is None:
                continue
            if state.can_be_observed is False:
                continue
            dist = np.linalg.norm(np.array([point.x - state.x, point.y - state.y]))
            if dist < closest_dist:
                closest_dist = dist
                closest_state = state
                closest_id = id
        if closest_dist < threshold:
            return closest_state, closest_id + self.state_timeseries.step_offset
        else:
            return None, None

    def plot_xy(self, save_dir: Path, save_stem: str = "xy"):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal", adjustable="datalim")
        ax = self.map.viz_background_map(ax)
        try:
            ax.set_title(
                f"{self.map.y_num}x{self.map.x_num} Grid Map\nLink Length={self.map.link_length}, Link Width={self.map.link_width}"
            )
        except AttributeError:
            ax.set_title("Mip Map")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        # plot arrow
        for i in range(self.state_timeseries.step_offset, len(self.state_timeseries) - 1):
            X = self.get_state(i)
            next_X = self.get_state(i + 1)
            plt.arrow(
                X.x,
                X.y,
                (next_X.x - X.x) * 0.9,
                (next_X.y - X.y) * 0.9,
                head_width=0.1,
                head_length=0.1,
                fc="k",
                ec="k",
            )
        # plt.grid()
        plt.savefig(save_dir / f"{save_stem}.png")
        plt.close()

    def plot_vel(self, save_dir: Path, save_stem: str = "vel"):
        plt.figure(figsize=(10, 10))
        ax = plt.axes()
        ax.set_title("Velocity")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("velocity [m/s]")
        times = [self.dt * i for i in range(self.step_offset, len(self.state_timeseries))]
        plt.plot(
            times,
            [X.vel for X in self.state_timeseries.state_list],
        )
        plt.savefig(save_dir / f"{save_stem}.png")
        plt.close()

    def save_csv(self, file_path: str):
        """Save state history to CSV file
        Args:
            file_path: path to save the CSV file
        """
        import pandas as pd

        data = {
            "step": list(range(self.state_timeseries.step_offset, len(self.state_timeseries))),
            "x": [x.x for x in self.state_timeseries.state_list],
            "y": [x.y for x in self.state_timeseries.state_list],
            "theta": [x.theta for x in self.state_timeseries.state_list],
            "vel": [x.vel for x in self.state_timeseries.state_list],
            "accel": [x.accel for x in self.state_timeseries.state_list],
            "yaw_rate": [x.yaw_rate for x in self.state_timeseries.state_list],
            "in_node_id": [x.in_node_id for x in self.state_timeseries.state_list],
            "in_link_id_0": [x.in_link_id[0] for x in self.state_timeseries.state_list],
            "in_link_id_1": [x.in_link_id[1] for x in self.state_timeseries.state_list],
            "next_node_id": [x.next_node_id for x in self.state_timeseries.state_list],
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)


if __name__ == "__main__":
    from ..plot_utils import setup_matplotlib
    from .map_constructor import GridMap, MipMap

    setup_matplotlib()

    map = GridMap(100.0, 5.5, 3, 3)
    route = [1, 2, 5, 6]

    # map = MipMap(config_dir_path="./map_info/kashiwa")
    # route = [4, 5, 6, 15, 22, 23]

    tp_type = "cyclist"
    simulator = MovingSimulator(
        map=map,
        dt=0.1,
        node_id_list=route,
        tp_type=tp_type,
        desired_vel=15 / 3.6,
        # init_point=Point(-100, -100),
    )

    simulator.plot_xy(save_dir=Path("./"), save_stem="xy_before_control")
    simulator.plot_vel(save_dir=Path("./"), save_stem="vel_before_control")