from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


plt.style.use("fast")
plt.rcParams["figure.figsize"] = [6.4, 4.0]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.autolayout"] = False
plt.rcParams["figure.subplot.left"] = 0.14
plt.rcParams["figure.subplot.bottom"] = 0.14
plt.rcParams["figure.subplot.right"] = 0.90
plt.rcParams["figure.subplot.top"] = 0.91
plt.rcParams["figure.subplot.wspace"] = 0.20
plt.rcParams["figure.subplot.hspace"] = 0.20

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 1
plt.rcParams["grid.color"] = "black"

plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.fancybox"] = False


@dataclass
class Point:
    x: float
    y: float


@dataclass
class TP_State:
    x: float
    y: float
    theta: float
    vel: float
    accel: float = 0.0
    yaw_rate: float = 0.0
    in_node_idx: int = -1
    in_link_idx: tuple[int, int] = (-1, -1)


@dataclass(init=False)
class StraightLine:
    start: Point
    goal: Point
    theta: float
    length: float
    in_node_idx: int = -1
    in_link_idx: tuple[int, int] = (-1, -1)

    def __init__(self, start: Point, goal: Point, in_node_idx: int = -1, in_link_idx: tuple[int, int] = (-1, -1)):
        self.start = start
        self.goal = goal
        self.theta = np.arctan2(goal.y - start.y, goal.x - start.x)
        self.length = np.linalg.norm(np.array([start.x, start.y]) - np.array([goal.x, goal.y]))
        self.in_node_idx = in_node_idx
        self.in_link_idx = in_link_idx

    def calc_point_from_length(self, length: float):
        assert length >= 0, "length should be positive"
        assert length <= self.length, "length should be smaller than line length"
        return TP_State(
            self.start.x + length * np.cos(self.theta), self.start.y + length * np.sin(self.theta), self.theta, None, None, 0
        )

class MovingSimulatorLine:
    def __init__(
        self,
        dt: float,
        init_state: TP_State,
        moving_line: StraightLine,
        desired_vel: float,
        desired_accel: float,
        desired_decel: float,
    ):
        self.dt = dt
        self.map = map
        self.init_point = Point(init_state.x, init_state.y)
        self.init_state = init_state
        self.state_list = [init_state]
        self.moving_line = moving_line
        
        self.desired_vel = desired_vel
        self.desired_accel = desired_accel
        self.desired_decel = desired_decel
        
        self.generate_state_timestamp()

    def generate_state_timestamp(self):
        max_offset = self.moving_line.length
        now_offset = 1e-17
        last_X = self.state_list[-1]
        while now_offset + self.dt * last_X.vel < max_offset:
            now_offset += self.dt * last_X.vel + 0.5 * last_X.accel * self.dt ** 2
            
            now_X = self.moving_line.calc_point_from_length(now_offset)
            
            # calc accel # now in node
            if last_X.vel < self.desired_vel:
                acc_in_dt = (self.desired_vel - last_X.vel) / self.dt
                if acc_in_dt > self.desired_accel:
                    acc = self.desired_accel
                else:
                    acc = acc_in_dt
            elif last_X.vel > self.desired_vel:
                decel_in_dt = (last_X.vel - self.desired_vel) / self.dt
                if decel_in_dt > self.desired_decel:
                    acc = -self.desired_decel
                else:
                    acc = -decel_in_dt
            else:
                acc = 0
            
            now_X.accel = acc
            now_X.yaw_rate = 0
            now_X.vel = last_X.vel + acc * self.dt
            self.state_list.append(now_X)
            last_X = now_X

    def __len__(self):
        return len(self.state_list)
    
    def get_state(self, idx: int):
        return self.state_list[idx]
    
    def plot_xy(self):
        plt.figure(figsize=(10, 10))
        ax = plt.axes()
        ax.set_title("XY")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.scatter([X.x for X in self.state_list], [X.y for X in self.state_list])
        plt.savefig("xy.png")
    
    def plot_vel(self):
        plt.figure(figsize=(10, 10))
        ax = plt.axes()
        ax.set_title("Velocity")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("velocity [m/s]")
        plt.plot([self.dt * i for i in range(len(self.state_list))], [X.vel for X in self.state_list])
        plt.savefig("vel.png")

if __name__ == "__main__":
    init_state = TP_State(0, 0, 0, 0)
    moving_line = StraightLine(Point(0, 0), Point(10, 0))
    simulator = MovingSimulatorLine(0.1, init_state, moving_line, 1.0, 0.5, 0.5)
    simulator.plot_xy()
    simulator.plot_vel()
