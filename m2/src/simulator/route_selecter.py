import heapq
from collections import Counter

import numpy as np

from .map_constructor import GridMap, MipMap


class BaseRouteGenerator:
    def route_generate(self, start_id: int | None = None, goal_id: int | None = None) -> list[int]:
        raise NotImplementedError

class GridMapRouteGenerator(BaseRouteGenerator):
    def __init__(self, x_num, y_num) -> None:
        self.x_num = x_num
        self.y_num = y_num
        self.node_list = []
        self.outer_node_list = []

        for i in range(self.x_num):
            for j in range(self.y_num):
                current_index = i * self.y_num + j + 1
                self.node_list.append(current_index)
                if i == 0 or i == self.x_num - 1 or j == 0 or j == self.y_num - 1:
                    self.outer_node_list.append(current_index)

    def route_generate(self, start_id: int | None = None, goal_id: int | None = None) -> list[int]:
        start = start_id if start_id is not None else np.random.choice(self.outer_node_list)
        goal = goal_id if goal_id is not None else np.random.choice(self.outer_node_list)
        while (start == goal) or abs(start - goal) == 1 or abs(start - goal) == self.y_num:
            start = np.random.choice(self.outer_node_list)
            goal = np.random.choice(self.outer_node_list)

        start_coord = [(start - 1) // self.y_num + 1, (start - 1) % self.y_num + 1]
        goal_coord = [(goal - 1) // self.y_num + 1, (goal - 1) % self.y_num + 1]
        move_amount = [goal_coord[0] - start_coord[0], goal_coord[1] - start_coord[1]]
        move_list = []
        if move_amount[0] > 0:
            move_list += ["up"] * move_amount[0]
        elif move_amount[0] < 0:
            move_list += ["down"] * abs(move_amount[0])
        if move_amount[1] > 0:
            move_list += ["right"] * move_amount[1]
        elif move_amount[1] < 0:
            move_list += ["left"] * abs(move_amount[1])

        route_list = [start]
        while len(move_list) > 0:
            current_node = route_list[-1]
            rondom_move = np.random.choice(move_list)
            if rondom_move == "up":
                next_node = current_node + self.y_num
            elif rondom_move == "down":
                next_node = current_node - self.y_num
            elif rondom_move == "right":
                next_node = current_node + 1
            elif rondom_move == "left":
                next_node = current_node - 1
            move_list.remove(rondom_move)
            route_list.append(next_node)

        # convert np.int to int
        route_list = [int(node) for node in route_list]
        return route_list


class MipMapRouteGenerator(BaseRouteGenerator):
    def __init__(self, map: MipMap):
        self.map = map
        self.outer_node_list = [4, 5, 6, 7, 8, 11, 13, 18, 20, 21, 22, 23, 24, 25]
        self.link_info_list: list[list[int, int, float]] = []  # [node_id1, node_id2, link_length]
        for link_id, link in self.map.links.items():
            link_length = np.sqrt(
                (link.entrance_point.x - link.exit_point.x) ** 2 + (link.entrance_point.y - link.exit_point.y) ** 2
            )
            self.link_info_list.append([link_id[0], link_id[1], link_length])

    def dijkstra(self, start_node_id, goal_node_id):
        graph = {}
        for node1, node2, length in self.link_info_list:
            if node1 not in graph:
                graph[node1] = {}
            if node2 not in graph:
                graph[node2] = {}
            graph[node1][node2] = length
            graph[node2][node1] = length

        distances = {node: float("infinity") for node in graph}
        previous_nodes = {node: None for node in graph}
        distances[start_node_id] = 0

        nodes = [(0, start_node_id)]
        while nodes:
            current_distance, current_node = heapq.heappop(nodes)

            if current_node == goal_node_id:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = previous_nodes[current_node]
                return path[::-1], distances[goal_node_id]

            if distances[current_node] < current_distance:
                continue

            for neighbor, weight in graph[current_node].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(nodes, (distance, neighbor))

        return [], float("infinity")

    def route_generate(self, start_id: int | None = None, goal_id: int | None = None) -> list[int]:
        start = start_id if start_id is not None else np.random.choice(self.outer_node_list)
        goal = goal_id if goal_id is not None else np.random.choice(self.outer_node_list)
        route_list, _ = self.dijkstra(start, goal)
        while len(route_list) < 3:
            start = start_id if start_id is not None else np.random.choice(self.outer_node_list)
            goal = goal_id if goal_id is not None else np.random.choice(self.outer_node_list)
            route_list, _ = self.dijkstra(start, goal)

        route_list = [int(node) for node in route_list]
        return route_list


def get_all_link_idx(route_list: list[int]) -> tuple[list[tuple[int, int]], list[bool]]:
    link_idx_list = []
    link_s2j_list = []
    for i in range(len(route_list) - 1):
        start_idx = route_list[i]
        end_idx = route_list[i + 1]
        if start_idx < end_idx:
            link_idx_list.append((start_idx, end_idx))
            link_s2j_list.append(True)
        else:
            link_idx_list.append((end_idx, start_idx))
            link_s2j_list.append(False)
    assert len(link_idx_list) == len(set(link_idx_list))
    assert len(link_idx_list) == len(route_list) - 1
    return link_idx_list, link_s2j_list

def build_route_generator(map: GridMap | MipMap) -> BaseRouteGenerator:
    if isinstance(map, GridMap):
        return GridMapRouteGenerator(map.x_num, map.y_num)
    elif isinstance(map, MipMap):
        return MipMapRouteGenerator(map)
    else:
        raise ValueError("Unsupported map type for route generator.")


if __name__ == "__main__":
    # r = GridMapRouteGenerator(3, 3)
    np.random.seed(0)
    import random

    random.seed(0)
    r = MipMapRouteGenerator(MipMap(config_dir_path="core/config/kashiwa"))
    aggretate_target_node_list = []
    for _ in range(100000):
        route_list, target_node = r.route_generate()
        aggretate_target_node_list.append(target_node)
    counter = Counter(aggretate_target_node_list)
    print(counter)
