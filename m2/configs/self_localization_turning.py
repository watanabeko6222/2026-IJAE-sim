from src.filter.sensor import GNSS, LinkInfoSensor, WheelVel
from src.simulator.map_constructor import GridMap
from src.trial.base_vehicle_run import SimulationConfig

grid_map = GridMap(link_length=100, link_width=5.5, x_num=4, y_num=3, rsu_node_ids=[])
cfg = SimulationConfig(
    seed=2,
    map=grid_map,
    ego_infos=[
        GNSS(pos_std=4.25, fps=1),
        WheelVel(vel_std=0.28, fps=10),
        LinkInfoSensor(map=grid_map, fps=1, std_denominator=1),
    ],
    node_id_list=[3, 2, 5, 4, 7, 10],
)
