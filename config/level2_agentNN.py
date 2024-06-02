from typing import List, NamedTuple, Dict
import numpy as np

# 定义数据结构
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.randn(hidden_size, input_size) * 0.01
        self.weights_hidden_hidden = np.random.randn(hidden_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(output_size, hidden_size) * 0.01

    def load_weights(self, filename):
        weights = np.loadtxt(filename)
        input_hidden_end = self.hidden_size * self.input_size
        hidden_hidden_end = input_hidden_end + self.hidden_size * self.hidden_size
        self.weights_input_hidden = weights[:input_hidden_end].reshape(self.hidden_size, self.input_size)
        self.weights_hidden_hidden = weights[input_hidden_end:hidden_hidden_end].reshape(self.hidden_size, self.hidden_size)
        self.weights_hidden_output = weights[hidden_hidden_end:].reshape(self.output_size, self.hidden_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, inputs):
        hidden_layer_input = np.dot(self.weights_input_hidden, inputs)
        hidden_layer_output = self.relu(hidden_layer_input)

        hidden_layer2_input = np.dot(self.weights_hidden_hidden, hidden_layer_output)
        hidden_layer2_output = self.relu(hidden_layer2_input)

        output_layer_input = np.dot(self.weights_hidden_output, hidden_layer2_output)
        output_layer_output = self.sigmoid(output_layer_input)
        
        return output_layer_output

class Vector(NamedTuple):
    x: int
    y: int

class CreatureDetail(NamedTuple):
    color: int
    type: int

class Creature(NamedTuple):
    creature_id: int
    pos: Vector
    speed: Vector
    detail: CreatureDetail

class RadarBlip(NamedTuple):
    creature_id: int
    dir: str

class Drone(NamedTuple):
    drone_id: int
    pos: Vector
    dead: bool
    battery: int
    scans: List[int]

# 初始化生物信息
creature_details: Dict[int, CreatureDetail] = {}

creature_count = int(input())
for _ in range(creature_count):
    creature_id, color, _type = map(int, input().split())
    creature_details[creature_id] = CreatureDetail(color, _type)

# 定义扫描顺序
solution = np.loadtxt("order.txt")
scan_order = solution[:12]  # 生物 ID 2 到 13
back_to_save = solution[12:]

# 准备状态记录
scan_save_status = {i: 0 for i in range(2, 14)}
radar_directions = {i: (0, 0) for i in range(2, 14)}

# 方向映射
direction_mapping = {
    'TL': (-1, -1),
    'TR': (1, -1),
    'BR': (1, 1),
    'BL': (-1, 1)
}

# 游戏循环
while True:
    my_scans: List[int] = []
    foe_scans: List[int] = []
    drone_by_id: Dict[int, Drone] = {}
    my_drones: List[Drone] = []
    foe_drones: List[Drone] = []
    visible_creatures: List[Creature] = []
    my_radar_blips: Dict[int, List[RadarBlip]] = {}

    my_score = int(input())
    foe_score = int(input())

    my_scan_count = int(input())
    for _ in range(my_scan_count):
        creature_id = int(input())
        my_scans.append(creature_id)

    foe_scan_count = int(input())
    for _ in range(foe_scan_count):
        creature_id = int(input())
        foe_scans.append(creature_id)

    my_drone_count = int(input())
    for _ in range(my_drone_count):
        drone_id, drone_x, drone_y, dead, battery = input().split()
        pos = Vector(int(drone_x), int(drone_y))
        drone = Drone(int(drone_id), pos, dead == '1', int(battery), [])
        drone_by_id[int(drone_id)] = drone
        my_drones.append(drone)
        my_radar_blips[int(drone_id)] = []

    foe_drone_count = int(input())
    for _ in range(foe_drone_count):
        drone_id, drone_x, drone_y, dead, battery = input().split()
        pos = Vector(int(drone_x), int(drone_y))
        drone = Drone(int(drone_id), pos, dead == '1', int(battery), [])
        drone_by_id[int(drone_id)] = drone
        foe_drones.append(drone)

    drone_scan_count = int(input())
    for _ in range(drone_scan_count):
        drone_id, creature_id = map(int, input().split())
        drone_by_id[drone_id].scans.append(creature_id)

    visible_creature_count = int(input())
    for _ in range(visible_creature_count):
        creature_id, creature_x, creature_y, creature_vx, creature_vy = map(int, input().split())
        pos = Vector(creature_x, creature_y)
        speed = Vector(creature_vx, creature_vy)
        visible_creatures.append(Creature(creature_id, pos, speed, creature_details[creature_id]))

    my_radar_blip_count = int(input())
    for _ in range(my_radar_blip_count):
        drone_id, creature_id, dir = input().split()
        drone_id = int(drone_id)
        creature_id = int(creature_id)
        my_radar_blips[drone_id].append(RadarBlip(creature_id, dir))
    
    # 更新每只生物的扫描和保存状态
    for creature_id in my_scans:
        if 2 <= creature_id <= 13:
            scan_save_status[creature_id] = 1  # 被扫描

    
    # 更新雷达方向
    for blips in my_radar_blips.values():
        for blip in blips:
            if 2 <= blip.creature_id <= 13:
                direction = blip.dir
                x_dir, y_dir = direction_mapping[direction]
                radar_directions[blip.creature_id] = (x_dir, y_dir)
    
    my_drone_position = Vector(0, 0)
    my_drone_battery = 0
    for drone in my_drones:
        if not drone.dead:
            my_drone_position = drone.pos
            my_drone_battery = drone.battery
            break

    # 生成最终的输出格式
    output_list = []
    for creature_id in range(2, 14):
        scan_status = scan_save_status[creature_id]
        radar_x, radar_y = radar_directions[creature_id]
        output_list.extend([scan_status, radar_x, radar_y])
    output_list.extend([(my_drone_position.x-5000)/5000, (my_drone_position.y-5000)/5000, my_drone_battery/30])

    input_size = 39
    hidden_size = 20
    output_size = 3
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.load_weights('matrix_weightNN.txt')
    result = nn.forward(output_list)
    target_x = int(result[0]*9999)
    target_y = int(result[1]*9999)
    light = 0
    if result[2] >= 0.5:
        light=1
    print(f"MOVE {target_x} {target_y} {light}")


