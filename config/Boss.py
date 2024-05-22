from typing import List, NamedTuple, Dict
import numpy as np

# Define the data structures as namedtuples
class Vector(NamedTuple):
    x: int
    y: int

class FishDetail(NamedTuple):
    color: int
    type: int

class Fish(NamedTuple):
    fish_id: int
    pos: Vector
    speed: Vector
    detail: FishDetail

class RadarBlip(NamedTuple):
    fish_id: int
    dir: str

class Drone(NamedTuple):
    drone_id: int
    pos: Vector
    dead: bool
    battery: int
    scans: List[int]
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.zeros((hidden_size, input_size))
        self.weights_hidden_output = np.zeros((output_size, hidden_size))

    def load_weights(self, filename):
        weights = np.loadtxt(filename)
        self.weights_input_hidden = weights[:self.hidden_size * self.input_size].reshape(self.hidden_size, self.input_size)
        self.weights_hidden_output = weights[self.hidden_size * self.input_size:].reshape(self.output_size, self.hidden_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        hidden_layer_input = np.dot(self.weights_input_hidden, inputs)
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(self.weights_hidden_output, hidden_layer_output)
        output_layer_output = self.sigmoid(output_layer_input)
        return output_layer_output

fish_details: Dict[int, FishDetail] = {}

fish_count = int(input())
for _ in range(fish_count):
    fish_id, color, _type = map(int, input().split())
    fish_details[fish_id] = FishDetail(color, _type)

# game loop
while True:
    my_scans: List[int] = []
    foe_scans: List[int] = []
    drone_by_id: Dict[int, Drone] = {}
    my_drones: List[Drone] = []
    foe_drones: List[Drone] = []
    visible_fish: List[Fish] = []
    my_radar_blips: Dict[int, List[RadarBlip]] = {}

    my_score = int(input())
    foe_score = int(input())

    my_scan_count = int(input())
    for _ in range(my_scan_count):
        fish_id = int(input())
        my_scans.append(fish_id)

    foe_scan_count = int(input())
    for _ in range(foe_scan_count):
        fish_id = int(input())
        foe_scans.append(fish_id)

    my_drone_count = int(input())
    droneinfo = []
    for _ in range(my_drone_count):
        drone_id, drone_x, drone_y, dead, battery = map(int, input().split())
        droneinfo = [drone_x/10000, drone_y/10000, battery/5]
        pos = Vector(drone_x, drone_y)
        drone = Drone(drone_id, pos, dead == '1', battery, [])
        drone_by_id[drone_id] = drone
        my_drones.append(drone)
        my_radar_blips[drone_id] = []

    foe_drone_count = int(input())
    for _ in range(foe_drone_count):
        drone_id, drone_x, drone_y, dead, battery = map(int, input().split())
        pos = Vector(drone_x, drone_y)
        drone = Drone(drone_id, pos, dead == '1', battery, [])
        drone_by_id[drone_id] = drone
        foe_drones.append(drone)
    
    drone_scan_count = int(input())
    for _ in range(drone_scan_count):
        drone_id, fish_id = map(int, input().split())
        drone_by_id[drone_id].scans.append(fish_id)

    visible_fish_count = int(input())
    for _ in range(visible_fish_count):
        fish_id, fish_x, fish_y, fish_vx, fish_vy = map(int, input().split())
        pos = Vector(fish_x, fish_y)
        speed = Vector(fish_vx, fish_vy)
        visible_fish.append(Fish(fish_id, pos, speed, fish_details[fish_id]))
    direction_array = []
    my_radar_blip_count = int(input())
    for _ in range(my_radar_blip_count):
        drone_id, fish_id, dir = input().split()
        drone_id = int(drone_id)
        fish_id = int(fish_id)
        my_radar_blips[drone_id].append(RadarBlip(fish_id, dir))
        direction_array.append(dir)
    
    
    scan_presence = [1] * 12

    # 更新列表，將出現的數字對應的位置設置為 1
    for scan in my_scans:
        if 2 <= scan <= 13:
            scan_presence[scan - 2] = 0
    
    direction_map = {
    'TL': 0,
    'TR': 0.33,
    'BL': 0.67,
    'BR': 1
    }
    direction_array = [direction_map[dir] for dir in direction_array]

#   with open('output.txt', 'w') as file:
#      file.write(f'my_radar_blip_count={direction_array}\n')
#      file.write(f'my_scans={scan_presence}\n')
#       file.write(f'droneinfo={droneinfo}\n')
    allinfo = direction_array+scan_presence+droneinfo
    input_size = 27
    hidden_size = 10
    output_size = 3
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    nn.load_weights('matrix_weight.txt')
    result = nn.forward(allinfo)
    A = int(result[0]*10000)
    target_y = int(result[1]*10000)
    light = 0
    if result[2] >= 0.5:
        light=1
    print(f"MOVE {A} {target_y} {light}")
    with open('output.txt', 'w') as file:
        file.write(f'my_radar_blip_count={result[0]}\n')

