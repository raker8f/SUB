from typing import List, NamedTuple, Dict
import numpy as np

# 定義資料結構
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

# 初始化 creature 的資訊
creature_details: Dict[int, CreatureDetail] = {}

creature_count = int(input())
for _ in range(creature_count):
    creature_id, color, _type = map(int, input().split())
    creature_details[creature_id] = CreatureDetail(color, _type)

# 定義掃描順序
#scan_order = list(range(2, 14))  # 生物 ID 2 到 13
scan_order = [2, 3,  4, 5, 6 ,7  ,8  ,9 , 10 ,11, 12, 13]
back_to_save = [0 , 0,  0, 0,  0,  0 , 0,  0 , 0,  0 , 0 , 0]
k = 0
back = 0
number_of_scan = 0
# 遊戲迴圈
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
        drone_id, drone_x, drone_y, dead, battery = map(int, input().split())
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

    light = 0
    if (number_of_scan < len(drone_by_id[my_drones[0].drone_id].scans)):
        if (back_to_save[k]==1):
            back =1
        k= k+1
    number_of_scan = len(drone_by_id[my_drones[0].drone_id].scans)
    for drone in my_drones:
        if drone.pos.y <= 500:
            back = 0
        if back == 1:
            print(f"MOVE {drone.pos.x} {0} {0}")
    remaining_targets = [cid for cid in scan_order if cid not in drone_by_id[my_drones[0].drone_id].scans+my_scans]
    unscanned_visible_creatures = [c for c in visible_creatures if c.creature_id in remaining_targets]
    if unscanned_visible_creatures:
        light =1
    if back == 0:
        if remaining_targets:
            # 分配潛水艇去掃描下一個目標
            next_target_id = remaining_targets[0]
            next_target_creature = next((c for c in visible_creatures if c.creature_id == next_target_id), None)

            if next_target_creature:
                target_position = next_target_creature.pos
                target_speed    = next_target_creature.speed
                # 目標可見，指示潛水艇移動到目標位置
                for drone in my_drones:
                    print(f"MOVE {target_position.x+target_speed.x} {target_position.y+ target_speed.y} {light}")
            else:
                # 目標不可見，使用雷達判斷位置
                for drone in my_drones:
                    radar_blips = my_radar_blips[drone.drone_id]
                    target_blip = next((blip for blip in radar_blips if blip.creature_id == next_target_id), None)
                    if target_blip:
                        # 根據雷達方向計算預計位置
                        move_distance = 600
                        if target_blip.dir == 'TL':
                            target_position = Vector(drone.pos.x - move_distance, drone.pos.y - move_distance)
                        elif target_blip.dir == 'TR':
                            target_position = Vector(drone.pos.x + move_distance, drone.pos.y - move_distance)
                        elif target_blip.dir == 'BR':
                            target_position = Vector(drone.pos.x + move_distance, drone.pos.y + move_distance)
                        elif target_blip.dir == 'BL':
                            target_position = Vector(drone.pos.x - move_distance, drone.pos.y + move_distance)
                        target_position = Vector(max(min(target_position.x,9999),0),max(min(target_position.y,9999),0))
                        print(f"MOVE {target_position.x} {target_position.y} {light}")
                    else:
                        # 沒有雷達訊號，等待或進行其他操作
                        print(f"WAIT {0}")
        else:
            # 所有目標都已掃描完成
            for drone in my_drones:
                print(f"MOVE {drone.pos.x} {0} {0}")
