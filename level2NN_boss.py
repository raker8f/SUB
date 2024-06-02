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

    def load_weights(self, matrix):
        weights = matrix
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
    matrix=np.array[(-0.063911,0.602019,-0.738977,0.761085,-0.141778,-0.815786,0.178622,0.572399,1.118194,-0.399323,0.946118,-0.066981,-0.087965,-0.451591,-0.646516,-1.820625,0.671662,-1.038176,-0.993383,0.621928,-0.206164,-1.061742,0.443996,-0.316960,-1.211949,0.341010,0.189912,-0.246696,0.958977,0.406102,-0.086329,-0.128881,-0.125983,0.276414,-0.242948,-0.520641,-0.143908,1.159528,-0.341470,-0.044974,-0.946020,0.865080,0.466842,-0.444212,-0.961515,1.706006,-0.501406,1.380977,-0.864187,0.021918,-0.486960,0.251461,-1.955986,0.479824,-0.497616,0.855729,0.683057,0.088171,-0.737433,0.661595,0.165680,-1.844641,-0.217831,-0.419082,0.268829,0.813163,0.906978,-0.216700,0.640495,0.102948,0.560333,-0.183181,0.376645,1.660069,-0.387853,0.541111,1.050376,-0.026843,0.036187,0.541597,0.141097,0.986707,0.034314,-0.261527,-0.289573,0.479439,0.193821,0.331596,0.507410,1.310075,0.375744,-0.769088,0.883075,0.271481,-0.351158,-0.486182,0.055033,-1.206211,-0.261787,-0.360481,-0.836118,-0.973076,-0.588111,-1.008614,-0.246789,-0.858833,-0.022209,0.421764,0.677690,-0.048335,-2.078028,-0.554222,0.289907,1.311725,-0.146856,2.358266,0.031813,0.646266,1.102012,0.451800,0.128834,1.088397,0.414565,1.510553,-0.572308,-0.160100,1.073785,0.456177,0.175406,0.660517,-1.073857,1.563406,-0.244734,1.126903,-2.335919,1.205012,-0.290276,-0.981219,0.529080,0.410757,-0.299076,0.546339,-0.597088,-0.367094,0.155643,-0.666155,-1.423860,-0.268024,-0.199130,1.460016,0.229321,0.353344,-0.047733,-1.099028,1.063332,0.828412,0.262848,1.659935,0.040984,0.576651,0.407820,-0.984712,0.382059,0.358442,-0.713747,-0.618383,-0.648202,0.966611,0.345187,-0.168145,-0.210681,1.339748,-0.033000,-0.080357,0.214776,-0.900549,0.730882,0.078775,-0.712178,0.570529,-0.313877,-0.099205,0.717028,0.912774,-1.001722,-0.296844,-0.417381,-0.236913,0.102007,0.559769,0.324935,0.459739,0.336165,0.617086,-1.228972,0.377175,-0.636560,-0.726717,0.478036,-1.211677,0.542363,1.057252,-0.196091,-1.003120,-0.857946,0.508427,0.457119,1.675646,-0.191821,-0.051370,0.107487,-0.456282,-0.330030,-0.656266,0.803077,0.237651,-1.032522,-0.318655,0.467504,-0.174156,-0.386818,-0.256714,-0.696539,-0.536844,1.668545,0.322274,1.095280,0.129731,-1.245725,1.166295,1.556178,-0.929771,0.518432,0.119273,-1.739230,-0.455191,-0.415168,-1.091377,-1.115971,1.044074,0.157360,-0.575815,-0.944838,-1.091334,0.100760,-0.148103,-0.249305,-0.366004,-0.435583,0.723129,0.868697,-0.221735,-0.004765,-0.107357,1.084134,-0.343316,0.254919,0.620133,0.578970,0.724437,0.521118,-0.167616,0.054328,0.398261,-0.647622,0.265477,-0.066703,0.193620,-0.200003,0.761529,0.025450,-0.466393,0.178241,-0.895297,1.092492,0.049157,-0.003790,-0.153864,0.766838,0.256038,-0.178269,0.554605,-0.231621,0.238040,0.114605,-0.691187,0.931276,0.385305,0.297314,-0.768664,0.102698,-0.556189,-0.346178,-0.062689,1.005907,-0.050397,0.515652,-0.244700,1.068650,-0.258520,-1.245603,0.011949,-0.648008,1.186791,1.320321,0.573873,0.086340,0.529715,-0.884599,-0.171738,0.363746,0.062728,-0.205143,-0.037351,-0.127706,-0.401826,0.568391,0.281902,-0.336907,2.019490,0.382506,0.329112,0.655501,0.402219,1.769789,-0.208177,0.013516,1.543768,1.113992,-0.687829,-0.076409,1.074518,-0.967274,-1.598657,-0.450283,1.043161,0.030064,-0.633931,-0.231757,0.406691,-0.417310,0.554790,0.437191,1.850038,-0.138018,0.451257,-0.721816,-0.471070,0.338948,-0.408959,-0.321749,0.982396,1.070237,0.224581,-0.513979,0.466572,-0.328954,-0.489024,-0.574487,-0.860906,0.350814,-0.175832,-0.694144,-0.062903,-1.293423,0.218910,-0.808419,1.331114,0.439297,1.226470,0.604663,0.403813,-0.555620,-0.514394,0.631612,-0.577970,0.562726,-0.146949,-1.161490,-0.820168,0.062479,0.030889,-0.821221,-1.102821,0.795633,-1.055068,-0.260108,1.525411,0.198134,-0.724088,1.191022,-0.282107,0.362650,-0.061440,0.153423,-0.487202,0.364319,-0.385007,0.098290,0.780770,-0.019028,-0.874817,-0.461324,-1.416469,-0.937466,0.980830,-1.038886,0.055328,-0.319252,0.248238,1.408820,-0.422239,-0.183465,0.860354,0.759169,0.228259,0.457891,0.605405,-1.526246,-0.115123,-0.924882,0.569479,-0.517839,0.729250,0.127439,-0.347618,-0.248193,-0.604422,0.276721,0.182785,-0.208026,0.013655,-0.196276,-1.006745,-0.762968,0.143181,0.275812,0.497872,0.950229,0.048758,0.971463,1.231035,2.108938,0.530143,1.815371,0.167719,0.095352,-0.667165,-0.941432,0.691329,-0.724604,-1.067237,0.411374,-0.455058,-2.021649,0.038727,0.028046,0.488314,0.144054,0.157364,0.463166,-0.003073,0.074562,0.920889,0.092518,-1.801996,0.910729,0.208401,1.584797,-0.219169,0.728383,-0.377990,0.314550,1.921724,0.178364,0.698183,0.227487,0.036958,-0.055878,-0.632749,-0.445041,-0.355730,0.252861,0.237592,-1.560940,0.697990,-1.511062,0.252470,0.072088,-1.273607,-0.514299,-0.207048,-0.145882,-1.277706,-1.506564,-0.408033,-0.329168,-0.886484,0.932140,-0.636860,-0.055267,-0.334257,1.315664,1.112881,-0.540959,0.739907,-0.423554,-0.946539,0.462374,0.470751,1.695384,-1.066900,0.529407,0.492575,-0.201553,-0.896258,-0.737287,0.181677,-0.028294,-2.269179,0.609842,0.159393,-0.668368,0.094992,-1.348291,0.700358,1.321536,-1.879068,1.019924,1.257900,0.697670,0.786525,0.037333,-0.121563,-0.303072,-1.308219,-1.160049,0.964950,-0.033532,-0.326214,0.497683,-1.103992,-0.791264,0.966925,0.226390,-0.659856,-0.104468,-0.319621,-0.847310,0.281909,0.066842,-0.367597,0.195147,0.214260,-0.502051,-0.031726,-0.008746,-0.411347,-0.711504,-0.375357,-0.258731,0.086585,-1.696433,-0.370391,-0.558674,-0.717338,0.179778,-0.623533,-0.006360,-1.027077,-0.440184,0.792697,0.708977,-1.228722,0.980549,1.302285,-0.220966,0.815874,0.738425,-1.107307,-1.303725,0.267209,0.114190,1.488795,-0.220022,-1.079164,0.801551,1.796986,-0.702962,0.116557,0.024408,1.666199,-0.874174,-0.569735,0.979794,0.505029,-0.236809,0.145185,0.938491,1.526870,-0.939095,-0.314226,-0.350891,0.152445,1.782813,0.242813,-0.223530,-0.374030,-0.851376,0.594656,-0.142959,0.700250,-0.099619,-1.256583,-0.607694,-0.795350,0.181026,-0.373320,-0.566830,0.358249,-1.549283,-0.035917,1.067993,-0.446524,0.820573,0.167910,0.497660,-0.171400,-0.659328,-0.250064,-0.304927,0.257035,1.059161,0.663164,-0.194965,0.181092,-0.454732,0.829399,-0.306912,-0.717240,0.032774,-0.404301,-0.584258,0.028314,0.369953,-0.111309,0.167993,-0.100328,1.332919,1.062819,0.302070,-0.378784,-0.334554,0.181486,0.267363,0.021972,-0.873225,0.292042,-1.298503,-0.198627,-0.530841,0.041662,-0.647563,-0.459524,0.820989,0.526260,-0.143407,0.244093,2.042171,-0.581304,-0.422601,-0.329571,0.251634,-0.444152,1.109388,0.323569,0.211538,0.503339,0.577854,-0.627201,0.031372,-1.237657,0.079921,-0.587456,-0.650016,-0.245738,0.232758,0.464497,-0.574116,0.486084,0.332614,-1.329541,0.265592,1.139807,0.336425,-0.374280,-0.280956,-0.517437,-0.053573,1.287899,-0.503428,0.580782,-0.539916,-0.641552,-0.250526,-0.088265,1.536398,-0.300107,0.185155,1.050053,1.012418,-1.332989,-0.181421,-0.432404,2.362927,-0.633336,-1.205891,-0.625221,0.146362,-0.845365,-0.390137,0.130207,0.329388,-0.105413,-0.252231,-1.323391,0.055391,-0.015469,-0.696982,-1.054130,-0.520444,0.930714,-0.078837,-0.416319,-0.308283,-0.178300,-1.045779,0.697823,0.429500,1.185997,-0.026340,0.390468,0.079646,0.044431,0.029138,-0.611183,-0.638842,-1.005925,0.135116,0.191172,-1.379567,-0.034137,0.128767,-1.072692,-0.465345,-0.289521,0.689143,1.246353,-0.022297,0.842960,0.265197,0.761633,-0.038763,-0.350816,0.218618,0.640435,-0.459564,-0.599315,-0.539565,-1.057893,-0.096767,0.808694,0.180770,1.048371,-0.419232,-0.352374,-0.307159,-0.869119,0.371634,0.204010,-2.017540,1.496732,1.400930,-0.071548,0.531532,-0.805691,-0.029557,0.084128,-0.977912,-1.166956,0.327955,0.349860,-0.090521,-0.113109,-0.078058,1.456564,-0.196997,0.848073,-0.108708,0.867436,-0.460574,-0.021464,0.431350,-0.286371,0.650837,-1.466623,-0.091879,-0.473126,0.304334,0.010070,-0.322957,-0.160430,0.914651,-0.619872,-0.246980,-0.473546,-0.538215,-0.363885,1.102930,0.313898,-0.101702,0.376782,0.693274,-0.316236,0.404268,-0.304117,0.711741,0.590568,0.626236,-1.836809,0.297768,-1.227988,0.015460,1.584294,-0.964319,0.516777,0.092483,-0.336450,0.956799,-0.071777,-1.304458,-0.860191,0.153112,0.037152,0.505429,-0.330097,0.894917,0.297528,-1.747451,1.136212,0.199784,-0.075978,1.293791,0.067402,0.220534,0.168576,-0.660641,-0.620179,0.988037,-0.584704,-0.042025,-0.217191,0.940557,0.053739,-0.859043,-0.304973,0.799131,-0.999723,0.813475,1.847847,-0.287847,1.464012,0.189385,-0.478796,-0.557595,0.416092,-0.770666,0.574920,1.283516,0.234498,0.955155,-0.133361,-0.633826,-0.355941,-0.084231,-0.227785,-0.029409,-0.336717,-0.448265,-0.726111,0.372814,-0.954717,1.387169,0.392933,0.024979,1.338275,0.883140,0.182232,0.673731,0.029291,0.512208,0.149752,-0.944008,-0.145843,0.425813,0.241836,0.856009,-0.584561,-0.217544,0.147770,-0.830856,0.010765,-0.033245,-0.621359,-1.593231,-0.631543,0.349911,-1.007011,1.665885,-0.992874,1.807662,0.723997,0.488967,0.710840,1.932655,0.205316,0.050210,0.229527,-0.450410,1.539381,-0.652296,-0.646997,-0.380103,-0.412147,-0.574870,-0.600767,-0.881558,-1.071803,0.184942,0.003043,0.143992,1.392456,-0.042830,0.187059,0.056827,0.911764,0.222405,-1.552711,0.023272,-0.900290,0.343314,-0.192569,-1.352334,0.387517,-0.568630,0.355699,0.881597,0.375209,0.215378,0.959451,1.400465,-0.226395,-1.186871,0.139667,0.614224,-1.328096,0.022678,-0.328012,-2.264523,0.900702,-0.471314,0.440047,0.375954,0.583232,-0.784729,-0.498081,-0.510908,0.772453,0.118263,0.343352,0.031144,0.179965,-0.288506,-0.074741,0.720995,-0.353728,0.641435,0.754448,-0.591252,-0.410738,-0.072098,-1.317297,0.118389,0.619616,-0.243058,0.730751,-0.074989,1.607733,1.280308,0.412918,-0.519298,0.491793,0.453386,0.179227,-0.223659,-0.458938,0.619772,0.371605,0.486755,-1.797114,0.617055,0.617485,1.661525,0.579734,0.080428,0.430131,-0.531547,0.420341,0.236817,-0.024144,-0.772008,1.346357,-0.148073,-0.103990,-0.366031,-0.113516,-2.826774,0.181559,-0.529417,-1.016694,0.282817,-0.069988,0.095751,0.431205,0.861582,-0.473783,-0.606609,0.487645,-0.802777,-1.049306,0.055469,0.439896,0.115717,-0.104472,0.378389,1.818234,0.955099,1.538932,-0.319848,0.288383,1.221511,-0.387328,-0.575956,-0.317808,0.870680,-0.606411,0.359390,-1.054190,-0.095135,1.729535,-0.777232,1.322645,0.107452,-0.202323,-0.358174,-0.231394,1.134359,0.013380,0.280003,0.916737,-0.544156,-0.617161,-0.441076,0.015064,0.057659,-0.844587,1.275784,0.780208,-2.057238,0.467399,0.108348,0.816503,0.123170,-0.483537,0.174845,0.724927,-1.174585,-0.389605,-0.925601,-1.322396,-0.432183,-0.024300,-0.413472,-0.462874,0.894207,0.261568,0.066225,0.005857,-0.863086,-0.312319,0.636646,1.379930,1.274990,-2.568926,1.292960,-0.175946,0.452039,-0.640881,-0.314512,-0.400144,1.500315,-1.281386,-0.253212,0.375407,0.297765,-1.723175,-0.996148,-0.776800,-0.810435,0.648090,-0.065163,-0.501666,-1.144909,-0.505039,-0.034534,0.418225,-1.148451,0.630000,-0.543109,0.855907,-0.485078,-0.307897,0.588900,-1.281402,0.123063,1.129821,0.392951,0.500726,-1.394771,0.288073,-0.080394,0.584722,0.776605,0.334971,-0.812254,0.448813,-0.540838,0.904303,-0.139448,-1.186381,-0.272114,-0.247997,0.412342,0.106094,-0.807237,1.411526,-0.669131,0.376784,0.380328,0.120490,-0.924438,-1.050357,0.410156,-0.015549,-0.403355,0.648477,-0.632506,-0.202278,-0.865585,-0.904743,0.360538,1.360158,-1.088018,0.522060,-1.580505,-0.675163,-0.733627,-0.647738,-0.161841,-0.805035,0.537230,0.440505,-0.842736,-0.297871,-0.373592,-0.039720,0.352985,1.245722,0.231771,1.037461,-0.700143,1.526762,-0.220831,0.158221,-0.765564,0.177633,-1.195301,-0.577563,-0.142005,0.489740,-0.772692,-0.449127,-0.830899,0.352843,-0.015350,1.139276,2.560039,-0.107469,0.652271,0.015733,-0.904399,0.742557,-0.912169,0.573913,0.514010,0.367731,-0.952471,-0.602547,-0.600982,-0.639394,-0.983555,0.673015,0.168828,-0.409520,1.851374,0.498670,0.990856,-0.400242,1.105422,-1.141701,-0.468791,1.487961,1.438681,0.767120,-0.219702,-0.126977,-0.671915,0.145418,0.394961,-0.301154)]
    nn.load_weights(matrix)
    result = nn.forward(output_list)
    target_x = int(result[0]*9999)
    target_y = int(result[1]*9999)
    light = 0
    if result[2] >= 0.5:
        light=1
    print(f"MOVE {target_x} {target_y} {light}")


