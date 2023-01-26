import numpy as np

CLASS_NUM = 9
TERRAIN = {"Desert": 0.0, "Arctic": 1.0, "Normal": 2.0}


def load_meta():
    meta_info = {}
    with open("/home/sl29/FoundationSense/src/data_preprocess/ACIDS/ACIDS_meta.csv", "r") as file:
        lines = file.readlines()
        for line in lines:
            filename, vehicle_id, speed, terrain_id, distance = line.split(",")
            vehicle_id = int(vehicle_id.split(" ")[1])
            vehicle_type = np.zeros(CLASS_NUM)
            vehicle_type[vehicle_id - 1] = 1

            terrain_type = np.zeros(len(TERRAIN.keys()))
            terrain_type[int(TERRAIN[terrain_id])] = 1

            if speed == "?":
                continue
            elif speed == ">30":
                speed = 50

            meta_info[filename] = list(vehicle_type) + [float(speed)] + list(terrain_type) + [float(distance)]

    return meta_info


# load_meta()
