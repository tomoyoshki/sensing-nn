import numpy as np

CLASS_NUM = 9
TERRAIN = {"Desert": 0.0, "Arctic": 1.0, "Normal": 2.0}
SPEED = {"1": 0, "5": 1, "10": 2, "15": 3, "20": 4, "30": 5, ">30": 6, "40": 6}
DISTANCE = {"5": 0, "10": 1, "25": 2, "50": 3, "75": 4}


def load_meta():
    meta_info = {}
    with open("ACIDS_meta.csv", "r") as file:
        lines = file.readlines()
        for line in lines:
            filename, vehicle_id, speed, terrain_id, distance = line.split(",")
            distance = distance.strip("\n")
            vehicle_id = int(vehicle_id.split(" ")[1])
            distance = DISTANCE[distance]

            if speed == "?":
                continue
            else:
                speed = SPEED[speed]

            meta_info[filename] = {
                "vehicle": float(vehicle_id),
                "terrain": float(TERRAIN[terrain_id]),
                "speed": float(speed),
                "distance": float(distance),
            }
    return meta_info


# load_meta()
