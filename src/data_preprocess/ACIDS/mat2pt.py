import torch
from data_loader import *
import numpy as np

PT_FILE_PATH = "/home/tianshi/data/ACIDS/pt_files"
INDEX_FILE_PATH = "/home/tianshi/data/ACIDS/pt_index"

# SPEED: 1~10: 0, 11~30: 1, 31~50: 2
# SPEED_ENCODING = {"1": 0, "5": 1, "10": 2, "15": 3, "20": 4, "30": 5, "40": 6, "50": 7}

# DISTANCE: 5~25: 0, 25~75: 1
# DISTANCE_ENCODING = {"5": 0, "10": 1, "25": 2, "50": 3, "75": 4}

train_X, train_Y, test_X, test_Y, train_sample_count, test_sample_count, train_labels, test_labels = load_data(mode="stft")
print("train_X.shape", train_X.shape)

vehicle_type_set = set()
speed_set = set()
terrain_type_set = set()
distance_set = set()

train_indexes = []
for i, (x, y) in enumerate(zip(train_X, train_Y)):
    vehicle_type = y[:9]
    speed_v = int(y[9])
    speed = np.zeros(3)
    if speed_v <= 10:
        speed[0] = 1
    elif speed_v <= 30:
        speed[1] = 1
    else:
        speed[2] = 1

    # speed[SPEED_ENCODING[str(int(y[9]))]] = 1
    terrain_type = y[10:13]

    distance_v = int(y[-1])
    distance = np.zeros(2)
    if distance_v <= 25:
        distance[0] = 1
    else:
        distance[1] = 1

    if np.argmax(vehicle_type) == 2 or np.argmax(vehicle_type) == 6:
        continue

    vehicle_type_set.add(np.argmax(vehicle_type))
    speed_set.add(np.argmax(speed))
    terrain_type_set.add(np.argmax(terrain_type))
    distance_set.add(np.argmax(distance))

    sample = {"data": {"shake": {"audio": np.expand_dims(x[0], 0), "seismic": np.expand_dims(x[1], 0)}}, "label": {"vehicle_type": vehicle_type, "speed": speed, "terrain_type": terrain_type, "distance": distance}}
    output_path = os.path.join(PT_FILE_PATH, f"train_{i}.pt")
    train_indexes.append(output_path)
    torch.save(sample, output_path)

print("in train set:")
print(f"vehicle_type_set: {vehicle_type_set}")
print(f"speed_set: {speed_set}")
print(f"terrain_type_set: {terrain_type_set}")
print(f"distance_set: {distance_set}")
print("=====================================")

with open(os.path.join(INDEX_FILE_PATH, "train_index.txt"), "w") as f:
    for index in train_indexes:
        f.write(index + "\n")

vehicle_type_set = set()
speed_set = set()
terrain_type_set = set()
distance_set = set()

test_indexes = []
for i, (x, y) in enumerate(zip(test_X, test_Y)):
    vehicle_type = y[:9]

    speed_v = int(y[9])
    speed = np.zeros(3)
    if speed_v <= 10:
        speed[0] = 1
    elif speed_v <= 30:
        speed[1] = 1
    else:
        speed[2] = 1

    # speed = np.zeros(8)
    # speed[SPEED_ENCODING[str(int(y[9]))]] = 1
    terrain_type = y[10:13]

    distance_v = int(y[-1])
    distance = np.zeros(2)
    if distance_v <= 25:
        distance[0] = 1
    else:
        distance[1] = 1

    if np.argmax(vehicle_type) == 2 or np.argmax(vehicle_type) == 6:
        continue

    vehicle_type_set.add(np.argmax(vehicle_type))
    speed_set.add(np.argmax(speed))
    terrain_type_set.add(np.argmax(terrain_type))
    distance_set.add(np.argmax(distance))

    sample = {"data": {"shake": {"audio": np.expand_dims(x[0], 0), "seismic": np.expand_dims(x[1], 0)}}, "label": {"vehicle_type": vehicle_type, "speed": speed, "terrain_type": terrain_type, "distance": distance}}
    output_path = os.path.join(PT_FILE_PATH, f"test_{i}.pt")
    test_indexes.append(output_path)
    torch.save(sample, output_path)

with open(os.path.join(INDEX_FILE_PATH, "test_index.txt"), "w") as f:
    for index in test_indexes:
        f.write(index + "\n")

with open(os.path.join(INDEX_FILE_PATH, "val_index.txt"), "w") as f:
    for index in test_indexes:
        f.write(index + "\n")

print("in test set:")
print(f"vehicle_type_set: {vehicle_type_set}")
print(f"speed_set: {speed_set}")
print(f"terrain_type_set: {terrain_type_set}")
print(f"distance_set: {distance_set}")
print("=====================================")
