import os
import sys
import time

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

"""
Side note:
    1) Sensors not using: gps, mic, lig
    2) Positions: all 7 positions
    3) Default frequency: 50 Hz
    4) Users not used: proband2 (missing forearm in climbing up), proband6 (missing thigh in jumping)
"""

SENSORS = ["acc", "gyr", "mag", "lig"]
SENSOR_NAME = {
    "acc": "acc",
    "gyr": "Gyroscope",
    "mag": "MagneticField",
    "lig": "Light",
}
ACTIVITIES = ["climbingdown", "climbingup", "jumping", "lying", "running", "sitting", "standing", "walking"]
ACTIVITIES_MAP = {
    "climbingdown": 0,
    "climbingup": 1,
    "jumping": 2,
    "lying": 3,
    "running": 4,
    "sitting": 5,
    "standing": 6,
    "walking": 7,
}
POSITIONS = ["head", "chest", "upperarm", "waist", "shin"]
USER_MISSING_ACTIVITY = {"proband2": "climbingup", "proband6": "jumping"}

# Index in raw data
TIME = 0
X = 1
Y = 2
Z = 3


def extract_user_list(file_path):
    user_list = []
    filename_list = os.listdir(file_path)

    for filename in filename_list:
        if "proband" in filename:
            user_list.append(filename)

    return user_list


def pair_user_data(user, input_path, output_path):
    """
    Pair user data into one file, including all sensor readings.
    :param user:
    :param input_path:
    :param output_path:
    :return:
    """
    user_input_path = os.path.join(input_path, user)

    for activity in ACTIVITIES:
        if (user in USER_MISSING_ACTIVITY) and (activity in USER_MISSING_ACTIVITY[user]):
            continue

        label = ACTIVITIES_MAP[activity]

        user_activity_out_file = os.path.join(output_path, user + "_" + activity + ".csv")
        if os.path.exists(user_activity_out_file):
            os.remove(user_activity_out_file)

        start_time = -np.inf
        end_time = np.inf
        df_list = []

        # Extract df for each (sensor, pos) and get start time and end time
        for sensor in SENSORS:
            sensor_name = SENSOR_NAME[sensor]
            user_activity_sensor_path = os.path.join(user_input_path, sensor + "_" + activity + "_csv")

            if not os.path.exists(user_activity_sensor_path):
                print("User-label missing: " + user + " " + activity + " " + sensor)

            for pos in POSITIONS:
                if user == "proband6" and activity in {"sitting"}:
                    user_acctivity_sensor_pos_file = os.path.join(
                        user_activity_sensor_path, sensor_name + "_" + activity + "_2_" + pos + ".csv"
                    )
                elif user == "proband4" and activity in {"climbingdown", "climbingup", "walking"}:
                    user_acctivity_sensor_pos_file = os.path.join(
                        user_activity_sensor_path, sensor_name + "_" + activity + "_2_" + pos + ".csv"
                    )
                elif user == "proband13" and activity in {"walking"}:
                    user_acctivity_sensor_pos_file = os.path.join(
                        user_activity_sensor_path, sensor_name + "_" + activity + "_2_" + pos + ".csv"
                    )
                elif user == "proband14" and activity in {"climbingdown", "climbingup"}:
                    user_acctivity_sensor_pos_file = os.path.join(
                        user_activity_sensor_path, sensor_name + "_" + activity + "_2_" + pos + ".csv"
                    )
                elif user == "proband7" and activity in {"climbingdown", "climbingup", "sitting"}:
                    user_acctivity_sensor_pos_file = os.path.join(
                        user_activity_sensor_path, sensor_name + "_" + activity + "_2_" + pos + ".csv"
                    )
                elif user == "proband8" and activity in {"standing"}:
                    user_acctivity_sensor_pos_file = os.path.join(
                        user_activity_sensor_path, sensor_name + "_" + activity + "_2_" + pos + ".csv"
                    )
                else:
                    user_acctivity_sensor_pos_file = os.path.join(
                        user_activity_sensor_path, sensor_name + "_" + activity + "_" + pos + ".csv"
                    )

                if not os.path.exists(user_acctivity_sensor_pos_file):
                    print("Missing data for: " + " ".join([user, activity, sensor, pos]))

                df = pd.read_csv(filepath_or_buffer=user_acctivity_sensor_pos_file)
                df = df.drop(labels=["id"], axis=1)
                df_list.append(df)

                df_min_time = df.attr_time.min()
                df_max_time = df.attr_time.max()

                if df_min_time > start_time:
                    start_time = df_min_time

                if df_max_time < end_time:
                    end_time = df_max_time

        if not start_time < end_time:
            raise Exception("Invalid data time range!")

        end_time = start_time + ((end_time - start_time) / 20) * 20
        interpolated_count = int((end_time - start_time) / 20 + 1)
        interpolated_times = np.linspace(start_time, end_time, interpolated_count)
        print(user + " " + activity + " start time: " + str(start_time) + ", end time: " + str(end_time))

        # Combine data from different sensors and positions into one file
        user_activity_data = np.array([interpolated_times, np.ones(len(interpolated_times)) * label]).T

        for df in df_list:
            df = df[(df.attr_time >= start_time) & (df.attr_time <= end_time)]
            sensor_pos_data = df.values

            # Make sure the start and end time of current (sensor, pos) data
            if sensor_pos_data[0, TIME] > start_time:
                sensor_pos_data[0, TIME] = start_time

            if sensor_pos_data[-1, TIME] < end_time:
                sensor_pos_data[-1, TIME] = end_time

            # Interpolate
            interpolator = interp1d(sensor_pos_data[:, 0], sensor_pos_data[:, 1:], axis=0)
            interpolated_sensor_pos_data = interpolator(interpolated_times)

            # Add into integrated data
            user_activity_data = np.concatenate((user_activity_data, interpolated_sensor_pos_data), axis=1)

        # Save to the file for per (user, activity)
        print("Data shape: " + str(np.shape(user_activity_data)))
        np.savetxt(user_activity_out_file, user_activity_data)


if __name__ == "__main__":
    input_path = "/home/sl29/data/RealWorld-HAR/extracted-data"
    output_path = "/home/sl29/data/RealWorld-HAR/processed-data/paired-data-5_pos-4_mod"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    start = time.time()

    user_list = extract_user_list(input_path)

    for user in user_list:
        pair_user_data(user, input_path, output_path)

    end = time.time()
    print("------------------------------------------------------------------------")
    print("Total execution time: %f s" % (end - start))
