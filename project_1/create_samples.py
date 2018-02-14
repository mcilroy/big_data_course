import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from geo import haversine_distance


def create_samples():
    """ create sub trajectory samples for use in my data mining models. The program loads the raw geolife data.
    Then I group the data by user and date. Then I ensure I properly group the data by transportation mode.
    Finally I calculate the 4 metrics (bearing, acceleration, speed and distance) and the 5 stats (min, max, median, mean and std. dec.)"""
    df = pd.read_csv("geolife_raw.csv")
    print("data size: " + str(df.size))
    df['collected_time'] = pd.to_datetime(df['collected_time'])
    grouped_by_id = df.groupby("t_user_id")
    new_data = []
    num_ids = len(grouped_by_id)
    print("Number of ids: " + str(num_ids))
    iteration = 0
    for id_name, id_group in grouped_by_id:
        print(id_name)
        print("Percent complete: " + str(iteration / float(num_ids)))
        grouped_by_id_and_date = id_group.groupby(pd.Grouper(key='collected_time', freq='D'))
        for id_date_name, id_date_group in grouped_by_id_and_date:
            create_sub_traj_by_transportation_mode(id_date_group, new_data)
        iteration += 1
    new_data = np.array(new_data)
    column_names = get_column_names()
    new_df = pd.DataFrame(data=new_data, columns=column_names)
    new_df.to_csv("traj_samples_v3.csv", sep=',', index=False)


def create_sub_traj_by_transportation_mode(id_date_group, new_data):
    """ I first iterate over all trajectories and group the trajectories by transportation mode to handle cases where
    a person goes back to the same mode. For example: walks -> takes train -> then walks again.
        Then for each trajectory I calculate the distances, speeds, accelerations and bearings between points to get
        a list of entries for each metric.
        I then calculate the mean, median, min, max and std. dev. for each metric"""
    if len(id_date_group) < 1:
        return
    grouped_by_mode = []
    mode = id_date_group.transportation_mode.values[0]
    sub_trajectory = []

    for i in range(len(id_date_group)):
        if mode != id_date_group.transportation_mode.values[i]:
            # new sub-trajectory
            grouped_by_mode.append(sub_trajectory)
            sub_trajectory = []
            mode = id_date_group.transportation_mode.values[i]
            entry = [id_date_group.collected_time.values[i], id_date_group.latitude.values[i],
                     id_date_group.longitude.values[i], id_date_group.transportation_mode.values[i],]
            sub_trajectory.append(entry)
        else:
            # same sub-trajectory
            entry = [id_date_group.collected_time.values[i], id_date_group.latitude.values[i],
                     id_date_group.longitude.values[i], id_date_group.transportation_mode.values[i],]
            sub_trajectory.append(entry)
    if len(sub_trajectory) > 0:
        grouped_by_mode.append(sub_trajectory)
        sub_trajectory = []
    for i in range(len(grouped_by_mode)):
        if len(grouped_by_mode[i]) >= 10:
            # get stats and add to sub trajectory
            sub_traj_times = calc_time_diff(grouped_by_mode[i])
            sub_traj_distances = calc_distance(grouped_by_mode[i])
            distance_stats = calc_stat_features(sub_traj_distances)
            sub_traj_speeds = calc_speed(sub_traj_times, sub_traj_distances)
            speed_stats = calc_stat_features(sub_traj_speeds)
            sub_traj_accelerations = calc_acceleration(sub_traj_times, sub_traj_speeds)
            acceleration_stats = calc_stat_features(sub_traj_accelerations)
            sub_traj_bearings = calc_bearing(grouped_by_mode[i])
            bearing_stats = calc_stat_features(sub_traj_bearings)

            data_row = np.concatenate((distance_stats, speed_stats, acceleration_stats,
                                       bearing_stats, [grouped_by_mode[i][0][3]]))
            new_data.append(data_row)


def calc_time_diff(sub_trajectory):
    """ calculate the difference in time between points."""
    rows = []
    for i in range(len(sub_trajectory)):
        if i+1 < len(sub_trajectory):
            time_diff = sub_trajectory[i+1][0] - sub_trajectory[i][0]
            rows.append(time_diff / np.timedelta64(1, 's'))
    return rows


def calc_distance(sub_trajectory):
    """ calculate the difference in distance using haversine distance between points."""
    rows = []
    for i in range(len(sub_trajectory)):
        if i+1 < len(sub_trajectory):
            new_row = haversine_distance(sub_trajectory[i][1], sub_trajectory[i][2],
                                         sub_trajectory[i+1][1], sub_trajectory[i+1][2])
            rows.append(new_row)
    return rows


def calc_speed(sub_traj_times, sub_traj_distances):
    """ calculate the speed using distance over time"""
    rows = []
    for i in range(len(sub_traj_times)):
        if sub_traj_times[i] == 0:
            speed = 0
        else:
            speed = sub_traj_distances[i]/sub_traj_times[i]
        rows.append(speed)
    return rows


def calc_acceleration(sub_traj_times, sub_traj_speeds):
    """ calculate the acceleration using the difference in speeds over time"""
    rows = []
    for i in range(len(sub_traj_times)):
        if i + 1 < len(sub_traj_times):
            if sub_traj_times[i] == 0:
                new_row = 0
            else:
                new_row = (sub_traj_speeds[i + 1] - sub_traj_speeds[i]) / sub_traj_times[i]
            rows.append(new_row)
    return rows


def calc_bearing(sub_trajectory):
    """ calculate the compass bearing"""
    rows = []
    for i in range(len(sub_trajectory)):
        if i + 1 < len(sub_trajectory):
            new_row = calculate_initial_compass_bearing((sub_trajectory[i][1], sub_trajectory[i][2]),
                                                        (sub_trajectory[i+1][1], sub_trajectory[i+1][2]))
            rows.append(new_row)
    return rows


def get_column_names():
    """ return the columns for use in saving the new csv file"""
    column_names = []
    features = ["distance", "speed", "acceleration", "bearing"]
    stat = ["min", "max", "mean", "median", "std"]
    for x in range(len(features)):
        for y in range(len(stat)):
            column_names.append(str(stat[y]+"_"+features[x]))
    column_names.append("transportation_mode")
    return column_names


def calc_stat_features(entries):
    """ calculate the statistics of a list of entries """
    min_time_diff = np.min(entries)
    max_time_diff = np.max(entries)
    mean_time_diff = np.mean(entries)
    median_time_diff = np.median(entries)
    std_dev_time_diff = np.std(entries)
    return [min_time_diff, max_time_diff, mean_time_diff, median_time_diff, std_dev_time_diff]


def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                           * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

create_samples()
