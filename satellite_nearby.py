#!/usr/bin/env python3
from glob import glob
import datetime

import matplotlib.pyplot as plt

from orbit_predictor.sources import EtcTLESource
from orbit_predictor.locations import Location
from orbit_predictor.predictors import Position

BNL_LOC = Location("BNL", latitude_deg=40.878180, longitude_deg=-72.856640, elevation_m=79)


def ll_distance(loc1, loc2):
    if isinstance(loc1, Position):
        loc1 = loc1.position_llh
        loc2 = loc2.position_llh
    return ((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)**0.5


def nearest_distance(predictor, loc, start_time=datetime.datetime.now(), delta=datetime.timedelta(seconds=300)):
    min_distance = float("inf")
    for i in range(86500 // delta.seconds):
        curr_time = start_time + delta * i
        position = predictor.get_position(curr_time)
        distance = ll_distance(position, loc)
        if min_distance > distance:
            min_distance = distance
            min_time = curr_time
    return min_distance, min_time


def plot_orbit(predictor, start_time, delta=datetime.timedelta(seconds=60), count=60, figure_num=0):
    x = []
    y = []
    for i in range(count):
        curr_time = start_time + delta * i
        position = predictor.get_position(curr_time)
        loc = position.position_llh[0:2]
        x.append(loc[0])
        y.append(loc[1])

    plt.figure(figure_num)
    plt.plot(x, y, label=predictor.sate_id)


sources = EtcTLESource()
for i in glob("gps/*.txt"):
    sources.load_file(i)



print("# Satellite ID, min distance, (satellite lon, lat, height)")
for sate_id in sources.satellite_list():
    predictor = sources.get_predictor(sate_id, True)
    min_distance, min_time = nearest_distance(predictor, BNL_LOC)
    if min_distance < 10:
        print(sate_id, min_distance, predictor.get_position(min_time).position_llh)
        plot_orbit(predictor, min_time - datetime.timedelta(seconds=1800))

plt.plot(BNL_LOC.position_llh[0], BNL_LOC.position_llh[1], marker='o', markersize=5, color='red', label="BMX")
plt.grid()
plt.title("GPS satellites passed over BMX")
plt.xlabel("Latitude, degree")
plt.ylabel("Longitude, degree")
lgd = plt.legend(bbox_to_anchor=(1, 1.2))
plt.savefig("satellite_plot.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
