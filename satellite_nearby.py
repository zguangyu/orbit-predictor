#!/usr/bin/env python3
from glob import glob
import datetime

from orbit_predictor.sources import EtcTLESource
from orbit_predictor.locations import Location

BNL_LOC = Location("BNL", latitude_deg=40.878180, longitude_deg=-72.856640, elevation_m=79)


def ll_distance(loc1, loc2):
    return ((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)**0.5


def nearest_distance(predictor, loc, start_time=datetime.datetime.now(), delta=datetime.timedelta(seconds=60)):
    min_distance = float("inf")
    for i in range(1440):
        curr_time = start_time + delta * i
        position = predictor.get_position(curr_time)
        distance = ll_distance(position.position_llh, loc)
        if min_distance > distance:
            min_distance = distance
    return min_distance


sources = EtcTLESource()
for i in glob("gps/*.txt"):
    sources.load_file(i)

for sate_id in sources.satellite_list():
    predictor = sources.get_predictor(sate_id)
    min_distance = nearest_distance(predictor, BNL_LOC.position_llh)
    if min_distance < 10:
        print(sate_id, min_distance)
