#!/usr/bin/env python3
from glob import glob
import datetime
import numpy as np

import matplotlib.pyplot as plt

from orbit_predictor.sources import EtcTLESource
from orbit_predictor.locations import Location
from orbit_predictor.predictors import Position

BNL_LOC = Location("BNL", latitude_deg=40.878180, longitude_deg=-72.856640, elevation_m=79)
now_time = datetime.datetime(2018, 4, 2, 17, 58, 46, 22518)

sources = EtcTLESource("gps/3le.txt")
sources2 = EtcTLESource("gps/gps-ops.txt")

for sate_id in sources.satellite_list():
    if "USA" in sate_id:
        predictor = sources.get_predictor(sate_id, True)
        print(sate_id, predictor.get_position(now_time).position_llh)


for sate_id in sources2.satellite_list():
    predictor = sources2.get_predictor(sate_id, True)
    print(sate_id, predictor.get_position(now_time).position_ecef)
