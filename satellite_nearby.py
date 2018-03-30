#!/usr/bin/env python3
from glob import glob
import datetime
import numpy as np

import matplotlib.pyplot as plt

from orbit_predictor.sources import EtcTLESource
from orbit_predictor.locations import Location
from orbit_predictor.predictors import Position

BNL_LOC = Location("BNL", latitude_deg=40.878180, longitude_deg=-72.856640, elevation_m=79)


def ll_distance(loc1, loc2):
    if hasattr(loc1, "position_llh"):
        loc1 = loc1.position_llh
    if hasattr(loc2, "position_llh"):
        loc2 = loc2.position_llh
    return ((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)**0.5


def llh_to_altaz(loc1, loc2, radian=False):
    '''Return the altaz looking from loc2'''
    loc1_llh = loc1.position_llh
    loc2_llh = loc2.position_llh
    loc1_xyz = loc1.position_ecef
    loc2_xyz = loc2.position_ecef

    if radian:
        coeff = 1
    else:
        coeff = 180 / np.pi

    dx = loc1_llh[1] - loc2_llh[1]
    dy = loc1_llh[0] - loc2_llh[0]
    az = np.arctan(np.float64(dx) / np.float64(dy)) * coeff
    if dy < 0:
        az += np.pi * coeff
    if az > np.pi * coeff:
        az -= 2 * np.pi * coeff

    # Earth ellipsoid parameters
    a = 6378.1370
    b = 6356.752314
    # earth radius
    loc1_lat = loc1_llh[0] * np.pi / 180
    loc2_lat = loc2_llh[0] * np.pi / 180
    n1 = a * a / np.sqrt(a * a * np.cos(loc1_lat) ** 2 + b ** 2 * np.sin(loc1_lat) ** 2)
    n2 = a * a / np.sqrt(a * a * np.cos(loc2_lat) ** 2 + b ** 2 * np.sin(loc2_lat) ** 2)
    dist = np.sqrt((loc1_xyz[0] - loc2_xyz[0])**2 + (loc1_xyz[1] - loc2_xyz[1])**2 + (loc1_xyz[2] - loc2_xyz[2])**2)\
    # cosA = (b^2 + c^2 - a^2) / 2bc
    cosalt = ((n2 + loc2_llh[2])**2 + dist**2 - (n1 + loc1_llh[2])**2) / (2 * (n2 + loc2_llh[2]) * dist)
    #print(cosalt, n1+loc1_llh[2], n2+loc2_llh[2], dist)
    alt = (np.arccos(cosalt) - 0.5 * np.pi) * coeff

    return alt, az


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


def plot_orbit(ax, predictor, start_time, delta=datetime.timedelta(seconds=60), count=60, polar=False):
    x = []
    y = []
    for i in range(count):
        curr_time = start_time + delta * i
        position = predictor.get_position(curr_time)
        loc = position.position_llh[0:2]
        if polar:
            altaz = llh_to_altaz(position, BNL_LOC, radian=True)
            x.append(altaz[1])
            y.append((0.5*np.pi - altaz[0]) * 180 / np.pi)
        else:
            x.append(loc[0])
            y.append(loc[1])
        print(position.position_llh[0] - BNL_LOC.position_llh[0])

    ax.plot(x, y, label=predictor.sate_id)



sources = EtcTLESource()
for i in glob("gps/*.txt"):
    sources.load_file(i)


ax = plt.subplot(111)
print("# Satellite ID, min distance, (satellite lon, lat, height)")
for sate_id in sources.satellite_list():
    predictor = sources.get_predictor(sate_id, True)
    min_distance, min_time = nearest_distance(predictor, BNL_LOC)
    if min_distance < 10:
        print(sate_id, min_distance, predictor.get_position(min_time).position_llh)
        plot_orbit(ax, predictor, min_time - datetime.timedelta(seconds=1800))

plt.plot(BNL_LOC.position_llh[0], BNL_LOC.position_llh[1], marker='o', markersize=5, color='red', label="BMX")
plt.grid()
plt.title("GPS satellites passed over BMX")
plt.xlabel("Latitude, degree")
plt.ylabel("Longitude, degree")
lgd = plt.legend(bbox_to_anchor=(1, 1.2))
plt.savefig("satellite_plot.png", bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.clf()
ax = plt.subplot(111, projection='polar')
for sate_id in sources.satellite_list():
    predictor = sources.get_predictor(sate_id)
    min_distance, min_time = nearest_distance(predictor, BNL_LOC)
    if min_distance < 10:
        print(sate_id, min_distance, predictor.get_position(min_time).position_llh)
        plot_orbit(ax, predictor, min_time - datetime.timedelta(seconds=1800), polar=True)


ax.plot(0, 0, marker='o', markersize=5, color='red', label="BMX")
plt.title("GPS satellites passed over BMX")
lgd = plt.legend(bbox_to_anchor=(1, 1.2))
plt.savefig("satellite_polar.png", bbox_extra_artists=(lgd,), bbox_inches='tight')