from sgp4.api import Satrec
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.positionlib import Barycentric
import numpy as np
import math
import pandas as pd


# Load timescale
ts = load.timescale()
eph = load('de421.bsp')
sun, earth = eph['sun'], eph['earth']

tlis = ts.utc(2024, 9, 24, 0, np.arange(0, 97900, 1), 0)

sunpos = np.transpose(sun.at(tlis).position.km)
earthpos = np.transpose(earth.at(tlis).position.km)
hubcoord = pd.read_csv('hubblecoordxyz.csv').to_numpy()
hubcoord = Barycentric(hubcoord + earthpos).position.au
intelcoord = pd.read_csv('intelsatcoordxyz.csv').to_numpy()
intelcoord = Barycentric(intelcoord + earthpos).position.au



for i in range(len(tlis)):
    zenith = hubcoord[i] - sunpos[i]
    intelsat = intelcoord[i] - sunpos[i]
    hubintel = intelsat - zenith
    theta = -1 * (math.acos(np.vdot(hubintel, zenith)/(np.linalg.norm(hubintel) * np.linalg.norm(zenith))) - 90)
    if theta < 0:
        hubintelproj = hubintel + zenith/np.linalg.norm(zenith)
    if theta > 0:
        hubintelproj = hubintel - zenith/np.linalg.norm(zenith)
    hubupproj = np.array([[1, 0, 0]])
    thetup = -1 * (math.acos(np.vdot(hubupproj, zenith)/(np.linalg.norm(hubupproj) * np.linalg.norm(zenith))) - 90)
    if theta < 0:
        hubupproj = hubupproj + zenith/np.linalg.norm(zenith)
    if theta > 0:
        hubupproj = hubupproj - zenith/np.linalg.norm(zenith)
    print(hubupproj[0], hubintelproj[1]], [hubupproj[0], hubintelproj[1])
    n_array = np.array([[hubupproj[0], hubintelproj[1]], [hubupproj[0], hubintelproj[1]]]) 
    print(np.arctan2(np.vdot(hubintelproj, hubupproj), np.linalg.det(n_array)))



# =============================================================================
# satpos = sat.at(t).position.au
# sunpos = sun.at(t).position.au
# earthpos = earth.at(t).position.au
# 
# satpos = Barycentric(satpos + earthpos).position.au
# satsun = sunpos - satpos
# satearth = earthpos - satpos
# =============================================================================
