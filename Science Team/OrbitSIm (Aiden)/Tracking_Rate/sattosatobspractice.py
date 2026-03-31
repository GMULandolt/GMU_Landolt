from sgp4.api import Satrec
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.positionlib import Barycentric
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


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
hubintel = intelcoord - hubcoord
hubintelr = np.sqrt(hubintel[:,0]**2 + hubintel[:,1]**2 + hubintel[:,2]**2)
hubintelphi = np.arctan2(hubintel[:,1],hubintel[:,0])
hubintelthe = np.arccos(hubintel[:,2]/hubintelr)

day = 1
length = 1
angularsep = np.zeros(int(1440 * length))

for i in range(1440*day, int(1440*day + 1440*length)):
    j = i - 1440*day
    angularsep[j] = np.rad2deg(np.arccos(np.sin(hubintelthe[i-1])*np.sin(hubintelthe[i]) + \
                              np.cos(hubintelthe[i-1])*np.cos(hubintelthe[i]) * \
                                  np.cos(hubintelphi[i] - hubintelphi[i-1]))) * 60
    
        
plt.plot(np.arange(0, 1440 * length, 1), angularsep)
plt.xlabel('Minutes in Day')
plt.ylabel('Tracking Rate each Second (Arcseconds)')
plt.title("Tracking Rate for Hubble to Observe Intelsat 40E (9/25/2024)")


    
# =============================================================================
#     theta = -1 * (math.acos(np.vdot(hubintel, zenith)/(np.linalg.norm(hubintel) * np.linalg.norm(zenith))) - 90)
#     if theta < 0:
#         hubintelproj = hubintel + zenith/np.linalg.norm(zenith)
#     if theta > 0:
#         hubintelproj = hubintel - zenith/np.linalg.norm(zenith)
#     hubupproj = np.array([[1, 0, 0]])
#     thetup = -1 * (math.acos(np.vdot(hubupproj, zenith)/(np.linalg.norm(hubupproj) * np.linalg.norm(zenith))) - 90)
#     if theta < 0:
#         hubupproj = hubupproj + zenith/np.linalg.norm(zenith)
#     if theta > 0:
#         hubupproj = hubupproj - zenith/np.linalg.norm(zenith)
#     print(hubupproj[0], hubintelproj[1]], [hubupproj[0], hubintelproj[1])
#     n_array = np.array([[hubupproj[0], hubintelproj[1]], [hubupproj[0], hubintelproj[1]]]) 
#     print(np.arctan2(np.vdot(hubintelproj, hubupproj), np.linalg.det(n_array)))
# =============================================================================



# =============================================================================
# satpos = sat.at(t).position.au
# sunpos = sun.at(t).position.au
# earthpos = earth.at(t).position.au
# 
# satpos = Barycentric(satpos + earthpos).position.au
# satsun = sunpos - satpos
# satearth = earthpos - satpos
# =============================================================================
