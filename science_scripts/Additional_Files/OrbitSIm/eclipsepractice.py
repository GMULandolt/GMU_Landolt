from sgp4.api import Satrec
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.positionlib import Barycentric
import numpy as np
import math


# Load timescale
ts = load.timescale()
eph = load('de421.bsp')
sun, earth = eph['sun'], eph['earth']

s = '1 25544U 98067A   24240.72922619  .00022094  00000-0  39156-3 0  9994'
t = '2 25544  51.6394 320.8864 0006271 280.6466 218.1101 15.50162965469642'
ascent = Satrec.twoline2rv(s, t)

# Convert Satrec object to EarthSatellite object
sat = EarthSatellite.from_satrec(ascent, ts)

# Define the location of GMU Observatory
obs = wgs84.latlon(38.8282, -77.3053, 140)
# Vector between sat and obs
difference = sat - obs


t = ts.now()


satpos = sat.at(t).position.au
sunpos = sun.at(t).position.au
earthpos = earth.at(t).position.au

satpos = Barycentric(satpos + earthpos).position.au
satsun = sunpos - satpos
satearth = earthpos - satpos
satsundist = np.linalg.norm(satsun)
satearthdist = np.linalg.norm(satearth)

asun = math.atan(0.00465047/satsundist)
aearth = math.atan(4.26354e-5/satearthdist)
theta = math.acos(np.vdot(satsun, satearth)/(satearthdist * satsundist))    
sunsolid = math.pi * asun ** 2
earthsolid = math.pi * aearth ** 2

if (asun > theta + aearth):
    print(earthsolid/sunsolid * 100)
elif (theta > asun + aearth or math.isclose(asun + aearth, theta)):
    print("0.00")
elif (aearth > asun + theta or math.isclose(asun + theta, aearth)):
    print("100.00")
else:
    a = (math.cos(aearth) - (math.cos(asun)*math.cos(theta)))/math.sin(theta)
    b = (math.sin(asun)**2 - a**2)**0.5
    
    p1 = np.array([0, 0, 1])
    p2 = np.array([math.sin(theta), 0, math.cos(theta)])
    p3 = np.array([a, -b, math.cos(asun)])
    p4 = np.array([a, b, math.cos(asun)])
            
    nb = np.cross(p1, p4 - p1)/np.linalg.norm(np.cross(p1, p4 - p1))
    nc = np.cross(p1, p3 - p1)/np.linalg.norm(np.cross(p1, p3 - p1))
    phi1 = math.acos(np.vdot(nb, nc))
    nb = np.cross(p2, p4 - p2)/np.linalg.norm(np.cross(p2, p4 - p2))
    nc = np.cross(p2, p3 - p2)/np.linalg.norm(np.cross(p2, p3 - p2))
    phi2 = math.acos(np.vdot(nb, nc))
    nb = np.cross(p4, p1 - p4)/np.linalg.norm(np.cross(p4, p1 - p4))
    nc = np.cross(p4, p3 - p4)/np.linalg.norm(np.cross(p4, p3 - p4))
    psi1 = math.acos(np.vdot(nb, nc))
    nb = np.cross(p4, p2 - p4)/np.linalg.norm(np.cross(p4, p2 - p4))
    nc = np.cross(p4, p3 - p4)/np.linalg.norm(np.cross(p4, p3 - p4))
    psi2 = math.acos(np.vdot(nb, nc))
            
    if (a < 0):
        digon = (2*math.pi-phi1)*(1-math.cos(asun)) + phi1 +2*psi1 - math.pi + phi2*(1-math.cos(aearth)) - (phi2+2*psi2-math.pi)
    else:
        digon = 2*math.pi - 2*(psi1 + psi2) - phi1 * math.cos(asun) - phi2 * math.cos(aearth)
    
    print(digon/sunsolid * 100)