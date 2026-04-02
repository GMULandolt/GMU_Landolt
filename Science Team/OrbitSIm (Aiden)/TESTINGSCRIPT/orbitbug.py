from skyfield.api import load, EarthSatellite, wgs84
from sgp4.api import Satrec, WGS72
import numpy as np
from pytz import timezone


#ISSUE 1
sat = Satrec()
sat.sgp4init(WGS72, 'i', 1, 25000, 0, 0, 0, 0, 0, -0.5, 0, 0.00437526951, 2)

ts = load.timescale()
sat = EarthSatellite.from_satrec(sat, ts)
obs = wgs84.latlon(38.8282, -77.3053, 140)
difference = sat - obs
t = ts.utc(2025, 1, 3, 5+np.arange(0, 24), 0, 0)
topocentric = difference.at(t)
ra, dec, distance = topocentric.radec()
ra = ra._degrees
print("Issue 1:")
for i in range(len(t)):
    print(t[i].astimezone(timezone('US/Eastern')), ra[i])


print()


#ISSUE 2
sat = Satrec()
sat.sgp4init(WGS72, 'i', 1, 0, 0, 0, 0, 0, 0, -0.1, 0, 0.00437526951, 4)

ts = load.timescale()
sat = EarthSatellite.from_satrec(sat, ts)
obs = wgs84.latlon(38.8282, -77.3053, 140)
difference = sat - obs
t = ts.utc(2025, 1, 12, 5+np.arange(0, 24), 0, 0)
topocentric = difference.at(t)
ra, dec, distance = topocentric.radec()
ra = ra._degrees
print("Issue 2:")
for i in range(len(t)):
    print(t[i].astimezone(timezone('US/Eastern')), ra[i])