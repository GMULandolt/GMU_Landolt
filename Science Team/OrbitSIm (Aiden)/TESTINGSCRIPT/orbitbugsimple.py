from skyfield.api import load, EarthSatellite, wgs84
from sgp4.api import Satrec, WGS72
import numpy as np
from pytz import timezone

inclin = np.linspace(-np.pi/4, np.pi/4, 10)
nodeo = np.linspace(0, np.pi, 10)
for inc in inclin:
    for nod in nodeo:
        #ISSUE 1
        sat = Satrec()
        sat.sgp4init(WGS72, 'i', 1, 27395, 0, 0, 0, 0, 0, inc, 0, 0.004375234196, nod)

        ts = load.timescale()
        sat = EarthSatellite.from_satrec(sat, ts)
        obs = wgs84.latlon(38.8282, -77.3053, 140)
        difference = sat - obs
        t = ts.utc(2025, 1, 1, 5+np.arange(0, 168), 0, 0)
        topocentric = difference.at(t)
        ra, dec, distance = topocentric.radec()
        ra = ra._degrees
        dec = dec._degrees
        print("Issue 1:")
        max= 50
        for i in range(1, len(t)):
            if ra[i] - ra[i-1] > max:
                max = ra[i] - ra[i-1]
                for j in range(-5, 5):
                    print(t[i+j].astimezone(timezone('US/Eastern')), ra[i+j], dec[i+j])
                print(max)
                print(inc, nod  )
                print()