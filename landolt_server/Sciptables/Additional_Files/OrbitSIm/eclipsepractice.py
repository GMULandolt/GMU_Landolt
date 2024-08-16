from sgp4.api import Satrec
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.framelib import itrs
from skyfield.positionlib import Barycentric


# Load timescale
ts = load.timescale()
eph = load('de421.bsp')
sun, earth = eph['sun'], eph['earth']

s = '1 25544U 98067A   24229.50766309  .00023344  00000-0  41274-3 0  9992'
t = '2 25544  51.6410  16.4966 0005341 207.2907 263.9809 15.50198988467908'
ascent = Satrec.twoline2rv(s, t)

# Convert Satrec object to EarthSatellite object
sat = EarthSatellite.from_satrec(ascent, ts)

# Define the location of GMU Observatory
obs = wgs84.latlon(38.8282, -77.3053, 140)
# Vector between sat and obs
difference = sat - obs


t = ts.now()

satpos = Barycentric(sat.at(t).position.au)
print(satpos)
sunpos = sat.at(t).observe(sun)
print(sunpos.position)
print(satpos.position)