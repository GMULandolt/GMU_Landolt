from sgp4.api import Satrec
from skyfield.api import load, EarthSatellite, wgs84


# Load timescale
ts = load.timescale()

s = '1 36032U 09058A   24150.55957466  .00000076  00000-0  00000-0 0  9998'
t = '2 36032   0.0894  86.1791 0004123 349.1098  70.7220  1.00270718 53410'
ascent = Satrec.twoline2rv(s, t)

# Convert Satrec object to EarthSatellite object
sat = EarthSatellite.from_satrec(ascent, ts)

# Define the location of GMU Observatory
obs = wgs84.latlon(38.8282, -77.3053, 140)
# Vector between sat and obs
difference = sat - obs


t = ts.now()

topocentric = difference.at(t)
ra, dec, distance = topocentric.radec()
alt, az, trash = topocentric.altaz()
          
print({'RA (Deg)': str(ra), 'Dec (Deg)': str(dec),
       'Az (Deg)': az.degrees, 'Alt (Deg)': alt.degrees,
       'Distance (Km)': distance.km})