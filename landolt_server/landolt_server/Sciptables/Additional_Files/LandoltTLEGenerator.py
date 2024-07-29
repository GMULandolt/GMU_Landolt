from sgp4.api import Satrec, WGS72
from sgp4 import exporter


# Initialize satellite using SGP4
sat = Satrec()
sat.sgp4init(
    WGS72,           # gravity model
    'i',             # 'a' = old AFSPC mode, 'i' = improved mode
    83379,               # satnum: Satellite number
    24156,               # epoch: days since 1949 December 31 00:00 UT
    0,           # bstar: drag coefficient (/earth radii)
    0,           # ndot: ballistic coefficient (radians/minute^2)
    0.0,             # nddot: second derivative of mean motion (radians/minute^3)
    0,             # ecco: eccentricity
    0,               # argpo: argument of perigee (radians)
    0,               # inclo: inclination (radians)
    0,               # mo: mean anomaly (radians)
    0.00437526951 ,  # no_kozai: mean motion (radians/minute) GEO
#    0.04908738521, #LEO
#    0.00872664625, #meo
    0                # nodeo: right ascension of ascending node (radians)
)

tle1, tle2 = exporter.export_tle(sat)
print(tle1)
print(tle2)