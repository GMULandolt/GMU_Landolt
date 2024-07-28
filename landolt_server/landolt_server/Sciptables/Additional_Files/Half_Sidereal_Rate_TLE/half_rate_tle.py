"""
For Landolt Mission Observing Mode 3

Given an input satellite TLE, generates a new TLE with the satellite moving at half the angular velocity
in an orbit that contains RA/Dec coordinates approximately equal to those of the original satellite's
orbit (but at half the rate). Valid only for time intervals close to the specified time. Accuracy for
geostationary satellites is generally <0.5" in RA/Dec.

Can be run as a standalone script or imported as a function:
    >>> from half_rate_tle import generate_half_rate_tle
    >>> tle, error = generate_half_rate_tle(TLE)
"""

# Use the most recent TLE for best results: https://celestrak.org/NORAD/elements
TLE = """
INTELSAT 40E (IS-40E)
1 56174U 23052A   24184.95208998 -.00000173  00000+0  00000+0 0  9994
2 56174   0.0239  45.7585 0001817  93.6411  33.8124  1.00269386  4677
"""

from datetime import datetime, timezone, timedelta
import math
import random
from scipy.optimize import curve_fit
from sgp4.api import Satrec, WGS72
from sgp4.conveniences import jday_datetime
from sgp4.exporter import export_tle
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.positionlib import Geocentric


OBSERVER = wgs84.latlon(38.8282, -77.3053, 140)  # George Mason University Observatory

RATE_FACTOR = 0.5  # half the rate
DISTANCE_FACTOR = (1 / RATE_FACTOR) ** (2 / 3)  # halving angular velocity -> multiply radius by cube root of 4
SATNUM = random.randint(70000, 99999)
JD, FR = jday_datetime(datetime.now())
EPOCH = JD + FR - 2433281.5  # days since 1949 December 31 00:00 UT
NOW = datetime.now(timezone.utc)
TS = load.timescale()
T = TS.from_datetime(NOW)

EARTH_RADIUS_AU = 4.2632e-5
EARTH = 399  # NAIF code
TWO_PI = 2 * math.pi

TIME_DISPLACEMENT = 5 * 60  # seconds
TIME_INTERVAL = 20  # seconds    

#### Helper functions ####
def to_tle(sat, title):
    return (title + f" HALF {sat.epochdays:.4f}\n" if title else "") + "\n".join(export_tle(sat))


def get_topocentric(sat):
    skyfield_sat = EarthSatellite.from_satrec(sat, TS)
    difference = skyfield_sat - OBSERVER
    return difference.at(T)


def calc_radec(sat):
    return get_topocentric(sat).radec()  # ICRF


def calc_altaz(sat):
    return get_topocentric(sat).altaz()


def sat_init(bstar, ndot, nddot, argpo, nodeo, mo, ecco, inclo, no):
    half_sat = Satrec()
    half_sat.sgp4init(
        WGS72,                      # gravity model
        "i",                        # 'a' = old AFSPC mode, 'i' = improved mode
        SATNUM,                     # satnum: Satellite number
        EPOCH,                      # epoch: days since 1949 December 31 00:00 UT
        bstar,                      # bstar: drag coefficient (1/earth radii)
        ndot * RATE_FACTOR,         # ndot: ballistic coefficient (radians/minute^2)
        nddot * RATE_FACTOR,        # nddot: mean motion 2nd derivative (radians/minute^3)
        ecco,                       # ecco: eccentricity
        argpo,                      # argpo: argument of perigee (radians)
        inclo,                      # inclo: inclination (radians)
        mo,                         # mo: mean anomaly (radians)
        no,                         # no_kozai: mean motion (radians/minute)
        nodeo,                      # nodeo: right ascension of ascending node (radians)
    )
    return half_sat


def ra_dec_t(sat, ra_weight, dec_weight, distance_weight, 
             time_displacement=TIME_DISPLACEMENT, time_interval=TIME_INTERVAL, adjust_distance=False):
    """Returns scaled ra/dec/distance for satellite position optimization. Does not return actual ra/dec values."""
    t_start = NOW - timedelta(seconds=time_displacement)
    t_end = NOW + timedelta(seconds=time_displacement)
    t = t_start

    sat = EarthSatellite.from_satrec(sat, TS)
    difference = sat - OBSERVER

    out = []
    while t <= t_end:
        ra, dec, distance = difference.at(TS.from_datetime(t)).radec()
        out.append(ra.radians * ra_weight)
        out.append(dec.radians * dec_weight)
        out.append(((distance.au + EARTH_RADIUS_AU) * DISTANCE_FACTOR - EARTH_RADIUS_AU if adjust_distance else distance.au) * distance_weight)
        t += timedelta(seconds=time_interval)
    return out


def apply_limits(mo, ecco, inclo, no):
    return (
        mo % TWO_PI,
        ecco,
        abs(inclo),
        no
    )

def ra_dec_t_sat(xdata, mo, ecco, inclo, no):
    ra_weight, dec_weight, distance_weight, bstar, ndot, nddot, argpo, nodeo = xdata
    sat = sat_init(bstar, ndot, nddot, argpo, nodeo, mo, ecco, inclo, no)
    return ra_dec_t(sat, ra_weight, dec_weight, distance_weight, 
                    time_displacement=TIME_DISPLACEMENT * RATE_FACTOR, time_interval=TIME_INTERVAL * RATE_FACTOR)


def generate_half_rate_tle(tle: str, now: datetime = None) -> tuple[str, tuple[float, float, float]]:
    """
    Generates a TLE for a satellite moving at half the angular velocity of the input satellite. 
    The orbit contains RA/Dec coordinates approximately equal to those of the original satellite's 
    orbit (but at half the rate). Valid only for time intervals close to the specified date/time.

    Args:
        tle (str): The input satellite TLE.
        now (datetime.datetime, optional): The date/time to generate the TLE at. Defaults to the current date/time.
            The generated satellite will be positioned to match the input satellite's location at this time.
    
    Returns:
        tle (str): The generated half-rate satellite TLE 
        error (tuple[float, float, float]): The error in RA/Dec/Distance between the original and generated satellite.
    """
    if now:
        global NOW, T
        NOW = now
        T = TS.from_datetime(NOW)
    print("Generating TLE for date/time:", NOW.isoformat())

    # Parse TLE
    tle = tle.strip().splitlines()
    title = None
    if len(tle) == 3:
        title = tle[0].strip()
        tle = tle[1:]

    # Initialize satellite
    sat = Satrec.twoline2rv(*tle)

    #### Procedure ####
    # Use SGP4 to calculate the real satellite's current location
    ra, dec, distance = calc_radec(sat)
    print()
    print("Real Satellite RA/Dec/Distance:     ", ra, '|', dec, '|', distance)

    # Convert to skyfield
    skyfield_sat = EarthSatellite.from_satrec(sat, TS)
    observer_to_sat_t = skyfield_sat - OBSERVER  # Vector function
    observer_to_sat = observer_to_sat_t.at(T)  # Vector

    # Extend the vector from the observer to the satellite. Farther away -> lower velocity.
    # Done from the observer's perspective so that RA/Dec coordinates stay the same.
    # Goal: find the inclination of the new orbit.
    observer_to_sat = Geocentric([c * DISTANCE_FACTOR for c in observer_to_sat.position.au], t=T)
    earth_to_observer = OBSERVER.at(T)
    earth_to_sat = Geocentric([a + b for a, b in zip(observer_to_sat.position.au, earth_to_observer.position.au)], t=T)
    lat, lon = wgs84.latlon_of(earth_to_sat)

    # SGP4 propagation - needed to get latest mean anomaly
    e, r, v = sat.sgp4(JD, FR)
    # To do it mathematically: mo = (sat.mo + sat.no_kozai * ((JD + FR) - (sat.jdsatepoch + sat.jdsatepochF)) * 24 * 60) % TWO_PI

    # Set orbital elements
    mo = sat.mm % TWO_PI
    inclo = abs(lat.radians)  # latitude = inclination
    nodeo = sat.nodeo
    argpo = sat.argpo
    ecco = sat.ecco
    no = sat.no_kozai * RATE_FACTOR  # mean motion (angular velocity) halved

    # Keep values in range
    if lat.radians < 0:
        nodeo = (sat.nodeo + math.pi) % TWO_PI
        argpo = (sat.argpo + math.pi) % TWO_PI

    # Weights for ra/dec/distance so they are weighted appropriately during optimization
    ra_weight = 1 / ra.radians
    dec_weight = 1 / dec.radians
    distance_weight = 1 / ((distance.au + EARTH_RADIUS_AU) * DISTANCE_FACTOR - EARTH_RADIUS_AU) / 10  # weight distance less

    # Optimization - tweak some of the orbital elements to more closely match the real satellite's past and future RA/Dec
    xdata = (ra_weight, dec_weight, distance_weight, sat.bstar, sat.ndot, sat.nddot, argpo, nodeo)  # Fixed parameters
    ydata = ra_dec_t(sat, ra_weight, dec_weight, distance_weight, adjust_distance=True)  # Real satellite's ra/dec/distance
    p0 = (mo, ecco, inclo, no)  # Initial guess - parameters to optimize
    # Geostationary
    # bounds = (
    #     (mo - 0.2, 0,           inclo - 0.05, no - 0.00005),
    #     (mo + 0.2, ecco + 0.01, inclo + 0.05, no + 0.00005),
    # )
    # LEO
    # bounds = (
    #     (mo - 0.5, 0, inclo - 0.5, no - 0.0001),
    #     (mo + 0.5, 1, inclo + 0.5, no + 0.0001),
    # )
    bounds = (
        (0,      0, 0,       no - 0.00005),
        (TWO_PI, 1, math.pi, no + 0.00005)
    )

    try:
        out, pcov = curve_fit(ra_dec_t_sat, xdata, ydata, p0, bounds=bounds)
        out = apply_limits(*out)
    except RuntimeError:
        print()
        print("WARNING:")
        print("Optimization failed. This is likely because the input TLE is out of date. Check for a more recent TLE.")
        print("Attempting to continue with original parameter values. Accuracy will be reduced.")
        print()
        out = p0

    # Output
    half_sat = sat_init(sat.bstar, sat.ndot, sat.nddot, argpo, nodeo, *out)
    half_ra, half_dec, half_distance = calc_radec(half_sat)
    print("Half-rate Satellite RA/Dec/Distance:", half_ra, "|", half_dec, "|", half_distance)
    print()

    print("Original parameters:", p0)
    print("Fitted parameters:  ", out)
    error = ra.radians - half_ra.radians, dec.radians - half_dec.radians, distance.au - half_distance.au
    print("Error (RA/Dec/Distance):", error)

    print()
    print("Half-rate Satellite TLE:")
    print("---------------------------------")

    half_sat_tle = to_tle(half_sat, title)
    print(half_sat_tle)
    return half_sat_tle, error


if __name__ == "__main__":
    generate_half_rate_tle(TLE)
