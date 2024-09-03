import numpy as np
import re
import datetime
from typing import Tuple, Union, Optional

from scipy.optimize import minimize_scalar
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5, AltAz, get_sun, EarthLocation
from astropy.time import Time
from numba import jit, njit, prange

from . import time_utils


@njit(parallel=True)
def _internal_altaz_to_radec(azimuth, altitude, latitude, lst, refraction):
    if refraction:
        pressure = 760
        temperature = 10
        if altitude >= 15.0:
            arg = (90.0 - altitude) * np.pi / 180.0
        elif altitude >= 0.0:
            arg = (90.0 - 15.0) * np.pi / 180.0
        else:
            arg = 0
        dalt = np.tan(arg)
        dalt = 58.294 * dalt - 0.0668 * dalt ** 3
        dalt /= 3600.
        dalt = dalt * (pressure / 760.) * (283. / (273. + temperature))

        altitude -= dalt

    alt = altitude % 360
    if alt > 180:
        alt -= 360
    az = azimuth % 360
    alt_r = alt * np.pi/180
    az_r = az * np.pi/180
    lat_r = latitude * np.pi/180

    HA_r = np.arctan2(-np.sin(az_r) * np.cos(alt_r),
                      np.cos(lat_r) * np.sin(alt_r) - np.sin(lat_r) * np.cos(alt_r) * np.cos(az_r))
    dec_r = np.arcsin(np.sin(lat_r) * np.sin(alt_r) + np.cos(lat_r) * np.cos(alt_r) * np.cos(az_r))
    ha = (np.degrees(HA_r) / 15) % 24
    if ha > 12:
        ha -= 24
    dec = (np.degrees(dec_r)) % 360
    if dec > 180:
        dec -= 360

    if dec > 90:
        hourangle = (ha + 12) % 24
        if hourangle > 12:
            hourangle -= 24
        declination = 180 - dec
    elif dec < -90:
        hourangle = (ha + 12) % 24
        if hourangle > 12:
            hourangle -= 24
        declination = dec + 180
    else:
        hourangle = ha
        declination = dec

    ra = (lst - hourangle) % 24
    return ra, declination


def convert_altaz_to_radec(azimuth: float, altitude: float, latitude: float, longitude: float,
                           time: datetime.datetime, leap_seconds: float = 0, refraction = True) -> Tuple[float, float]:
    """
    Convert horizontal coordinates to celestial coordinates.
    Formulas gathered from the following references.
    References:
        1.  K. Collins and J. Kielkopf, “Astroimagej: Imagej for astronomy,” (2013). Astrophysics source code library.
            https://github.com/karenacollins/AstroImageJ.

    Parameters
    ----------
    azimuth : FLOAT
        Given azimuth of target, degrees.
    altitude : FLOAT
        Given altitude of target, degrees.
    latitude : FLOAT
        Latitude of observatory, degrees North.
    longitude : FLOAT
        Longitude of observatory, degrees East.
    time : datetime.datetime object
        Time to be converted.
    leap_seconds : INT
        Leap second offset between TAI and TT.
    refraction : BOOL
        Whether or not to correct for atmospheric refraction.

    Returns
    -------
    ra : FLOAT
        Calculated right ascension of target.
    declination : FLOAT
        Calculated declination of target.
    """
    lst = time_utils.get_local_sidereal_time(longitude, time, leap_seconds)
    ra, dec = _internal_altaz_to_radec(azimuth, altitude, latitude, lst, refraction)
    return ra, dec


@njit(parallel=True)
def _internal_radec_to_altaz(ra, dec, latitude, longitude, lst, refraction):
    ha = (lst - ra) % 24
    if ha > 12:
        ha -= 24
    # Convert to degrees
    ha *= 15
    dec_r = dec * np.pi/180
    latitude_r = latitude * np.pi/180
    longitude_r = longitude * np.pi/180
    HA_r = ha * np.pi/180

    alt_r = np.arcsin(np.sin(dec_r) * np.sin(latitude_r) + np.cos(dec_r) * np.cos(latitude_r) * np.cos(HA_r))
    az_r = np.arctan2(-np.cos(dec_r) * np.sin(HA_r),
                      np.sin(dec_r) * np.cos(latitude_r) - np.sin(latitude_r) * np.cos(dec_r) * np.cos(HA_r))
    az = az_r * 180/np.pi
    alt = alt_r * 180/np.pi
    az = az % 360

    if refraction:
        pressure = 760
        temperature = 10
        if alt >= 15.0:
            arg = (90.0 - alt) * np.pi / 180
        elif alt >= 0.0:
            arg = (90.0 - 15.0) * np.pi / 180
        else:
            return az, alt
        dalt = np.tan(arg)
        dalt = 58.276 * dalt - 0.0824 * dalt ** 3
        dalt /= 3600.
        dalt = dalt * (pressure / 760.) * (283. / (273. + temperature))
        alt += dalt
    return az, alt


def convert_radec_to_altaz(ra: float, dec: float, latitude: float, longitude: float,
                           time: datetime.datetime, leap_seconds: float = 0, refraction=True) -> Tuple[float, float]:
    """
    Convert celestial coordinates to horizontal coordinates.
    Formulas gathered from the following references.
    References:
        1.  K. Collins and J. Kielkopf, “Astroimagej: Imagej for astronomy,” (2013). Astrophysics source code library.
            https://github.com/karenacollins/AstroImageJ.

    Parameters
    ----------
    ra : FLOAT
        Given right ascension of target.
    dec : FLOAT
        Given declination of target.
    latitude : FLOAT
        Latitude of observatory.
    longitude : FLOAT
        Longitude of observatory.
    time : datetime.datetime object
        Time to be converted.
    leap_seconds : INT
        Leap second offset between TAI and TT.
    refraction : BOOL
        Whether or not to correct for atmospheric refraction.

    Returns
    -------
    az : FLOAT
        Calculated azimuth of target.
    alt : FLOAT
        Calculated altitude of target.
    """
    lst = time_utils.get_local_sidereal_time(longitude, time, leap_seconds)
    az, alt = _internal_radec_to_altaz(ra, dec, latitude, longitude, lst, refraction)
    return az, alt


def convert_j2000_to_apparent(ra: float, dec: float) -> Tuple[float, float]:
    """
    Parameters
    ----------
    ra : FLOAT
        Right ascension to be converted from J2000 coordinates to apparent
        coordinates.
    dec : FLOAT
        Declination to be converted from J2000 coordinates to apparent coordinates.

    Returns
    -------
    coords_apparent.ra.hour: FLOAT
        Right ascension of target in local topocentric coordinates ("JNow").
    coords_apparent.dec.degree: FLOAT
        Declination of target in local topocentric coordinates ("JNow").
    """
    obstime = Time(datetime.datetime.now(datetime.timezone.utc))
    # Start with ICRS
    coords_j2000 = SkyCoord(ra=ra*u.hourangle, dec=dec*u.degree, frame='icrs')
    # Convert to FK5 (close enough to ICRS) with equinox at current time
    coords_apparent = coords_j2000.transform_to(FK5(equinox=obstime))
    return coords_apparent.ra.hour, coords_apparent.dec.degree


def convert_apparent_to_j2000(ra: float, dec: float) -> Tuple[float, float]:
    """
    Parameters
    ----------
    ra : FLOAT
        Right ascension to be converted from apparent to J2000 coordinates.
        coordinates.
    dec : FLOAT
        Declination to be converted from apparent to J2000 coordinates.

    Returns
    -------
    coords_j2000.ra.hour: FLOAT
        Right ascension of target in J2000.
    coords_j2000.dec.degree: FLOAT
        Declination of target in J2000.
    """
    obstime = Time(datetime.datetime.now(datetime.timezone.utc))
    # Start with FK5 (close enough to ICRS) with equinox at apparent time
    coords_apparent = SkyCoord(ra=ra*u.hourangle, dec=dec*u.degree, frame=FK5(equinox=obstime))
    # ICRS Equinox is always J2000
    coords_j2000 = coords_apparent.transform_to('icrs')
    return coords_j2000.ra.hour, coords_j2000.dec.degree


def get_sun_elevation(time: Union[str, datetime.datetime], latitude: float, longitude: float) -> float:
    """
    Parameters
    ----------
    time : datetime.datetime object
        Time to get sun elevation for.
    latitude : FLOAT
        Latitude at which to calculate sun elevation.
    longitude : FLOAT
        Longitude at which to calculate sun elevation.

    Returns
    -------
    alt : FLOAT
        Degrees above/below the horizon that the Sun is located at for the specified
        time at the specified coordinates.  Negative = below horizon.

    """
    if type(time) is not datetime.datetime:
        time = time_utils.convert_to_datetime_utc(time)
    astrotime = Time(time, format='datetime', scale='utc')
    coords = get_sun(astrotime)
    (az, alt) = convert_radec_to_altaz(float(coords.ra.hour), float(coords.dec.degree), latitude, longitude, time)
    return alt


def get_sunset(day: Union[str, datetime.datetime], latitude: float, longitude: float) -> Optional[datetime.datetime]:
    """
    Parameters
    ----------
    day : datetime.datetime object
        Day to calculate sunset time for.
    latitude : FLOAT
        Latitude at which to calculate sunset time.
    longitude : FLOAT
        Longitude at which to calculate sunset time.

    Returns
    -------
    datetime.datetime object
        Time that the Sun will set
        below the horizon for the specified day at the specified
        coordinates.

    """
    if type(day) is not datetime.datetime:
        day = time_utils.convert_to_datetime(day)

    def sunalt12(hours):
        hms = sexagesimal(hours)
        h, m, s = hms.split(':')
        return (get_sun_elevation(day.replace(hour=int(h), minute=int(m), second=int(float(s))), latitude, longitude) + 12)**2

    sunset_hours = minimize_scalar(sunalt12, bounds=(12, 23), method='bounded')['x']
    hour, minute, second = sexagesimal(sunset_hours).split(':')
    return day.replace(hour=int(hour), minute=int(minute), second=int(float(second)), tzinfo=datetime.timezone.utc) - day.utcoffset()


@njit(parallel=True)
def airmass(altitude: float) -> float:
    return 1/np.cos(np.pi/2 - np.radians(altitude))


@njit
def truncate(number, digits) -> float:
    stepper = np.power(10, digits)
    return int(stepper * number) / stepper


def sexagesimal(decimal: float, precision=5) -> str:
    hh = int(decimal)
    f1 = hh if hh != 0 else 1

    extra = decimal % f1
    if f1 == 1 and decimal < 0:
        extra -= 1
    mm = int(extra * 60)
    f2 = mm if mm != 0 else 1

    extra2 = (extra * 60) % f2
    if f2 == 1 and (extra * 60) < 0:
        extra2 -= 1
    ss = extra2 * 60

    hh = abs(hh)
    mm = abs(mm)
    ss = abs(ss)

    ss = truncate(ss, precision)
    fmt = '{:02d}:{:02d}:{:0%d.%df}' % (precision+3, precision)
    sign = '-' if decimal < 0 else ''
    return sign + fmt.format(hh, mm, ss)


def decimal(sexagesimal: str) -> float:
    splitter = 'd|h|m|s|:| '
    valtup = re.split(splitter, sexagesimal)
    hh, mm, ss = float(valtup[0]), float(valtup[1]), float(valtup[2])
    if hh > 0 or valtup[0] == '+00' or valtup[0] == '00':
        return hh + mm/60 + ss/3600
    elif hh < 0 or valtup[0] == '-00':
        return hh - mm/60 - ss/3600