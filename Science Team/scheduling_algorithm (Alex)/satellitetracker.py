from observatory_characteristics import ObservatoryCharacteristics
import time

import datetime
import numpy as np
from sgp4.api import Satrec, WGS72
from skyfield.api import load, EarthSatellite, wgs84
from datetime import datetime
from skyfield.api import load
from pytz import timezone
import pandas as pd
import re

ts = load.timescale()

class SatelliteSettings:
    def __init__(self):
        self.epoch = 0
        self.bstar = 0
        self.ndot = 0
        self.nddot = 0
        self.ecco = 0
        self.argpo = 0
        self.inclo = 0
        self.mo = 0
        self.no_kozai = 0.00437526951
        self.nodeo = 0
        self.tle1 = "NA"
        self.tle2 = "NA"

        d = {
             "nodeo": 0,
             "timezone": 'US/Eastern',
             "start": "2000-01-01 00:00:00",
             "end": "2000-01-01 00:00:1",
             "elev": 140,
             "tdelta": 10,
             "chunks": 100,
             #settings
             "start": "2024-01-01 00:00:00",
             "end": "2024-12-31 00:00:00",
             "tdelta": 3600000,
             "lat": 38.8282, 
             "lon": -77.3053}

        for k, v in d.items():
            setattr(self, k, v) 
        
        stimes = re.split("[-\s:]", self.start)
        itimes = [int(i) for i in stimes]
        time = datetime(itimes[0], itimes[1], itimes[2], itimes[3], itimes[4], itimes[5])
        time = ts.from_datetime(timezone(self.timezone).localize(time))
        setattr(self, "start", time)
        stimes = re.split("[-\s:]", self.end)
        itimes = [int(i) for i in stimes]
        time = datetime(itimes[0], itimes[1], itimes[2], itimes[3], itimes[4], itimes[5])
        time = ts.from_datetime(timezone(self.timezone).localize(time))
        setattr(self, "end", time)



class SatelliteTracker: 

    def __init__(self): 
        self.satSettings = SatelliteSettings()
        # Initialize satellite using SGP4
        self.sat = Satrec()
        self.sat.sgp4init(
            WGS72,           # gravity model
            'i',             # 'a' = old AFSPC mode, 'i' = improved mode
            1,               # satnum: Satellite number
            self.satSettings.epoch,               # epoch: days since 1949 December 31 00:00 UT
            self.satSettings.bstar,           # bstar: drag coefficient (/earth radii)
            self.satSettings.ndot,           # ndot: ballistic coefficient (radians/minute^2)
            self.satSettings.nddot,             # nddot: second derivative of mean motion (radians/minute^3)
            self.satSettings.ecco,             # ecco: eccentricity
            self.satSettings.argpo,               # argpo: argument of perigee (radians)
            self.satSettings.inclo,               # inclo: inclination (radians)
            self.satSettings.mo,               # mo: mean anomaly (radians)
            self.satSettings.no_kozai,  # no_kozai: mean motion (radians/minute) GEO
        #    0.04908738521, #LEO
        #    0.00872664625, #meo
            self.satSettings.nodeo                # nodeo: right ascension of ascending node (radians)
        )
        if (self.satSettings.tle1 != "NA" or self.satSettings.tle2 != "NA"):
            self.sat = Satrec.twoline2rv(self.satSettings.tle1, self.satSettings.tle2)
        # Convert Satrec object to EarthSatellite object
        self.sat = EarthSatellite.from_satrec(self.sat, ts)
    
    def computeAltitude(self, obs: ObservatoryCharacteristics, dt: datetime):
        # Define the location of GMU Observatory
        obsLocation = wgs84.latlon(obs.latitude, obs.longitude, obs.elevation)
        # Vector between sat and obs
        difference = self.sat - obsLocation
        # convert time
        t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

        satcord = self.sat.at(t)
        obscoord = obsLocation.at(t)        

        topocentric = difference.at(t)
        ra, dec, distance = topocentric.radec()
        alt, az, trash = topocentric.altaz()
        return alt
    
    def checkAltitude(self, obs:ObservatoryCharacteristics, dt:datetime): 
        if self.computeAltitude(obs, dt).degrees > 20:
            return True 
        else:
            return False
        

