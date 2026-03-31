import json
from datetime import datetime
from pytz import timezone
from skyfield.api import load
ts = load.timescale()
import re
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class Settings:
    def __init__(self):
        
        d = {"epoch": 27577,
             "bstar": 0,
             "ndot": 0,
             "nddot": 0,
             "ecco": 0,
             "argpo": 0,
             "inclo": 0,
             "mo": 0,
             "no_kozai": 0.00437519126,
             "nodeo": 0,
             "timezone": 'US/Eastern',
             "start": "2000-01-01 00:00:00",
             "end": "2000-01-01 00:00:00",
             "lat": 38.8282,
             "lon": -77.3053,
             "elev": 140,
             "tdelta": 10,
             "chunks": 100,
             "tle1": "NA",
             "tle2": "NA",
             "t_eff": 0.5,
             "ccd_eff": 0.8,
             "t_diam": 0.8128,
             "beta": 0,
             "n": 0,
             "lat_loc": "38.8308",
             "lon_loc": "-77.3075",
             "humidity": "0.5"}

        for k, v in d.items():
            setattr(self, k, v) 
        
        with open(os.path.join(__location__,'settings.json')) as f:
            variables = json.load(f)
        for key, value in variables.items():
            if value is not None and value != "":
                setattr(self, key, value)
        
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

parameters = Settings()