import json
from datetime import datetime
from pytz import timezone
from skyfield.api import load
ts = load.timescale()
import re

class Settings:
    def __init__(self):
        
        d = {"epoch": 0,
             "bstar": 0,
             "ndot": 0,
             "nddot": 0,
             "ecco": 0,
             "argpo": 0,
             "inclo": 0,
             "mo": 0,
             "no_kozai": 0.00437526951,
             "nodeo": 0,
             "timezone": 'US/Eastern',
             "start": "2000-01-01 00:00:00",
             "end": "2000-01-01 00:00:1",
             "lat": 38.8282,
             "lon": -77.3053,
             "elev": 140,
             "tdelta": 10,
             "chunks": 100,
             "tle1": "NA",
             "tle2": "NA"}

        for k, v in d.items():
            setattr(self, k, v) 
        
        with open('settings.json') as f:
            variables = json.load(f)
        for key, value in variables.items():
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