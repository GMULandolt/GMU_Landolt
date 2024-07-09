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
             "end": "2000-01-01 00:00:01",
             "tdelta": 1}

        for k, v in d.items():
            setattr(self, k, v) 
        
        with open('settings.json') as f:
            variables = json.load(f)
        for key, value in variables.items():
            setattr(self, key, value)
        
        stimes = re.split(self.start)
        setattr(self, "start", datetime(2014, 1, 18, 1, 35, 37.5))
        etimes = re.split(self.end)
        setattr(self, "end", self.end)

parameters = Settings()