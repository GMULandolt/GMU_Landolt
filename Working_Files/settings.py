import json

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
             "start": "",
             "end": ""}

        for k, v in d.items():
            setattr(self, k, v) 
        
        with open('settings.json') as f:
            variables = json.load(f)
        for key, value in variables.items():
            setattr(self, key, value)

parameters = Settings()