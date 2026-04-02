from observatory_availability import ObservatoryAvailability
from observatory_calibrations import ObservatoryCalibrations
from observatory_characteristics import ObservatoryCharacteristics
from observatory_forecast import ObservatoryForecast

class Observatory():
    def __init__(self, characteristics: ObservatoryCharacteristics):
        self.characteristics: ObservatoryCharacteristics = characteristics
        self.availability = ObservatoryAvailability()
        self.calibrations = ObservatoryCalibrations()
        self.forecast = ObservatoryForecast()

