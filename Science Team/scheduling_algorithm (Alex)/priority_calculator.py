from observatory import Observatory
from datetime import datetime
from satellitetracker import SatelliteTracker

class LongTermPriorityCalculator:
    maxDiameter = 10.0
    maxAltitude = 90.0
    maxElevation = 5035.0
    maxLightPollution = 14.0

    def __init__ (self):
        self.diameterWeight = 0.5
        self.altitudeWeight = 0.25
        self.elevationWeight = 0.2
        self.lightPollutionWeight = 0.05                               

    def computeLongTermPriority(self, obs: Observatory, dt: datetime): 
        priorityValue = 0 
        if obs.characteristics.diameter == None:
            print("no diameter for" + obs.characteristics.name)
        diameterValue = self.diameterWeight * obs.characteristics.diameter / self.maxDiameter

        st = SatelliteTracker() 
        altitudeValue = self.altitudeWeight * st.computeAltitude(obs.characteristics, dt).degrees / self.maxAltitude

        elevationValue = self.elevationWeight * obs.characteristics.elevation / self.maxElevation
        lightPollutionValue = self.lightPollutionWeight * (1 - obs.characteristics.light_pollution / self.maxLightPollution)
        
        priorityValue = diameterValue + altitudeValue + elevationValue + lightPollutionValue

        return priorityValue
    
 

class ShortTermPriorityCalculator:
    maxDiameter = 10.0
    maxAltitude = 90.0
    maxElevation = 5035.0
    maxLightPollution = 14.0

    def __init__ (self):
        self.diameterWeight = 0.4
        self.altitudeWeight = 0.2
        self.weatherWeight = 0.25
        self.elevationWeight = 0.1
        self.lightPollutionWeight = 0.05                            

    def computeShortTermPriority(self, obs: Observatory, currentDT: datetime, forecastDT: datetime): 
        priorityValue = 0 
        diameterValue = self.diameterWeight * obs.characteristics.diameter / self.maxDiameter

        st = SatelliteTracker() 
        altitudeValue = self.altitudeWeight * st.computeAltitude(obs, forecastDT).degrees / self.maxAltitude

        weatherValue = obs.forecast.extractForecast(currentDT, forecastDT)

        elevationValue = self.elevationWeight * obs.characteristics.elevation / self.maxElevation
        lightPollutionValue = self.lightPollutionWeight * (1 - obs.characteristics.light_pollution / self.maxLightPollution)
        
        priorityValue = diameterValue + altitudeValue + weatherValue + elevationValue + lightPollutionValue

        return priorityValue
    
 

