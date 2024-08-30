from observatory_characteristics import ObservatoryCharacteristics
from datetime import datetime
from datetime import timedelta
import random 
import time

class AvailabilityPriorityCalculator:
    maxDiameter = 10.0
    maxAltitude = 90.0
    maxElevation = 5035.0
    maxLightPollution = 14.0

    def __init__ (self):
        self.diameterWeight = 0.5
        self.elevationWeight = 0.3
        self.lightPollutionWeight = 0.2 

    def computeAvailabilityPriority(self, obs: ObservatoryCharacteristics):
        availabilityPriorityValue = 0 

        diameterValue = self.diameterWeight * obs.diameter / self.maxDiameter
        elevationValue = self.elevationWeight * obs.elevation / self.maxElevation
        lightPollutionValue = self.lightPollutionWeight * (1 - obs.light_pollution / self.maxLightPollution)
                                                           
        availabilityPriorityValue = diameterValue + elevationValue + lightPollutionValue

        return availabilityPriorityValue   

class ObservatoryAvailability: 
    def __init__(self):
        self.dates = []
    
    def calculateDayRange(self, startDate: datetime, endDate: datetime):
        delta = endDate - startDate
        return delta.days

    def generateRandomAvailability(self, obs: ObservatoryCharacteristics, startDate: datetime, endDate: datetime):
        apc = AvailabilityPriorityCalculator()
        availabilityValue = apc.computeAvailabilityPriority(obs)
        dayrange = self.calculateDayRange(startDate, endDate)
        givenDate = startDate 
        self.dates = []

        for i in range(dayrange): 
            if (random.randint(0, 100) > (availabilityValue * 100)):
                self.dates.append(givenDate) 
            givenDate += timedelta(days=1)
        

    def checkAvailability(self, date: datetime):
        for itemObj in self.dates:
           item: datetime = itemObj
           if item.year == date.year and item.month == date.month and item.day == date.day: 
               return True
        return False


def main():
    print(1)

if __name__ == "__main__":
    main()