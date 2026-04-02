from datetime import datetime, timedelta
from observatory_repository import ObservatoryRepository
from observatory import Observatory
from priority_calculator import LongTermPriorityCalculator
from priority_calculator import ShortTermPriorityCalculator
from satellitetracker import SatelliteTracker
import csv
from queue import PriorityQueue
from weather_conditions_checking import WeatherConditions

class ScheduledObservation():
    def __init__(self, name: str, dt: datetime):
        self.name = name
        self.datetime = dt

class PerformedObservation():
    def __init__(self, name: str, dt: datetime, quality: int):
        self.name = name
        self.datetime = dt
        self.quality = quality

class Scheduler():
    shortTermWeatherThreshold = 10

    def __init__(self, orep: ObservatoryRepository, rolling: bool):
        self.schedule = []
        self.performed = []
        self.repository = orep
        self.lpc = LongTermPriorityCalculator()
        self.rolling = rolling
        self.lastObservation: datetime = None
        self.numberOfObservationsPerDay = 3.0
        self.lengthOfObservationSeconds = 3600.0 * 24.0 / self.numberOfObservationsPerDay
        self.weatherCondition = WeatherConditions()
    
    def addScheduledObservation(self, name: str, dt: datetime):
        so = ScheduledObservation(name, dt)
        self.schedule.append(so)
        self.lastObservation = dt

    def getScheduledObservation(self, dt: datetime):
        for item in self.schedule:
            so = ScheduledObservation(item)
            if so.datetime == dt:
                return self.repository.getObservatoryByName(so.name)

    def addPerformedObservation(self, name: str, dt: datetime, quality: int):
        po = PerformedObservation(name, dt, quality)
        self.performed.append(po)
        self.generateLongTermSingleSchedule(self, self.lastObservation + self.lengthOfObservationSeconds)
    
    def countObservations(self, name: str):
        count = 0
        for item in self.schedule:
            so: ScheduledObservation = item
            if so.name == name: 
                count += 1
        return count
    
    def exportScheduleToCSV(self, filename: str):
        f = open(filename, 'w')
        writer = csv.writer(f)
        header = ['date', 'name']
        writer.writerow(header)
        
        for item in self.schedule:
            so: ScheduledObservation = item
            data = [so.datetime, so.name]
            writer.writerow(data)

    def generateLongTermSingleSchedule(self, date: datetime):
        st = SatelliteTracker()
        lpc = LongTermPriorityCalculator()
        lpq = PriorityQueue()
        for obsItem in self.repository.observatoryList:
            obs: Observatory = obsItem
            pv = lpc.computeLongTermPriority(obs, date)
            pv = pv / (self.countObservations(obs.characteristics.name) + 1)
            lpq.put((1 - pv, obs.characteristics.name))

        #print sample priority list 

        #f = open("observatoryFileRepository/Priority_List.csv", "w")
        #writer = csv.writer(f)
        #header = ['priority value', 'name']
        #writer.writerow(header)
        #while lpq.empty()  == False:
        #    obsItem = lpq.get()
        #    obsName: str = obsItem[1]
        #    obs = self.repository.getObservatoryByName(obsName)
        #    data = [obsItem[0], obsItem[1]]
        #    writer.writerow(data)

        while lpq.empty()  == False:
            obsItem = lpq.get()
            obsName: str = obsItem[1]
            obs = self.repository.getObservatoryByName(obsName)
            if obs.availability.checkAvailability(date) and st.checkAltitude(obs.characteristics, date):
                self.addScheduledObservation(obs.characteristics.name, date)
                return
            
    def generateLongTermFixedSchedule(self, startDate: datetime, endDate: datetime):
        dayRange = (endDate - startDate).days
        givenDate = startDate

        for i in range(dayRange):
            for j in range(int(self.numberOfObservationsPerDay)):
                self.generateLongTermSingleSchedule(givenDate)
                givenDate = givenDate + timedelta(seconds = self.lengthOfObservationSeconds)

    def updateShortTermSingleSchedule(self, currentDT: datetime):
        spc = ShortTermPriorityCalculator()
        lpq = PriorityQueue()
        self.weatherCondition.updateRepositoryForecast(currentDT, self.repository)

        for obsItem in self.repository.observatoryList:
            obs: Observatory = obsItem
            pv = spc.computeShortTermPriority(obs.characteristics, currentDT)
            pv = pv / (self.countObservations(obs.characteristics.name) + 1)
            lpq.put((1 - pv, obs.characteristics.name))
    
