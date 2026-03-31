from observatory_init import observatoryInitList
from observatory_characteristics import ObservatoryCharacteristics
from observatory_availability import ObservatoryAvailability
from observatory import Observatory
from datetime import datetime
import csv

class ObservatoryRepository():
    def __init__(self):
        self.observatoryList = []

    def getObservatoryByName(self, name: str): 
        for obsItem in self.observatoryList:
            obs: Observatory = obsItem
            if (obs.characteristics.name == name):
                return obs

    def observatoryInitFromList(self):
        self.observatoryList = []

        for item in observatoryInitList: 
            oc = ObservatoryCharacteristics(item["name"], item["latitude"], item["longitude"], item["timezone"], item["diameter"], item["elevation"], item["light_pollution"])
            obs = Observatory(oc)
            self.observatoryList.append(obs)

    def exportObservatoryCharacteristicsToCSV(self, filename: str):
        f = open(filename, 'w')
        writer = csv.writer(f)
        header = ['name', 'latitude', 'longitude', 'timezone', 'diameter', 'elevation', 'light_pollution']
        writer.writerow(header)
        for obs in self.observatoryList:
            obsChar: ObservatoryCharacteristics = obs.characteristics
            data = [obsChar.name, obsChar.latitude, obsChar.longitude, obsChar.timezone,obsChar.diameter, obsChar.elevation, obsChar.light_pollution]
            writer.writerow(data)

    def importObservatoryCharacteristicsFromCSV(self, filename: str):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            first = True
            for line in reader:
                if first:
                    first = False
                else:
                    oc = ObservatoryCharacteristics(line[0], float(line[1]), float(line[2]), line[3], float(line[4]), int(line[5]), int(line[6]))
                    obs = Observatory(oc)
                    self.observatoryList.append(obs)

    def exportObservatoryAvailabilityToCSV(self, filename: str):
        f = open(filename, 'w')
        writer = csv.writer(f)
        header = ['name', 'availability dates']
        writer.writerow(header)
        for obs in self.observatoryList:
            obs1: Observatory = obs
            obsChar: ObservatoryCharacteristics = obs1.characteristics
            obsAvail: ObservatoryAvailability = obs1.availability
            data = [obsChar.name] + obsAvail.dates
            writer.writerow(data)
    
    def importObservatoryAvailabilityFromCSV(self, filename: str):
        x = 0

    def generateRandomObservatoryAvailability(self, startDate: datetime, endDate: datetime):
        for obs in self.observatoryList:
            obs1: Observatory = obs
            obs1.availability.generateRandomAvailability(obs1.characteristics, startDate, endDate)
        
def main():
    orep = ObservatoryRepository()
    orep.importObservatoryCharacteristicsFromCSV("observatoryFileRepository/Observatory_Characteristics.csv")
    sd = datetime(2024, 1, 1)
    ed = datetime(2024, 2, 1)
    orep.generateRandomObservatoryAvailability(sd, ed)
    orep.exportObservatoryAvailabilityToCSV("observatoryFileRepository/Observatory_Availability.csv")
    #orep.observatoryInitFromList()
    #orep.exportObservatoryCharacteristicsToCSV("Observatory_Characteristics.csv")
    #print(orep.observatoryList)

if __name__ == "__main__":
    main()