from observatory_repository import ObservatoryRepository
from datetime import datetime
from scheduler import Scheduler

#In progress 

#run this to generate an updated short-term schedule

def main():
    orep = ObservatoryRepository()
    orep.importObservatoryCharacteristicsFromCSV("observatoryFileRepository/Observatory_Characteristics.csv")
    sd = datetime(2024, 1, 1)
    ed = datetime(2024, 2, 1)
    orep.generateRandomObservatoryAvailability(sd, ed)
    sc = Scheduler(orep, False)
    sc.generateLongTermFixedSchedule(sd, ed)
    sc.exportScheduleToCSV("observatoryFileRepository/Observatory_Short_Term_Schedule.csv")
    

if __name__ == "__main__":
    main()