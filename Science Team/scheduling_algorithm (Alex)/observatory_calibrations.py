from datetime import datetime

class ObservatoryCalibrations: 
    def __init__(self):
        self.dates = []

    def addCalibration(self, date: datetime):
        self.dates.append(date)

    def getCalibrationsNumber(self):
        return len(self.dates)

    def getCalibrationsList(self):
        return self.dates 
    
    def getLastCalibration(self):
        return self.dates[len(self.dates) - 1] 

def main():
    print(1)

if __name__ == "__main__":
    main()