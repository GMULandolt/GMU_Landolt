from datetime import datetime, timedelta
import time

def roundToAnHour(dt: datetime):
    return (dt.replace(second = 0, microsecond = 0, minute = 0, hour = dt.hour) + timedelta(hours = dt.minute//30)) 

class ObservatoryForecast: 
    def __init__(self):
        self.currentDT = None
        self.jsonWeather = None

    def updateForecast(self, dt: datetime, jsonWeather: any):
        self.currentDT = dt
        self.jsonWeather = jsonWeather

    def extractForecast(self, currentDT: datetime, forecastDT: datetime):
        if self.currentDT != currentDT: 
            print("Weather forecast call error")
            return None
    
        forecastDT = roundToAnHour(forecastDT)
        forecastDate = forecastDT.date()
        forecastDateString = time.strptime(forecastDate, "%Y-%m-%d")
        forecastDateTimeString = time.strptime(forecastDate, "%Y-%m-%d %H:%M")


        for jsonDay in self.jsonWeather["forecast"]["forecastday"]:
            if jsonDay["date"] == forecastDateString:
                for jsonHour in jsonDay["hour"]: 
                    if jsonHour["time"] == forecastDateTimeString:
                        temperature = float(jsonHour["temp_f"])
                        humidity = float(jsonHour["humidity"])
                        wind = float(jsonHour["wind_mph"])
                        cloud = float(jsonHour["cloud"])
                        rain = float(jsonHour["chance_of_rain"])

                        if (rain > 40 or humidity > 85 or temperature > 100 or wind > 20 or cloud > 60): 
                            return float('-inf')
        
                        weatherValue = cloud * 0.5 + humidity * 0.2 + temperature * 0.2 + wind * 0.1
                        return weatherValue
        return None
