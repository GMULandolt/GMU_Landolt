import urllib.request
import urllib.error
import urllib3.exceptions
import requests
import os
import threading
import logging
import json
import time
from typing import Dict
from observatory_characteristics import ObservatoryCharacteristics
from observatory_repository import ObservatoryRepository
from observatory import Observatory
from datetime import datetime

class WeatherConditions: 

    def __init__(self):
        self.forecastDate = None
        
    def updateRepositoryForecast(self, dt: datetime, orep: ObservatoryRepository):
        for obsItem in orep.observatoryList: 
            obs: Observatory = obsItem
            self.updateObservatoryForecast(dt, obs)

    def updateObservatoryForecast(self, dt: datetime, obs: Observatory):
        observatory_lat = obs.characteristics.latitude
        observatory_long = obs.characteristics.longitude
        #current_directory = os.path.abspath(os.path.dirname(__file__))
        #weather_directory = os.path.join(
        #    current_directory, r"..", r"..", r"resources", r"weather_status"
        #)

        #current_weather_url = "https://api.weatherapi.com/v1/current.json?key=fe686757107d46519c010740232712&q=" + str(observatory_lat )+ "," + str(observatory_long)

        forecast_weather_url = "https://api.weatherapi.com/v1/forecast.json?key=fe686757107d46519c010740232712&q=" + str(observatory_lat )+ "," + str(observatory_long) + "&days=7"

        user_agent = "(George Mason University Observatory, gmuobservatory@gmail.com)"
        #cloud_satellite = "goes-16"
        #weather_api_key = '"SUN_V3_API_KEY(.+?)":"(.+?)",'

        connection_alert = threading.Event()

        s = requests.Session()
        humidity = wind = rain = temperature = cloud = None

        #target_path = os.path.abspath(os.path.join(weather_directory, r"weather.txt"))

        if None in (humidity, wind, rain, cloud):
            success = False
            encountered_error = False

            for i in range(1, 9):
                if i != 1:
                    time.sleep(6 * i)

                try:
                    weather = s.get(forecast_weather_url, headers={"User-Agent": user_agent})

                except (
                    urllib3.exceptions.MaxRetryError,
                    urllib3.exceptions.HTTPError,
                    urllib3.exceptions.TimeoutError,
                    urllib3.exceptions.InvalidHeader,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.HTTPError,
                    json.decoder.JSONDecodeError,
                    KeyError,
                ) as e:
                    logging.warning(f"Could not connect to weatherapi.com API: try {i}")
                    logging.exception(e)
                    encountered_error = True
                    continue

                try:
                    res = json.loads(weather.text)
                    obs.forecast.updateForecast(res)

                    #temperature = float(res["current"]["temp_f"])
                    #humidity = float(res["current"]["humidity"])
                    #wind = float(res["current"]["wind_mph"])
                    #cloud = float(res["current"]["cloud"])

                    success = True
                    break

                except (json.decoder.JSONDecodeError, KeyError) as e:
                    logging.warning(f"Failed to read weatherapi.com API: try {i}")
                    # logging.exception(e)
                    logging.info(weather.text)
                    encountered_error = True

            if not success:
                logging.warning(
                    f"Could not read weatherapi.com  after {i} tries. Setting connection alert."
                )
                connection_alert.set()
                return None, None, None, None

            if success and encountered_error:
                logging.info(f"Successfully read weatherapi.com after {i} tries.")

        #logging.debug(f"Humidity: {humidity}, Wind: {wind}, Temperature: {temperature}")

        #if (rain != None or humidity > 85 or temperature > 100 or wind > 20 or cloud > 60): 
            #return float('-inf')
        
        #weatherValue = cloud * 0.5 + humidity * 0.2 + temperature * 0.2 + wind * 0.1
        
        #return weatherValue

    def main():
        print(0)
        #for obs in ObservatoryCharacteristics: 
            #weatherCheck(obs)
            #print(weatherCheck(obs))


    if __name__ == "__main__":
        main()