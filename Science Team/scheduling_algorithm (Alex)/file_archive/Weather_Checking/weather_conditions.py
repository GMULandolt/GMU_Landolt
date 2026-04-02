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

observatories = [

     {
        "name": "George Mason Univ. Observatory, Virginia",
        "latitude": 38.8282,
        "longitude": -77.3053,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Bowling Green State Univ. Observatory, Ohio",
        "latitude": 41.378333,
        "longitude": -83.659167,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Collins Observatory, Colby College, Maine",
        "latitude": 44.56667,
        "longitude": -69.656378,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Smith College Observatory, Northampton, MA",
        "latitude": 42.317036,
        "longitude": -72.639514,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Moore Observatory, Univ. of Louisville, Kentucky",
        "latitude": 38.344792,
        "longitude": -85.528475,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Harvard Clay Telescope, Cambridge, MA",
        "latitude": 42.3766,
        "longitude": -71.1169,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Oak Ridge Observatory, Harvard, MA",
        "latitude": 42.505261,
        "longitude": -71.558144,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Leander McCormick Observatory, Univ. of Virginia",
        "latitude": 38.033333,
        "longitude": -78.523333,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Black Moshannon Observatory, State College PA",
        "latitude": 40.921667,
        "longitude": -78.005000,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Michael L. Britton Observatory, Dickinson College, PA",
        "latitude": 40.20398,
        "longitude": -77.19786,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Fan Mountain Observatory, VA",
        "latitude": 37.878333,
        "longitude": -78.693333,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Whitin Observatory, Wellesley College, MA",
        "latitude": 42.295000,
        "longitude": -71.305833,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Olin Observatory, Connecticut College, CT",
        "latitude": 41.378889,
        "longitude": -72.105278,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Sperry Observatory, Union County College, NJ",
        "latitude": 40.66632,
        "longitude": -74.32327,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Peter van de Kamp Observatory, Swarthmore College, PA",
        "latitude": 39.907100,
        "longitude": -75.355550,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Union College Observatory, NY",
        "latitude": 42.8176,
        "longitude": -73.9283,
        "timezone": "EST5EDT",
    },
    {
        "name": "Van Vleck Observatory, Wesleyan University, CT",
        "latitude": 41.555000,
        "longitude": -72.659167,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Vassar College Observatory, Poughkeepsie, NY",
        "latitude": 41.683011,
        "longitude": -73.890604,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Williams College Observatory, MA",
        "latitude": 42.7115,
        "longitude": -73.2052,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Mittelman Observatory, Middlebury College, VT",
        "latitude": 44.0134,
        "longitude": -73.1813,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "George R. Wallace, Jr. Astrophysical Observatory, MA",
        "latitude": 42.295,
        "longitude": -71.485,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Foggy Bottom Observatory, Colgate Univ., NY",
        "latitude": 42.81651,
        "longitude": -75.532568,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Breyo Observatory, Siena College, NY",
        "latitude": 42.719546,
        "longitude": -73.751433,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "C.E.K. Mees Observatory, Univ. Rochester, NY",
        "latitude": 42.7002778,
        "longitude": -77.4087667,
        "timezone_integer": 5,
        "timezone": "EST5EDT",
    },
    {
        "name": "Observatoire du Mont-Mégantic, Québec",
        "latitude": 45.455683,
        "longitude": -71.1521,
        "timezone_integer": 5,
        "timezone": "America/Toronto",
    },
]

def getObservatory(name: str): 
    for obj in observatories:
        if (obj["name"] == name):
            return obj
    return None

def location_weather_check(name: str):

    observatory = getObservatory(name)
    observatory_lat = observatory["latitude"]
    observatory_long = observatory["longitude"]
    current_directory = os.path.abspath(os.path.dirname(__file__))
    weather_directory = os.path.join(
        current_directory, r"..", r"..", r"resources", r"weather_status"
    )
    weather_url = "https://api.weatherapi.com/v1/current.json?key=fe686757107d46519c010740232712&q=" + str(observatory_lat )+ "," + str(observatory_long)

    user_agent = "(George Mason University Observatory, gmuobservatory@gmail.com)"
    cloud_satellite = "goes-16"
    weather_api_key = '"SUN_V3_API_KEY(.+?)":"(.+?)",'

    connection_alert = threading.Event()

    s = requests.Session()
    humidity = wind = rain = temperature = None

    target_path = os.path.abspath(os.path.join(weather_directory, r"weather.txt"))

    if None in (humidity, wind, rain):
        success = False
        encountered_error = False

        for i in range(1, 9):
            if i != 1:
                time.sleep(6 * i)

            try:
                weather = s.get(weather_url, headers={"User-Agent": user_agent})

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
                temperature = float(res["current"]["temp_f"])
                humidity = float(res["current"]["humidity"])
                wind = float(res["current"]["wind_mph"])

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

    logging.debug(f"Humidity: {humidity}, Wind: {wind}, Temperature: {temperature}")
    return humidity, wind, rain, temperature


def main():
    print(location_weather_check("George Mason Univ. Observatory, Virginia"))


if __name__ == "__main__":
    main()
