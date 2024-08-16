import time
import numpy as np
from sgp4.api import Satrec, WGS72
from skyfield.api import load, EarthSatellite, wgs84
import pandas as pd
from pytz import timezone
from settings import parameters

start_time = time.time()
print("Simulating Sattelite Orbit...")
# Load timescale
ts = load.timescale()
tscale = int((parameters.end - parameters.start) * 24 * 60 * 60 * 1000 + 1)


# Initialize satellite using SGP4
sat = Satrec()
sat.sgp4init(
    WGS72,           # gravity model
    'i',             # 'a' = old AFSPC mode, 'i' = improved mode
    1,               # satnum: Satellite number
    parameters.epoch,               # epoch: days since 1949 December 31 00:00 UT
    parameters.bstar,           # bstar: drag coefficient (/earth radii)
    parameters.ndot,           # ndot: ballistic coefficient (radians/minute^2)
    parameters.nddot,             # nddot: second derivative of mean motion (radians/minute^3)
    parameters.ecco,             # ecco: eccentricity
    parameters.argpo,               # argpo: argument of perigee (radians)
    parameters.inclo,               # inclo: inclination (radians)
    parameters.mo,               # mo: mean anomaly (radians)
    parameters.no_kozai,  # no_kozai: mean motion (radians/minute) GEO
#    0.04908738521, #LEO
#    0.00872664625, #meo
    parameters.nodeo                # nodeo: right ascension of ascending node (radians)
)
if (parameters.tle1 != "NA" or parameters.tle2 != "NA"):
    sat = Satrec.twoline2rv(parameters.tle1, parameters.tle2)
# Convert Satrec object to EarthSatellite object
sat = EarthSatellite.from_satrec(sat, ts)

# Define the location of GMU Observatory
obs = wgs84.latlon(parameters.lat, parameters.lon, parameters.elev)
# Vector between sat and obs
difference = sat - obs



chunk_size = parameters.tdelta * parameters.chunks
num_chunks = int(tscale/chunk_size)
satcords = np.zeros((num_chunks, 3, int(chunk_size / parameters.tdelta)), object)
satlatlon = np.zeros((num_chunks, 2, int(chunk_size / parameters.tdelta)), object)
obscords = np.zeros((num_chunks, 3, int(chunk_size / parameters.tdelta)), object)
timelist = np.zeros((num_chunks, int(chunk_size / parameters.tdelta)), object)
tempdf = np.zeros((num_chunks, 5, int(chunk_size / parameters.tdelta)), object)
temptime = np.zeros((num_chunks, int(chunk_size / parameters.tdelta)), object)

for i in range(num_chunks):
   t = ts.utc(parameters.start.utc.year, \
              parameters.start.utc.month, \
              parameters.start.utc.day, \
              parameters.start.utc.hour, \
              parameters.start.utc.minute, \
              parameters.start.utc.second + np.arange(i*chunk_size, (i+1) * chunk_size, parameters.tdelta) * 0.001)
   timelist[i] = t
   temptime[i] = t.astimezone(timezone(parameters.timezone))
   
   
   satcord = sat.at(t)
   obscoord = obs.at(t)
   satcords[i] = satcord.position.km
   obscords[i] = obscoord.position.km
   lat, lon = wgs84.latlon_of(satcord)
   satlatlon[i] = [lat.degrees, lon.degrees]
   
   topocentric = difference.at(t)
   ra, dec, distance = topocentric.radec()
   alt, az, trash = topocentric.altaz()
   tempdf[i] = np.array([ra._degrees, dec.degrees, az.degrees, alt.degrees, distance.km])
          
   print('\r' + str(int(i/num_chunks * 10000)/100) + "%", end='', flush=True)

temptime = temptime.flatten()
print('\r' + "100.00%\n", end='', flush=True)  

df = pd.DataFrame({'Time (EST)': temptime, 
                   'RA (Deg)': tempdf[:, 0, :].flatten(), 'Dec (Deg)': tempdf[:, 1, :].flatten(),
                   'Az (Deg)': tempdf[:, 2, :].flatten(), 'Alt (Deg)': tempdf[:, 3, :].flatten(),
                   'Distance (Km)': tempdf[:, 4, :].flatten()})
df.to_csv('satcoord.csv', index=False)

df = pd.DataFrame({'Lat': satlatlon[:, 0, :].flatten(), 'Lon': satlatlon[:, 1, :].flatten()})
df.to_csv('satlatlon.csv', index=False)
df = pd.DataFrame({'X': satcords[:, 0, :].flatten(), 'Y': satcords[:, 1, :].flatten(), 'Z': satcords[:, 2, :].flatten()})
df.to_csv('satcoordxyz.csv', index=False)

end_time = time.time()
print("Simulation run complete and data stored...")
print('Execution time = %.6f seconds' % (end_time-start_time))