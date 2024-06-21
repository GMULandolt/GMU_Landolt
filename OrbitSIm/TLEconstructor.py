import time
import numpy as np
from sgp4.api import Satrec, WGS72
from skyfield.api import load, EarthSatellite, wgs84
import pandas as pd
from pytz import timezone

start_time = time.time()
# Load timescale
ts = load.timescale()
eastern = timezone('US/Eastern')
seconds = 43200
tscale = seconds


# Initialize satellite using SGP4
sat = Satrec()
sat.sgp4init(
    WGS72,           # gravity model
    'i',             # 'a' = old AFSPC mode, 'i' = improved mode
    1,               # satnum: Satellite number
    0,               # epoch: days since 1949 December 31 00:00 UT
    0,           # bstar: drag coefficient (/earth radii)
    0,           # ndot: ballistic coefficient (radians/minute^2)
    0.0,             # nddot: second derivative of mean motion (radians/minute^3)
    0.1,             # ecco: eccentricity
    0,               # argpo: argument of perigee (radians)
    0,               # inclo: inclination (radians)
    0,               # mo: mean anomaly (radians)
    0.00437526951 ,  # no_kozai: mean motion (radians/minute) GEO
#    0.04908738521, #LEO
#    0.00872664625, #meo
    0                # nodeo: right ascension of ascending node (radians)
)

# Convert Satrec object to EarthSatellite object
sat = EarthSatellite.from_satrec(sat, ts)

# Define the location of GMU Observatory
obs = wgs84.latlon(38.8282, -77.3053, 140)
# Vector between sat and obs
difference = sat - obs



chunk_size = 100
num_chunks = int(tscale/chunk_size)
satcords = np.zeros((num_chunks, 3, chunk_size), object)
obscords = np.zeros((num_chunks, 3, chunk_size), object)
timelist = np.zeros((num_chunks, chunk_size), object)
tempdf = np.zeros((num_chunks, 5, chunk_size), object)
temptime = np.zeros((num_chunks, chunk_size), object)

for i in range(num_chunks):
   t = ts.utc(2024, 1, 1, 5, 0, np.arange(i*chunk_size, (i+1) * chunk_size))
   timelist[i] = t
   temptime[i] = t.astimezone(eastern)
   
   
   satcord = sat.at(t)
   obscoord = obs.at(t)
   satcords[i] = satcord.position.km
   obscords[i] = obscoord.position.km
   
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

end_time = time.time()
print("Simulation run complete and data stored...")
print('Execution time = %.6f seconds' % (end_time-start_time))