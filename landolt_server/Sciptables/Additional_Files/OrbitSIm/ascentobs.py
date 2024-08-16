import time
import numpy as np
from sgp4.api import Satrec
from skyfield.api import load, EarthSatellite, wgs84
import pandas as pd
from pytz import timezone

start_time = time.time()
# Load timescale
ts = load.timescale()
eastern = timezone('US/Eastern')



s = '1 51287U 21118E   24149.61788671 -.00000002  00000-0  00000-0 0  9993'
t = '2 51287   2.2617  87.2655 0005139  44.8639 314.7193  1.01256551  8730'
ascent = Satrec.twoline2rv(s, t)

seconds = 7200
tscale = seconds * 1000

# Convert Satrec object to EarthSatellite object
sat = EarthSatellite.from_satrec(ascent, ts)

# Define the location of GMU Observatory
obs = wgs84.latlon(38.8282, -77.3053, 140)
# Vector between sat and obs
difference = sat - obs



chunk_size = 1000
num_chunks = int(tscale/chunk_size)
satcords = np.zeros((num_chunks, 3, chunk_size), object)
obscords = np.zeros((num_chunks, 3, chunk_size), object)
timelist = np.zeros((num_chunks, chunk_size), object)
tempdf = np.zeros((num_chunks, 5, chunk_size), object)
temptime = np.zeros((num_chunks, chunk_size), object)

for i in range(num_chunks):
   t = ts.utc(2024, 5, 31, 1, 30, np.arange(i*chunk_size, (i+1) * chunk_size) * 0.001)
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
df.to_csv('ascentcoord.csv', index=False)

end_time = time.time()
print("Simulation run complete and data stored...")
print('Execution time = %.6f seconds' % (end_time-start_time))