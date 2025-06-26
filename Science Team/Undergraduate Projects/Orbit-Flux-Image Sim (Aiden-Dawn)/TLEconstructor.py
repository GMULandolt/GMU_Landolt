#NOTE: This entire script along with the flux, image sim, and settings code need to be put into jupiter notebooks and made runnable. Documentation must be completed

import time
import numpy as np
from sgp4.api import Satrec, WGS72
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.positionlib import Barycentric
import pandas as pd
from pytz import timezone
from settings import parameters
import math

#This functions acts as an add on to calculate the solid angle overlap of the earth and the sun. This way we can calcualte how much of the sun is eclipsed by the earth and predict umbra/penubra
#However, since the script calculates single array elements instead of a whole array at once, it is very inefficient
#Allowing numpy to run these calculations would be far more efficient, and may drastically decrase runtime especially for the orbit propigator packeged elsewhere (which is a branch of this code)
def eclipse(satpos, earthpos, sunpos):

    #Nere we fetch planet positions at each time step and draw vectlrs to each
    satpos = Barycentric(satpos + earthpos).position.au
    satsun = sunpos - satpos
    satearth = earthpos - satpos
    satsundist = np.linalg.norm(satsun)
    satearthdist = np.linalg.norm(satearth)

    #calculating solid angle. I have no idea where those values for tan came from, will need to return to comment
    asun = math.atan(0.00465047/satsundist)
    aearth = math.atan(4.26354e-5/satearthdist)
    theta = math.acos(np.vdot(satsun, satearth)/(satearthdist * satsundist))    
    sunsolid = math.pi * asun ** 2
    earthsolid = math.pi * aearth ** 2

    #edge cases for whether or not to actually calculate the solid angle to save on runtime
    if (asun > theta + aearth):
        return str(round(earthsolid/sunsolid * 100)) + "%"
    elif (theta > asun + aearth or math.isclose(asun + aearth, theta)):
        return "0%"
    elif (aearth > asun + theta or math.isclose(asun + theta, aearth)):
        return "100%"
    #This is my own derivation of the problem, however a simple explanation of the problem can be found here https://www.youtube.com/watch?v=mMYCyeGVKUo
    #Will need to return when documenting to write up a mathmatical derivation
    #Fairly costly, however we luckly only run this code for a few minutes as the satelite is in penumbra
    else:
        a = (math.cos(aearth) - (math.cos(asun)*math.cos(theta)))/math.sin(theta)
        b = (math.sin(asun)**2 - a**2)**0.5
        
        p1 = np.array([0, 0, 1])
        p2 = np.array([math.sin(theta), 0, math.cos(theta)])
        p3 = np.array([a, -b, math.cos(asun)])
        p4 = np.array([a, b, math.cos(asun)])
                
        nb = np.cross(p1, p4 - p1)/np.linalg.norm(np.cross(p1, p4 - p1))
        nc = np.cross(p1, p3 - p1)/np.linalg.norm(np.cross(p1, p3 - p1))
        phi1 = math.acos(np.vdot(nb, nc))
        nb = np.cross(p2, p4 - p2)/np.linalg.norm(np.cross(p2, p4 - p2))
        nc = np.cross(p2, p3 - p2)/np.linalg.norm(np.cross(p2, p3 - p2))
        phi2 = math.acos(np.vdot(nb, nc))
        nb = np.cross(p4, p1 - p4)/np.linalg.norm(np.cross(p4, p1 - p4))
        nc = np.cross(p4, p3 - p4)/np.linalg.norm(np.cross(p4, p3 - p4))
        psi1 = math.acos(np.vdot(nb, nc))
        nb = np.cross(p4, p2 - p4)/np.linalg.norm(np.cross(p4, p2 - p4))
        nc = np.cross(p4, p3 - p4)/np.linalg.norm(np.cross(p4, p3 - p4))
        psi2 = math.acos(np.vdot(nb, nc))
                
        if (a < 0):
            digon = (2*math.pi-phi1)*(1-math.cos(asun)) + phi1 +2*psi1 - math.pi + phi2*(1-math.cos(aearth)) - (phi2+2*psi2-math.pi)
        else:
            digon = 2*math.pi - 2*(psi1 + psi2) - phi1 * math.cos(asun) - phi2 * math.cos(aearth)
        
        return str(round(digon/sunsolid * 100)) + "%"
    #dead code, should never fail unless inputs are fucked up. Only present for if
    return "fail"















#begining of script, we calculate runtime
start_time = time.time()
print("Simulating Satellite Orbit...")

ts = load.timescale()
#Calculating number of timesteps to calculate in the miliseconds
tscale = int((parameters.end - parameters.start) * 24 * 60 * 60 * 1000 + 1)

#loading the ephemeri of the sun and earth is intesive so its out here instead of the eclipse script
eph = load('de421.bsp')
sun, earth = eph['sun'], eph['earth']


# Initialize satellite using SGP4, for more information see https://pypi.org/project/sgp4/#providing-your-own-elements
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
#If TLE is present it will be used instead
if (parameters.tle1 != "NA" or parameters.tle2 != "NA"):
    sat = Satrec.twoline2rv(parameters.tle1, parameters.tle2)
# Convert Satrec object to EarthSatellite object
sat = EarthSatellite.from_satrec(sat, ts)

# Define the location of Observatory
obs = wgs84.latlon(parameters.lat, parameters.lon, parameters.elev)
# Vector between sat and obs
difference = sat - obs










#This chunking serves as a way for large amounts of time steps to be generated in single runs of the for loop.
#For some reason sgp4 cannot actually compute more than something like 10000 time intervals at once which is the point of the chuncking
#This chunking system has a fairly big bug where if the chuncks do not perfectly line up with the time intervals requested, it will fail
#to output the remaining time since it won't load the final chunk
#EXAMPLE: 
#input - chunk size = 3, start = 1, end = 8
#output - 1, 2, 3, 4, 5, 6
#         |______| |_____|  ...
#          chunk 1  chunk 2  no chunk 3
#As you can see times 7 and 8 were never output since chunk 3 couldn't fit.
chunk_size = parameters.tdelta * parameters.chunks
num_chunks = int(tscale/chunk_size)
satcords = np.zeros((num_chunks, 3, int(chunk_size / parameters.tdelta)), object)
satlatlon = np.zeros((num_chunks, 2, int(chunk_size / parameters.tdelta)), object)
obscords = np.zeros((num_chunks, 3, int(chunk_size / parameters.tdelta)), object)
eclipsepec = np.zeros((int(num_chunks*parameters.chunks), 1), object)
timelist = np.zeros((num_chunks, int(chunk_size / parameters.tdelta)), object)
tempdf = np.zeros((num_chunks, 5, int(chunk_size / parameters.tdelta)), object)
temptime = np.zeros((num_chunks, int(chunk_size / parameters.tdelta)), object)

for i in range(num_chunks):
    #Here we set an array of skyfield time objects to calculate positions at
    #
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
   
   
   satpos = satcord.position.au
   sunpos = sun.at(t).position.au
   earthpos = earth.at(t).position.au
   for j in range(len(satpos[1])):
       eclipsepec[j + i*len(satpos[1])] = eclipse(satpos[:,j], earthpos[:,j], sunpos[:,j])
   
   topocentric = difference.at(t)
   ra, dec, distance = topocentric.radec()
   alt, az, trash = topocentric.altaz()
   tempdf[i] = np.array([ra._degrees, dec.degrees, az.degrees, alt.degrees, distance.km])
          
   print('\r' + str(int(i/num_chunks * 10000)/100) + "%", end='', flush=True)

temptime = temptime.flatten()
print('\r' + "100.00%\n", end='', flush=True)  

df1 = pd.DataFrame({'Time (EST)': temptime, 
                   'RA (Deg)': tempdf[:, 0, :].flatten(), 'Dec (Deg)': tempdf[:, 1, :].flatten(),
                   'Az (Deg)': tempdf[:, 2, :].flatten(), 'Alt (Deg)': tempdf[:, 3, :].flatten(),
                   'Distance (Km)': tempdf[:, 4, :].flatten()})
df1["Eclipse %"] = eclipsepec.flatten().tolist()
df1.to_csv('satcoord.csv', index=False)

df = pd.DataFrame({'Lat': satlatlon[:, 0, :].flatten(), 'Lon': satlatlon[:, 1, :].flatten()})
df.to_csv('satlatlon.csv', index=False)
df = pd.DataFrame({'X': satcords[:, 0, :].flatten(), 'Y': satcords[:, 1, :].flatten(), 'Z': satcords[:, 2, :].flatten()})
df.to_csv('satcoordxyz.csv', index=False)

end_time = time.time()
print("Simulation run complete and data stored...")
print('Execution time = %.6f seconds' % (end_time-start_time))
