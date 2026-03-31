import time
import numpy as np
from sgp4.api import Satrec, WGS72
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.positionlib import Barycentric
import pandas as pd
from pytz import timezone
from settings import parameters
import math
from skyfield import almanac
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import geodatasets
import matplotlib.pyplot as plt

def eclipse(satpos, earthpos, sunpos):

    satpos = Barycentric(satpos + earthpos).position.au
    satsun = sunpos - satpos
    satearth = earthpos - satpos
    satsundist = np.linalg.norm(satsun)
    satearthdist = np.linalg.norm(satearth)

    asun = math.atan(0.00465047/satsundist)
    aearth = math.atan(4.26354e-5/satearthdist)
    theta = math.acos(np.vdot(satsun, satearth)/(satearthdist * satsundist))    
    sunsolid = math.pi * asun ** 2
    earthsolid = math.pi * aearth ** 2

    if (asun > theta + aearth):
        return str(round(earthsolid/sunsolid * 100)) + "%"
    elif (theta > asun + aearth or math.isclose(asun + aearth, theta)):
        return "0%"
    elif (aearth > asun + theta or math.isclose(asun + theta, aearth)):
        return "100%"
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
    return "fail"












def func (inc, eccen, node):
    # Load timescale
    ts = load.timescale()
    tscale = int((parameters.end - parameters.start) * 24 * 60 * 60 * 1000 + 1)

    eph = load('de421.bsp')
    sun, earth = eph['sun'], eph['earth']


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
        eccen,             # ecco: eccentricity
        parameters.argpo,               # argpo: argument of perigee (radians)
        inc,               # inclo: inclination (radians)
        parameters.mo,               # mo: mean anomaly (radians)
        parameters.no_kozai,  # no_kozai: mean motion (radians/minute) GEO
    #    0.04908738521, #LEO
    #    0.00872664625, #meo
        node                # nodeo: right ascension of ascending node (radians)
    )

    if (parameters.tle1 != "NA" or parameters.tle2 != "NA"):
        sat = Satrec.twoline2rv(parameters.tle1, parameters.tle2)
    # Convert Satrec object to EarthSatellite object
    sat = EarthSatellite.from_satrec(sat, ts)

    # Define the location of GMU Observatory
    obs = wgs84.latlon(parameters.lat, parameters.lon, parameters.elev)
    # Vector between sat and obs
    difference = sat - obs


    rubin = wgs84.latlon(-30, -70.7493, 2647)
    mason = wgs84.latlon(38.8282, -77.3053, 140)
    palomar = wgs84.latlon(33.1, -116.8649, 1872)
    sniffs = wgs84.latlon(19.82, -155.4694, 4245)

    rubindiff = sat - rubin
    masondiff = sat - mason
    palomardiff = sat - palomar
    sniffsdiff = sat - sniffs
    


    chunk_size = parameters.tdelta * parameters.chunks
    num_chunks = int(tscale/chunk_size)
    satcords = np.zeros((num_chunks, 3, int(chunk_size / parameters.tdelta)), object)
    satlatlon = np.zeros((num_chunks, 2, int(chunk_size / parameters.tdelta)), object)
    obscords = np.zeros((num_chunks, 3, int(chunk_size / parameters.tdelta)), object)
    eclipsepec = np.zeros((int(num_chunks*parameters.chunks), 1), object)
    timelist = np.zeros((num_chunks, int(chunk_size / parameters.tdelta)), object)
    tempdf = np.zeros((num_chunks, 5, int(chunk_size / parameters.tdelta)), object)
    obstempdf = np.zeros((num_chunks, 12, int(chunk_size / parameters.tdelta)), object)
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
        
        satpos = satcord.position.au
        sunpos = sun.at(t).position.au
        earthpos = earth.at(t).position.au
        for j in range(len(satpos[1])):
            eclipsepec[j + i*len(satpos[1])] = eclipse(satpos[:,j], earthpos[:,j], sunpos[:,j])
        
        topocentric = difference.at(t)
        ra, dec, distance = topocentric.radec()
        alt, az, trash = topocentric.altaz()
        tempdf[i] = np.array([ra._degrees, dec._degrees, az.degrees, alt.degrees, distance.km])
                

        rubinalt, rubinaz, trash = rubindiff.at(t).altaz()
        masonalt, masonaz, trash = masondiff.at(t).altaz()
        palomaralt, palomaraz, trash = palomardiff.at(t).altaz()
        sniffsalt, sniffsaz, trash = sniffsdiff.at(t).altaz()

        obstempdf[i] = np.array([rubinalt.degrees, rubinaz.degrees, almanac.dark_twilight_day(eph, rubin)(t), masonalt.degrees, masonaz.degrees, almanac.dark_twilight_day(eph, mason)(t), palomaralt.degrees, palomaraz.degrees, almanac.dark_twilight_day(eph, palomar)(t), sniffsalt.degrees, sniffsaz.degrees, almanac.dark_twilight_day(eph, sniffs)(t)])


    temptime = temptime.flatten() 
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
    df = pd.DataFrame({'X': obscords[:, 0, :].flatten(), 'Y': obscords[:, 1, :].flatten(), 'Z': obscords[:, 2, :].flatten()})
    df.to_csv('obscordsxyz.csv', index=False)


    obsdf = pd.DataFrame({'Time (EST)': temptime, 
                    'Rubin Alt (Deg)': obstempdf[:, 0, :].flatten(), 'Rubin Az (Deg)': obstempdf[:, 1, :].flatten(), 'Rubin TIME': obstempdf[:, 2, :].flatten(),
                    'Mason Alt (Deg)': obstempdf[:, 3, :].flatten(), 'Mason Az (Deg)': obstempdf[:, 4, :].flatten(), 'Mason TIME': obstempdf[:, 5, :].flatten(),
                    'Palomar Alt (Deg)': obstempdf[:, 6, :].flatten(), 'Palomar Az (Deg)': obstempdf[:, 7, :].flatten(),  'Palomar TIME': obstempdf[:, 8, :].flatten(),
                    'SNIFFS Alt (Deg)': obstempdf[:, 9, :].flatten(), 'SNIFFS Az (Deg)': obstempdf[:, 10, :].flatten(),  'SNIFFS TIME': obstempdf[:, 11, :].flatten()})
    obsdf["Eclipse %"] = eclipsepec.flatten().tolist()

    return obsdf


func((2*np.pi) - 20*(np.pi/180), 0, 120*(np.pi/180))


df = pd.read_csv("satlatlon.csv", delimiter=',', skiprows=0, low_memory=False)

geometry = [Point(xy) for xy in zip(df['Lon'], df['Lat'])]
gdf = GeoDataFrame(df, geometry=geometry)   

#this is a simple map that goes with geopandas
# deprecated: world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = gpd.read_file(geodatasets.data.naturalearth.land['url'])
gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15)
plt.savefig("world.png")