import pandas as pd
import math
import os
import TLEconstructor
import numpy as np
import matplotlib.pyplot as plt
import time
from playsound import playsound


res = 30
airmass = 2
inclist = np.linspace(-40*np.pi/180, 40*np.pi/180, res)
for i in range(len(inclist)):
    if inclist[i] < 0:
        inclist[i] = 2*np.pi + inclist[i]
nodelist = np.linspace(0, 2*np.pi, res)
orbitlist = np.empty((0, 12))

start_time = time.time()
print("Simulating Satellite Orbits...")
i = 0
for node in nodelist:
    for inc in inclist:
        df = TLEconstructor.func(inc, 0, node)
        rubin = df[(df['Rubin Alt (Deg)'] > 90 - np.rad2deg(np.arccos(1/airmass))) & (df['Rubin TIME'] < 2)]
        mason = df[(df['Mason Alt (Deg)'] > 90 - np.rad2deg(np.arccos(1/airmass))) & (df['Mason TIME'] < 2)]
        palomar = df[(df['Palomar Alt (Deg)'] > 90 - np.rad2deg(np.arccos(1/airmass))) & (df['Palomar TIME'] < 2)]
        sniffs = df[(df['SNIFFS Alt (Deg)'] > 90 - np.rad2deg(np.arccos(1/airmass))) & (df['SNIFFS TIME'] < 2)]

        erubin = rubin[rubin["Eclipse %"] != "0%"]
        erubin.loc[:, 'Time (EST)'] = erubin['Time (EST)'].astype(str).str.slice(0, 10)
        erubin = erubin.drop_duplicates(subset=['Time (EST)'], keep='first')

        emason = mason[mason["Eclipse %"] != "0%"]
        emason.loc[:, 'Time (EST)'] = emason['Time (EST)'].astype(str).str.slice(0, 10)
        emason = emason.drop_duplicates(subset=['Time (EST)'], keep='first')

        epalomar = palomar[palomar["Eclipse %"] != "0%"]
        epalomar.loc[:, 'Time (EST)'] = epalomar['Time (EST)'].astype(str).str.slice(0, 10)
        epalomar = epalomar.drop_duplicates(subset=['Time (EST)'], keep='first')

        esniffs = sniffs[sniffs["Eclipse %"] != "0%"]
        esniffs.loc[:, 'Time (EST)'] = esniffs['Time (EST)'].astype(str).str.slice(0, 10)
        esniffs = esniffs.drop_duplicates(subset=['Time (EST)'], keep='first')

        obspercent = np.zeros(4)
        obspercent[0] = len(rubin['Rubin Alt (Deg)'].to_numpy())/np.size(np.where(df['Rubin TIME'].to_numpy() < 2))*100
        obspercent[1] = len(mason['Mason Alt (Deg)'].to_numpy())/np.size(np.where(df['Mason TIME'].to_numpy() < 2))*100
        obspercent[2] = len(palomar['Palomar Alt (Deg)'].to_numpy())/np.size(np.where(df['Palomar TIME'].to_numpy() < 2))*100
        obspercent[3] = len(sniffs['SNIFFS Alt (Deg)'].to_numpy())/np.size(np.where(df['SNIFFS TIME'].to_numpy() < 2))*100
        orbitlist = np.append(orbitlist, [[node, inc, 0, np.mean(obspercent), obspercent[0], len(erubin), obspercent[1], len(emason), obspercent[2], len(epalomar), obspercent[3], len(esniffs)]], axis=0)
        i += 1
        print('\r' + str(int(i/(len(inclist)*len(nodelist))*100)) + "%", end='', flush=True)

end_time = time.time()
print()
print('Execution time = %.6f seconds' % (end_time-start_time))
np.save('air2res30night.npy', orbitlist)
#playsound('rawtime.mp3')