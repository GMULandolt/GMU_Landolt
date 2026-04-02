import pandas as pd
import math
import os
import TLEconstructor
import numpy as np
import matplotlib.pyplot as plt

kmlis = np.arange(35, 36.6, 0.1).tolist()
trackrate = []
for km in kmlis:
    print()
    kaizo = 60/math.sqrt((6378+km*1000)**3/(6.6743*10**-20*5.972*10**24))
    TLEconstructor.func(kaizo)
    data = pd.read_csv("satlatlon.csv")
    datacoord = pd.read_csv("satcoord.csv")
    print("km: " + str(km) + "000")
    print(datacoord["Distance (Km)"].min())
    lon = np.radians(data["Lon"].to_numpy())
    change = []
    for i in range(1, len(lon)):
        change.append(np.rad2deg(np.arccos(np.sin(lon[i-1])*np.sin(lon[i]) + \
                              np.cos(lon[i-1])*np.cos(lon[i])))*60*24)
    trackrate.append(np.average(change))

print(kmlis)
print(trackrate)

plt.plot(kmlis, trackrate)

# Add labels and title
#plt.xlim(10000, 35000)
#plt.ylim(0, 100)
plt.xlabel('Altitude (km)')
plt.ylabel('Average Change in Longitude per Day (Deg)')
#plt.title('Rate of Sattelite Walk Per Day')
plt.savefig('lon.png') 