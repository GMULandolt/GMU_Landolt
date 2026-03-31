import pandas as pd
import math
import os
import TLEconstructor
import numpy as np
import matplotlib.pyplot as plt

kmlis = []
trackrate = []
for km in range(1, 37):
    print()
    kaizo = 60/math.sqrt((6378+km*1000)**3/(6.6743*10**-20*5.972*10**24))
    TLEconstructor.func(kaizo)
    data = pd.read_csv("satcoord.csv")
    print("km: " + str(km) + "000")
    print(data["Distance (Km)"].min())
    kmlis.append(km*1000)
    signchange = np.where(data["Alt (Deg)"] < 0, -1, 1)
    rsignchange = np.where(signchange[:-1] != signchange[1:])[0] + 1
    if (len(rsignchange) < 3):
        sindx = 0
        eindx = -1
    else:
        sindx = rsignchange[0]
        eindx = rsignchange[1]
        if (signchange[rsignchange[0]] == -1):
            sindx = rsignchange[1]
            eindx = rsignchange[2]
    data = data[sindx:eindx]
    alt = np.radians(data["Alt (Deg)"].to_numpy())
    change = []
    for i in range(1, len(alt)):
        change.append(np.rad2deg(np.arccos(np.sin(alt[i-1])*np.sin(alt[i]) + \
                              np.cos(alt[i-1])*np.cos(alt[i])))/60)
    trackrate.append(np.average(change) * 3600)
print(kmlis)
print(trackrate)

plt.plot(kmlis, trackrate)

# Add labels and title
#plt.xlim(10000, 35000)
#plt.ylim(0, 100)
plt.xlabel('Altitude (km)')
plt.ylabel('Average Angular Seperation per Second (\'\')')
plt.title('Rough Tracking Rate When Above Horizon')
plt.savefig('roughestimate.png') 