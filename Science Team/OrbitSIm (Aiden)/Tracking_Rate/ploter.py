import pandas as pd
import math
import os
import TLEconstructor
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
kmlis = [35.0, 35.1, 35.2, 35.300000000000004, 35.400000000000006, 35.50000000000001, 35.60000000000001, 35.70000000000001, 35.80000000000001, 35.90000000000001, 36.000000000000014, 36.100000000000016, 36.20000000000002, 36.30000000000002, 36.40000000000002, 36.50000000000002, 36.60000000000002]
trackrate = [10.33566068594565, 8.997557412492508, 7.665478404676182, 6.324062962927866, 5.017667294094105, 3.692717379684632, 2.4143933565990343, 1.0297725805425642, 0.02053456626767273, 1.4577961602719212, 2.734205104925894, 3.9915701773181187, 5.2349131634594, 6.48986571308882, 7.739913621880365, 8.969664250942426, 10.196352934880696]
plt.plot(kmlis, trackrate)

plt.minorticks_on()
# Add labels and title
#plt.xlim(10000, 35000)
#plt.ylim(0, 100)
plt.hlines(y = 7.85, color = 'darkred', linestyle = ':', xmin = 35, xmax = 35.18, label = "Crosses US in Week")
plt.vlines(x = 35.18, color = 'darkred', linestyle = ':', ymin = 0, ymax = 7.85)
plt.hlines(y = 1.8, color = 'darkblue', linestyle = ':', xmin = 35, xmax = 35.645, label = "Crosses US in Month")
plt.vlines(x = 35.645, color = 'darkblue', linestyle = ':', ymin = 0, ymax = 1.8)
plt.xlabel('Altitude (km)')
plt.ylabel('Average Change in Longitude per Day (Deg)')
plt.legend(loc = 'upper center') 
#plt.title('Rate of Sattelite Walk Per Day')
plt.savefig('lon.png') 