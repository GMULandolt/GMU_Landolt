import numpy as np
from sgp4.api import Satrec, WGS72
from skyfield.api import load, EarthSatellite, wgs84
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# Load timescale
ts = load.timescale()

# Initialize satellite using SGP4
sat = Satrec()
sat.sgp4init(
    WGS72,           # gravity model
    'i',             # 'a' = old AFSPC mode, 'i' = improved mode
    1,               # satnum: Satellite number
    25000,               # epoch: days since 1949 December 31 00:00 UT
    0,           # bstar: drag coefficient (/earth radii)
    0,           # ndot: ballistic coefficient (radians/minute^2)
    0.0,             # nddot: second derivative of mean motion (radians/minute^3)
    0,             # ecco: eccentricity
    0,               # argpo: argument of perigee (radians)
    0,               # inclo: inclination (radians)
    0,               # mo: mean anomaly (radians)
    0.03582972808 ,  # no_kozai: mean motion (radians/minute)
    0                # nodeo: right ascension of ascending node (radians)
)

# Convert Satrec object to EarthSatellite object
sat = EarthSatellite.from_satrec(sat, ts)

# Define the location of GMU Observatory
obs = wgs84.latlon(38.8282, -77.3053, 140)
# Vector between sat and obs
difference = sat - obs

# Define initial time
satcords = np.empty((0, 3), float)
obscords = np.empty((0, 3), float)
time = []
df = pd.DataFrame(columns=['Time', 'Ra', 'Dec', 'Azm', 'Alt', 'Dist'])


mindays = 24*60
# Loop through each hour of the year 2024 and store satellite positions
for i in range(0, mindays):
    t = ts.utc(2025, 1, 1, 0, i, 0)
    time.append(t)
    
    satcord = sat.at(t)
    obscoord = obs.at(t)
    satcords = np.vstack([satcords, satcord.position.km])
    obscords = np.vstack([obscords, obscoord.position.km])
    
    topocentric = difference.at(t)
    ra, dec, distance = topocentric.radec()
    alt, az, trash = topocentric.altaz()
    df.loc[len(df.index)] = [t.utc_iso(), ra.radians, dec.radians, az.radians, alt.radians, distance.km] 
    
    print('\r' + str(int(i/mindays * 100)) + "%", end='', flush=True)

df.to_csv('satcoord.csv')
    




print("Calculations Done")


print(satcords[:, 0])


exit()



# Frame rate for animation
nfr = len(satcords)
fps = 10

# Create a figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the scatter plots
sct_sat, = ax.plot([], [], [], 'o', markersize=2, color='blue', label='Satellite')
sct_obs, = ax.plot([], [], [], 'o', markersize=5, color='red', label='Observatory')

# Function to update the plot for each frame of the animation
def update(ifrm, xa, ya, za, xb, yb, zb, coords):
    ax.view_init(elev=20, azim=ifrm)
    
    t = coords.iloc[ifrm]['Time']
    ra = coords.iloc[ifrm]['Ra']
    az = coords.iloc[ifrm]['Dec']
    dec = coords.iloc[ifrm]['Azm']
    alt = coords.iloc[ifrm]['Alt']
    distance = coords.iloc[ifrm]['Dist']
    
    start = max(ifrm - 20, 0)
    sct_sat.set_data(xa[start:ifrm], ya[start:ifrm])
    sct_sat.set_3d_properties(za[start:ifrm])
    sct_obs.set_data([xb[ifrm]], [yb[ifrm]])
    sct_obs.set_3d_properties(np.array(zb[ifrm]))
    
    index.set_text("Time: " + str(t) + \
                   "\nRA: " + str(ra) + "   Azm: " + str(az) + \
                   "\nDec: " + str(dec) + "   Alt: " + str(alt) + \
                   "\nDist: {:.6f} km".format(distance))

# Set the initial view limits for the plot
ax.set_xlim(-50000, 50000)
ax.set_ylim(-50000, 50000)
ax.set_zlim(-50000, 50000)

# Annotation for displaying current time
index = ax.annotate('', xy=(0, 0.90), xycoords='axes fraction')


# Create animation
ani = animation.FuncAnimation(fig, update, nfr, fargs=(satcords[:, 0], satcords[:, 1], satcords[:, 2], obscords[:, 0], obscords[:, 1], obscords[:, 2], df), interval=1000/fps)

# Save animation as a GIF
fn = 'orbitTest'
ani.save(fn + '.gif', writer='Pillow', fps=fps)