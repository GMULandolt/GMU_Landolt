import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Working_Files.TLEconstructor import satcords, obscords, df


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
ani.save(fn + '.gif', writer='imagemagick', fps=fps)