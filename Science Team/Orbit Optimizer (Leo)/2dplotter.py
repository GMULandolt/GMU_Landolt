import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import imageio
import os
from matplotlib.lines import Line2D

res = 30

# Load the .npy file
data = np.load('air1.6res30night.npy') 

data[:,0] = data[:,0]*180/np.pi
data[:,1] = data[:,1]*180/np.pi

for i in range(len(data[:,1])):
    if data[i,1] > 90:
        data[i,1] = data[i,1] - 360
image_files = []

w_rubin  = 0.25
w_mason  = 0.25
w_palomar= 0.25
w_snifs  = 0.25

weights = np.array([w_rubin, w_mason, w_palomar, w_snifs])

# which columns correspond to each telescope?
vis_cols   = [4, 6, 8, 10]   # Rubin, Mason, Palomar, SNIFS visibility %
ecl_cols   = [5, 7, 9, 11]   # Rubin, Mason, Palomar, SNIFS eclipse days

min_vis_pct = np.min(data[:, vis_cols], axis=1)


for i in range(res): # Creates 30 frames
    #idat = data[i*res**2:(i+1)*res**2]

    x = data[:,1]
    y = data[:,0]

    # ──────────── EDIT TO plot the minimum number of hours across the four observatories ────────────
    # z = data[:, vis_cols].dot(weights)
    # use the precomputed “minimum” instead of weighted %
    z = min_vis_pct
    # ──────────── END of edit ────────────

    
    eclipse = data[:, ecl_cols].dot(weights)

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate z values onto the grid
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    # do the same for eclipse days
    eclipsei = griddata((x, y), eclipse, (Xi, Yi), method='cubic')

    plt.figure(figsize=(16, 12))
    contour = plt.contourf(Xi, Yi, Zi, levels=np.arange(0, 110), cmap='viridis', extend='min')

    
    # plt.colorbar(contour, label='Weighted-average Night Visibility Percentage')
    plt.colorbar(contour, label='Minimum Night Visibility Percentage')

    levels = [10,30,50,70,95,125,160,200,250,300,350] # [10,30,50,70,90,110,130,150,170,190]
    # levels = list(range(10, 360, 35))
    eclcont = plt.contour(Xi, Yi, eclipsei, colors='black', levels=levels)

    
    plt.clabel(eclcont, fontsize=10)
    # Scatter plot of original points
    plt.xlabel('Inclination (Deg)')
    plt.ylabel('Right Ascension of Ascending Node (Deg)')
    plt.title('Ideal Orbits (Under 1.6 Airmass)')
    #plt.gcf().text(0.75, 0.895, "RAAN: "+str(int(idat[0,0])), fontsize=12)
    filename = (
        f"16_wR{int(w_rubin*100)}"
        f"_wM{int(w_mason*100)}"
        f"_wP{int(w_palomar*100)}"
        f"_wS{int(w_snifs*100)}.png"
    )
    filename = "MIN_OF_" + filename

    # Add grid lines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Add more ticks on axes
    plt.xticks(np.arange(int(min(x)), int(max(x)) + 1, 2))
    plt.yticks(np.arange(int(min(y)), int(max(y)) + 1, 5))

    # ───────────────────── START OF LEGEND ───────────────────────
    # Make a black‐line proxy for the eclipse contours
    eclipse_proxy = Line2D([0], [0], color='black', lw=1)

    # Grab the current axes and add the legend
    ax = plt.gca()
    ax.legend(
        [eclipse_proxy],
        ['Weighted eclipse days'],
        loc='upper right',
        framealpha=0.8
    )
    # ────────────────────── END OF LEGEND ────────────────────────

    plt.savefig(filename)
    image_files.append(filename)
    plt.close()

#images = []
#for filename in image_files:
    #images.append(imageio.imread(filename))
#imageio.mimsave('rubin-15airmass-eclipseonly.gif', images, fps=1, loop=0) # Adjust fps for speed

#for filename in image_files:
    #os.remove(filename)
