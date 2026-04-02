import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import imageio
import os

res = 3

# Load the .npy file
data = np.load('air15res3.npy') 

data[:,0] = data[:,0]*180/np.pi
data[:,1] = data[:,1]*180/np.pi

image_files = []
for i in range(res): # Creates 20 frames
    # Print the loaded array

    idat = data[i*res**2:(i+1)*res**2]

    x = idat[:,1]
    y = idat[:,2]
    z = idat[:,3]

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate z values onto the grid
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')


    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Xi, Yi, Zi, levels=np.arange(0, 31), cmap='viridis', extend='min')
    plt.colorbar(contour, label='Average Percentage of Year Sattelite is Observable')
    # Scatter plot of original points
    plt.xlabel('Inclination (Deg)')
    plt.ylabel('Eccentricity (Deg)')
    plt.title('Ideal Orbit For The 4 Observatories (GEOSYNCHRONOUS)')
    plt.gcf().text(0.75, 0.895, "RAAN: "+str(int(idat[0,0])), fontsize=12)
    filename = f"plot_{i}.png"
    plt.savefig(filename)
    image_files.append(filename)
    plt.close()

images = []
for filename in image_files:
    images.append(imageio.imread(filename))
imageio.mimsave('foo.gif', images, fps=1, loop=0) # Adjust fps for speed