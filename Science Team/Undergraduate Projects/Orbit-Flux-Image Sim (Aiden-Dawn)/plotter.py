import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import imageio
import os

res = 10

# Load the .npy file
data = np.load('air12res10neargeo.npy') 

data[:,2] = data[:,2]*180/np.pi
data[:,1] = data[:,1]*180/np.pi
for i in range(len(data[:,1])):
    if data[i,1] > 90:
        data[i,1] = data[i,1] - 360
print(data[:,4].max())
image_files = []
for i in range(res): # Creates 20 frames
    # Print the loaded array

    idat = data[i*res**2:(i+1)*res**2]

    x = idat[:,1]
    y = idat[:,2]
    z = idat[:,10]
    eclipse = idat[:,11]

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate z values onto the grid
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    # do the same for eclipse days
    eclipsei = griddata((x, y), eclipse, (Xi, Yi), method='cubic')

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Xi, Yi, Zi, levels=np.arange(0, 110), cmap='viridis', extend='both')
    plt.colorbar(contour, label='Average Percentage of Year Sattelite is Observable')
    eclcont = plt.contour(Xi, Yi, eclipsei, colors='black', levels=[10,30,50,70,90,110,130,150,170,190])
    plt.clabel(eclcont, fontsize=10)
    # Scatter plot of original points
    plt.xlabel('Inclination (Deg)')
    plt.ylabel('Right Ascension of Ascending Node (Deg)')
    plt.title('Ideal Orbit For SNIFS (Under 1.2 Airmass)')
    idat[0,0] = ((6.6743*10**-20*5.972*10**24)*((60/idat[0,0])**2))**(1/3) - 6378
    plt.gcf().text(0.75, 0.895, "Altitude (km): "+str(np.round(idat[0,0], 1)), fontsize=8)
    filename = f"plot_{i}.png"
    plt.savefig(filename)
    image_files.append(filename)
    plt.close()

images = []
for filename in image_files:
    images.append(imageio.imread(filename))
imageio.mimsave('snifs-12airmass-neargeo.gif', images, fps=1, loop=0) # Adjust fps for speed

#for filename in image_files:
    #os.remove(filename)