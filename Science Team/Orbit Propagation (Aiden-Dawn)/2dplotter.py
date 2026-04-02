import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import imageio
import os

res = 20

# Load the .npy file
data = np.load('air15res20subgeo.npy') 

data[:,0] = data[:,0]*180/np.pi
data[:,1] = data[:,1]*180/np.pi
print(max(data[:,10]))
for i in range(len(data[:,1])):
    if data[i,1] > 90:
        data[i,1] = data[i,1] - 360
image_files = []
for i in range(res): # Creates 20 frames
    # Print the loaded array

    #idat = data[i*res**2:(i+1)*res**2]

    x = data[:,1]
    y = data[:,0]
    z = data[:,4]
    eclipse = data[:,5]

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate z values onto the grid
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    # do the same for eclipse days
    eclipsei = griddata((x, y), eclipse, (Xi, Yi), method='cubic')

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Xi, Yi, Zi, levels=np.arange(0, 35), cmap='viridis', extend='min')
    plt.colorbar(contour, label='Percentage of Nights Sattelite is Observable Over a Year')
    eclcont = plt.contour(Xi, Yi, eclipsei, colors='black', levels=[10,30,50,70,90,110,130,150,170,190])
    plt.clabel(eclcont, fontsize=10)
    # Scatter plot of original points
    plt.xlabel('Inclination (Deg)')
    plt.ylabel('Right Ascension of Ascending Node (Deg)')
    plt.title('Ideal Orbit For SNIFS (Under 2.0 Airmass)')
    #plt.gcf().text(0.75, 0.895, "RAAN: "+str(int(idat[0,0])), fontsize=12)
    filename = "test.png"
    plt.savefig(filename)
    image_files.append(filename)
    plt.close()

#images = []
#for filename in image_files:
    #images.append(imageio.imread(filename))
#imageio.mimsave('rubin-15airmass-eclipseonly.gif', images, fps=1, loop=0) # Adjust fps for speed

#for filename in image_files:
    #os.remove(filename)