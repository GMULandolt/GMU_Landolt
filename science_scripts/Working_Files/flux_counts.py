import numpy as np
from scipy.integrate import quad
from settings import parameters

"""
CODE FOR COMPUTING THE EXPECTED COUNTS AT A DETECTOR FROM THE LANDOLT SATELLITE WITH NO ATMOSPHERIC ABSORPTION
"""

data = np.genfromtxt('satcoord.csv',delimiter=',',skip_header=1) # inputs data file from orbit simulations
datalatlon = np.genfromtxt('satlatlon.csv',delimiter=',',skip_header=1) # inputs latitude and longitude information of the satellite
dataxyz = np.genfromtxt('satcoordxyz.csv',delimiter=',',skip_header=1) # inputs position data of the satellite

### VARIABLES ###

tdelta = parameters.tdelta # increment of time in the file loaded in
lat_obs = parameters.lat*(np.pi/180) # latitude of the center of the beam path
lon_obs = parameters.lon*(np.pi/180) # longitude of the center of the beam path
lat_loc = float(parameters.lat_loc)*(np.pi/180) # latitude of observer
lon_loc = float(parameters.lon_loc)*(np.pi/180) # longitude of observer
t_efficiency = float(parameters.t_eff) # telescope efficiency
ccd_efficiency = float(parameters.ccd_eff) # CCD efficiency
diam_t = float(parameters.t_diam) # diameter of telescope
lmbda_n = int(parameters.n) # determines which laser is being looked at (0 - 488nm, 1 - 785nm, 2 - 976nm, 3 - 1550nm)
humidity = float(parameters.humidity)
fob = 1 # frequency of blinking (in seconds)
aod = 0.1 # aerosol optical depth
# aod varies w/ humidity, the code factors that in here
if humidity >= 0.6:
    aod = aod + 0.05
if humidity >= 0.8:
    aod = aod + 0.05

z = data[:,5]*1e3 # extracting distance of observer to satellite in units of meters
alt = data[:,4]*(np.pi/180) # altitude of satellite in sky at the center of the beam path
alt0 = data[0,4]*(np.pi/180) # altitude of satellite in sky at any given location
alpha = np.pi/2 - alt # angle a line perpendicular to the center of the beam path makes with a tangent line located at the center of the beam path
t = np.linspace(0,len(z)-1,num=len(z))*tdelta # creates array of times incrementing with t=tdelta
airmass = (1/np.cos(alpha)) - 0.0018167*((1/np.cos(alpha))-1) - 0.002875*((1/np.cos(alpha))-1)**2 - 0.0008083*((1/np.cos(alpha))-1)**3 # airmass at a given altitude in the sky

orient_x = 6371000*np.sin((np.pi/2) - lat_obs)*np.cos(lon_obs) # location of center of beam path in the x direction using the volumetric mean radius of earth
orient_y = 6371000*np.sin((np.pi/2) - lat_obs)*np.sin(lon_obs) # similarly for the y direction
orient_z = 6371000*np.cos((np.pi/2) - lat_obs) # similarly for the z direction
orient_xloc = 6371000*np.sin((np.pi/2) - lat_loc)*np.cos(lon_loc) # location of the observer in the x direction using the volumetric mean radius of earth
orient_yloc = 6371000*np.sin((np.pi/2) - lat_loc)*np.sin(lon_loc) # similarly for the y direction
orient_zloc = 6371000*np.cos((np.pi/2) - lat_loc) # similarly for the z direction
orient_xloc = orient_xloc - orient_x # to get vector from center of the beam path to observer
orient_yloc = orient_yloc - orient_y # similarly
orient_zloc = orient_zloc - orient_z # similarly
sat_x = dataxyz[:,0] # x position of satellite in GCRS coordinates
sat_y = dataxyz[:,1] # y position of satellite in GCRS coordinates
sat_z = dataxyz[:,2] # z position of satellite in GCRS coordinates
sat_lat = datalatlon[:,0] # latitude of satellite projected to earth
sat_lon = datalatlon[:,1] # longitude of satellite projected to earth
r_sat = np.sqrt(sat_x**2 + sat_y**2 + sat_z**2)*1e3 # distance from satellite to the center of the earth in meters
orient_xsat = r_sat*np.sin((np.pi/2) - sat_lat)*np.cos(sat_lon) # location of the satellite in the x direction in lat/lon coordinates
orient_ysat = r_sat*np.sin((np.pi/2) - sat_lat)*np.sin(sat_lon) # similarly for the y direction
orient_zsat = r_sat*np.cos((np.pi/2) - sat_lat) # similarly for the z direction
d0 = np.sqrt(orient_xloc**2 + orient_yloc**2 + orient_zloc**2) # distance of observer from center of beam path
beta = np.arccos(((orient_xloc*orient_xsat) + (orient_yloc*orient_ysat) + (orient_zloc*orient_zsat))/(d0*r_sat)) # angle between a line made between the center of the beam path and observer and the satellite-to-center of beam path vector projected on to earth's surface
if d0 < diam_t/2:
    d0 = diam_t/2 # fixes error where starting at zero creates invalid variables, sets distance from center of the beam path to the radius of the telescope at the very minimum

w_z = np.zeros(len(z))
FWHM = np.zeros(len(z))
error_p = np.zeros(len(z))
z_new = np.zeros(len(t))
w_z = np.zeros(len(t))
flux_z = np.zeros(len(t))
tflux = np.zeros(len(t))
counts = np.zeros(len(t))
I_final = np.zeros(len(t))
counts_final = np.zeros(len(t))

MFD = 1e-5 # mode field diameter of optical fiber
w_0 = MFD/2 # waist radius of the gaussian beam
lmbda = [488e-9, 785e-9, 976e-9, 1550e-9] # wavelength of all four lasers
P_0 = [0.25, 0.1, 0.5, 0.1] # power of all four lasers
a_t = np.pi*(diam_t/2)**2 # area that the telescope is able to take in light

### CALCULATIONS ###

I_0 = (2*P_0[lmbda_n])/(np.pi*w_0**2) # incident intensity of the laser
z_r = (np.pi/lmbda[lmbda_n])*w_0**2 # raleigh range

# calculates flux as a gaussian distribution for height above center of beam path given
w_z0 = w_0*np.sqrt(1+(z[0]/z_r)**2) # beam radius at distance z
FWHM = np.sqrt(2*np.log(2))*w_z0 # full width at half maximum of the beam profile for a given distance from the waist
x = np.arange(d0 - diam_t/2, d0 + diam_t/2, 0.0001) # the distance on one direction perpendicular to the laser vector
theta = np.arctan(d0/z) # angle made between the normal of earth's surface and a beam of light landing a given distance away from the normal
print('Calculating Gaussian distribution of flux...')
for j in range(len(t)):
    z_new[j] = (z[j]+(0.00008*d0))/np.cos(theta[j]) # amount of distance a given light ray travels factoring in the curvature of the earth
    if alt0 <= alt[j]: # identifies if observer is closer or further from the satellite using its relative altitude in the sky
        z_new[j] = z_new[j] - d0*np.tan(alpha[j])*np.sin(beta[j])
    else:
        z_new[j] = z_new[j] + d0*np.tan(alpha[j])*np.sin(beta[j])
    w_z[j] = w_0*np.sqrt(1+(z_new[j]/z_r)**2) # beam radius observed on earth's surface accounting for the curvature of earth
    flux_z[j] = I_0*((w_0/w_z[j])**2)*np.e**((-2*d0**2)/w_z[j]**2) # flux along one 2D slice of the 3D gaussian beam profile for different distances from the satellite in the center of the beam path
print('Done!')

dis_t = x # distance of telescope from center of beam
error_p[1:] = z_new[1:]*0.00000484813681109536*t[1:] # error in pointing in km

print('Calculating flux recieved at telescope...') # finds the total flux over a given detector area
for i in range(len(t)):
    def flux_fn(r):
        return r*I_0*((w_0/w_z[i])**2)*np.e**((-2*r**2)/w_z[i]**2)
    tflux_temp = quad(flux_fn, -(diam_t/2) + d0, (diam_t/2) + d0) # flux taken in by a given telescope
    coeftemp = np.pi*(((diam_t/2) + d0)**2 - (-(diam_t/2) + d0)**2) / a_t # calculates fraction of distribution needed to be swept over to get an area a_t
    tflux[i] = tflux_temp[0]*(np.pi/coeftemp) # integrating over an angle that gives us an arclength of the diameter of the telescope
    counts[i] = (tflux[i]*lmbda[lmbda_n])/(6.62607015e-34*299792458) # total counts taken in
print('Done!')

# Functions for calculating the scattering coefficient

def r_coef(cs, d, N):
    """
    Using the Rayleigh scattering cross section cs, the 
    scattering coefficient is found given the distance d from the satellite to the detector,
    and the amount of molecules N per cubic meter.
    """
    r_coef = np.e**(-cs*N*d*(1/np.cos(theta)))
    return r_coef

### CALCULATING INTENSITY WITH ATMOSPHERIC ABSORPTION ###

N = 1e44 / (((4/3)*np.pi*(6371000 + z_new)**3) - (4/3)*np.pi*(6371000)**3)# average number of molecules that measured light passes per square meter

# rayleigh scattering cross sections for 488 nm for the five most abundant gases in the atmosphere
if lmbda_n == 0:
    cs_n2 = 7.26e-31
    cs_o2 = 6.50e-31
    cs_ar = 7.24e-31
    cs_co2 = 23e-31
    cs_ne = 0.33e-31

# rayleigh scattering cross sections for 785 nm for the five most abundant gases in the atmosphere
elif lmbda_n == 1:
    cs_n2 = 2.65e-31
    cs_o2 = 2.20e-31
    cs_ar = 2.38e-31
    cs_co2 = 6.22e-31
    cs_ne = 0.128e-31

# rayleigh scattering cross sections for 976nm for the five most abundant gases in the atmosphere
# these values are extrapolated assuming a direct 1/lambda^4 relationship
elif lmbda_n == 2:
    cs_n2 = 4.54e-32
    cs_o2 = 4.06e-32
    cs_ar = 4.53e-32
    cs_co2 = 1.44e-31
    cs_ne = 2.06e-33

# rayleigh scattering cross sections for 1550nm for the five most abundant gases in the atmosphere
# these values are extrapolated assuming a direct 1/lambda^4 relationship
else:
    cs_n2 = 7.13e-33
    cs_o2 = 6.39e-33
    cs_ar = 7.11e-33
    cs_co2 = 2.26e-32
    cs_ne = 3.24e-34

r_coef1 = 1 - r_coef(cs_n2, z_new, N*0.78084) # scattering coefficient from rayleigh scattering for n2
r_coef2 = 1 - r_coef(cs_o2, z_new, N*0.20946) # scattering coefficient from rayleigh scattering for o2
r_coef3 = 1 - r_coef(cs_ar, z_new, N*0.00934) # scattering coefficient from rayleigh scattering for argon
r_coef4 = 1 - r_coef(cs_co2, z_new, N*0.000397) # scattering coefficient from rayleigh scattering for co2
r_coef5 = 1 - r_coef(cs_ne, z_new, N*1.818e-5) # scattering coefficient from rayleigh scattering for neon
m_coef = 1 - np.ones(len(t))*np.e**(-aod*(1/np.cos(theta))) # transmission coefficient from mie scattering

for i in range(len(t)):
    I_final[i] = (tflux[i] - tflux[i]*(m_coef[i]+r_coef1[i]+r_coef2[i]+r_coef3[i]+r_coef4[i]+r_coef5[i])*airmass[i])*t_efficiency # calculates flux observed at telescope
    counts_final[i] = ((I_final[i]*lmbda[lmbda_n])/(6.62607015e-34*299792458))*ccd_efficiency # total counts taken in

for i in range(int(fob/1e-3)):
    I_final[i::2*int(fob/1e-3)] = 0
    counts_final[i::2*int(fob/1e-3)] = 0
    
    
I_final = I_final[0:len(t)]
counts_final = counts_final[0:len(t)]

# code for outputting the array of flux values
heading = np.array('Radiant Flux (W)',dtype='str')
heading2 = np.array('Counts')
heading3 = np.array('Airmass')
data = np.genfromtxt('satcoord.csv',dtype='str',delimiter=',') # inputs data file
data = data[:,:6]
I_final = np.asarray(I_final,dtype='str')
counts_final = np.asarray(counts_final,dtype='str')
airmass = np.asarray(airmass,dtype='str')
I_final = np.insert(I_final[:],0,heading)
counts_final = np.insert(counts_final[:],0,heading2)
airmass = np.insert(airmass[:],0,heading3)
output = np.column_stack((data,I_final))
output = np.column_stack((output,counts_final))
output = np.column_stack((output,airmass))
np.savetxt('satcoord.csv', output, fmt='%s', delimiter=',')


