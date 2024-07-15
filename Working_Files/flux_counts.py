import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from scipy.integrate import quad
from miepython import mie


"""
CODE FOR COMPUTING THE EXPECTED COUNTS AT A DETECTOR FROM THE LANDOLT SATELLITE WITH NO ATMOSPHERIC ABSORPTION
"""

data = np.genfromtxt('satcoord.csv',delimiter=',',skip_header=1) # inputs data file
z = data[:,5]
z = z*1e3
t = np.linspace(0,len(z)-1,num=len(z))*1e-3 # creates array of times incrementing in milliseconds
w_z = np.zeros(len(z))
FWHM = np.zeros(len(z))
alt = data[0,4]*(np.pi/180) # altitude of satellite in sky at the center of the beam path
alt_loc = data[:,4]*(np.pi/180) # array of altitudes of satellite in sky at any given location
beta = np.pi/2 # angle between the distance from the center of the beam path to the observatory and a line perpendicular to the beam path
alpha = np.pi/2 - alt # angle a line perpendicular to the center of the beam path makes with a tangent line located at the center of the beam path
fob = 1 # frequency of blinking (in seconds)
t_efficiency = 1 # telescope efficiency
t_inc = 1e-3 # increment of time in the file loaded in
d0 = 0 # distance of observer from center of beam path

### VARIABLES ###

MFD = 1e-5 # mode field diameter of optical fiber
w_0 = MFD/2 # waist radius of the gaussian beam
lmbda = [488e-9, 785e-9, 976e-9, 1550e-9] # wavelength of laser
lmbda_n = 0 # determines which laser is being looked at (0 - 488nm, 1 - 785nm, 2 - 976nm, 3 - 1550nm)
P_0 = [0.25, 0.1, 0.5, 0.1] # power of laser
diam_t = 0.8128 # diameter of telescope
a_t = np.pi*(diam_t/2)**2 # area that the telescope is able to take in light
current_time = 18 # current time in units of the incriment value of that data file from the start of it
aod = 0.15 # aerosol optical depth
airmass = (1/np.cos(alpha)) - 0.0018167*((1/np.cos(alpha))-1) - 0.002875*((1/np.cos(alpha))-1)**2 - 0.0008083*((1/np.cos(alpha))-1)**3 # airmass at a given altitude in the sky
latitude = np.pi/4 # lattitude of observer

### CALCULATIONS ###

I_0 = (2*P_0[lmbda_n])/(np.pi*w_0**2) # incident intensity of the laser
z_r = (np.pi/lmbda[lmbda_n])*w_0**2 # raleigh range
rad_vel = (2*np.pi)/86400 # rotational velocity of earth
r = 6371000*np.sin((np.pi/2) - latitude) # distance from observer from the axis of rotation of earth
vel = r*rad_vel # tangential velocity of the observer

if d0 == 0:
    d0 = diam_t/2 # fixes error where starting at zero creates invalid variables

# calculates flux as a gaussian distribution for height above center of beam path given
w_z0 = w_0*np.sqrt(1+(z[current_time]/z_r)**2) # beam radius at distance z
FWHM = np.sqrt(2*np.log(2))*w_z0 # full width at half maximum of the beam profile for a given distance from the waist
x = np.arange(d0, w_z0, vel/1000) # the distance on one direction perpendicular to the direction of travel
x = x[:len(t)] # cuts out extraneous variables that goes beyond the scope of the original data file
theta = np.linspace(np.arctan(x[0]/z[current_time]),np.arctan(max(x)/z[current_time]),num=len(x)) # angle made between the normal of earth's surface and a beam of light landing a given distance away from the normal
z_new = np.zeros(len(x))
w_z = np.zeros(len(x))
flux_z = np.zeros(len(x))
print('Loop Start')
for j in range(len(x)):
    z_new[j] = (z[current_time]+(0.00008*x[j]))/np.cos(theta[j]) # amount of distance a given light ray travels factoring in the curvature of the earth
    if alt_loc[j] <= alt: # identifies if observer is closer or further from the satellite using its relative altitude in the sky
        z_new[j] = z_new[j] - x[j]*np.tan(alpha)*np.sin(beta)
    else:
        z_new[j] = z_new[j] + x[j]*np.tan(alpha)*np.sin(beta)
    w_z[j] = w_0*np.sqrt(1+(z_new[j]/z_r)**2) # beam radius observed on earth's surface accounting for the curvature of earth
    flux_z[j] = I_0*((w_0/w_z[j])**2)*np.e**((-2*x[j]**2)/w_z[j]**2) # flux along one 2D slice of the 3D gaussian beam profile for different distances from the satellite in the center of the beam path
print('Done!')

dis_t = x # distance of telescope from center of beam

tflux = np.zeros(len(dis_t))
counts = np.zeros(len(dis_t))
diff_tflux = np.zeros(len(dis_t))
diff_counts = np.zeros(len(dis_t))
diff_alt = np.zeros(len(dis_t))

print('Loop Start') # finds the total flux over a given detector area
for i in range(len(dis_t)):
    def flux_fn(r):
        return r*I_0*((w_0/w_z[i])**2)*np.e**((-2*r**2)/w_z[i]**2)
    tflux_temp = quad(flux_fn, -(diam_t/2) + dis_t[i], (diam_t/2) + dis_t[i]) # flux taken in by a given telescope
    coeftemp = np.pi*(((diam_t/2) + dis_t[i])**2 - (-(diam_t/2) + dis_t[i])**2) / a_t # calculates fraction of distribution needed to be swept over to get an area a_t
    tflux[i] = tflux_temp[0]*(np.pi/coeftemp) # integrating over an angle that gives us an arclength of the diameter of the telescope
    counts[i] = (tflux[i]*lmbda[lmbda_n])/(6.62607015e-34*299792458) # total counts taken in
print('\nDone!')

"""
TELLURIC ABSORPTION
"""

# Functions for calculating the absorption coefficient

def V(x,f,alpha,gamma):
    """
    Creates a Voigt line shape at x with Lorentzian FWHM alpha and Gaussian FWHM gamma at frequency f
    """
    sigma = alpha / np.sqrt(2 * np.log(2))
    
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)

def S(lmbda, n_1, B_12, T):
    """
    Function for the line intensity at wavelength lmbda, number density of molecules 
    in their ground energy state n_1, Einstein coefficient B_12, and temperature T
    """
    
    h = 6.62607015e-34
    k = 1.380649e-23
    c = 299792458
    
    return ((lmbda**2)/8*np.pi)*n_1*B_12*(1-np.e**(-(h*c)/(lmbda*k*T)))

def abs_coef(S_12, phi):
    """
    Calculates the absorption coefficient with line intensity S_12 and line shape phi
    """
    return S_12*phi

def transmissivity(abs_coef, d):
    """
    Calculates the transmissivity of the atmosphere from the absorption coefficient and
    distance d from the satellite to the detector
    """
    return np.e**(-abs_coef*d)

# Functions for calculating the scattering coefficient

def r_coef(cs, d, N):
    """
    Using the Rayleigh scattering cross section cs, the 
    scattering coefficient is found given the distance d from the satellite to the detector,
    and the amount of molecules N per cubic meter.
    """
    r_coef = np.e**(-cs*N*d*(1/np.cos(theta)))
    return r_coef

def mie_coef(lmbda, d, N):
    """
    Calculates the scattering cross-section for mie scattering from the average index of refraction
    from aerosols in the atmosphere and a given wavelength. Using this, the scattering coefficient is 
    found from the distance d from the satellite to the detector and the amount of molecules N per
    cubic meter
    """
    m = 1.57 - 0.02j # average index of refraction for aerosols in the atmosphere
    x = 2*np.pi*1e-7 / lmbda
    gcs = np.pi * 1e-7**2
    qext, qsca, qback, g = mie(m,x)
    scs = qsca * gcs
    
    m_coef = scs*d*N 
    return m_coef

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

I_final = np.zeros(len(z_new))
counts_final = np.zeros(len(z_new))
r_coef1 = 1 - r_coef(cs_n2, z_new, N*0.78084) # scattering coefficient from rayleigh scattering for n2
r_coef2 = 1 - r_coef(cs_o2, z_new, N*0.20946) # scattering coefficient from rayleigh scattering for o2
r_coef3 = 1 - r_coef(cs_ar, z_new, N*0.00934) # scattering coefficient from rayleigh scattering for argon
r_coef4 = 1 - r_coef(cs_co2, z_new, N*0.000397) # scattering coefficient from rayleigh scattering for co2
r_coef5 = 1 - r_coef(cs_ne, z_new, N*1.818e-5) # scattering coefficient from rayleigh scattering for neon
m_coef = 1 - np.ones(len(dis_t))*np.e**(-aod*(1/np.cos(theta))) # transmission coefficient from mie scattering

for i in range(len(z_new)):
    I_final[i] = (tflux[i] - tflux[i]*(m_coef[i]+r_coef1[i]+r_coef2[i]+r_coef3[i]+r_coef4[i]+r_coef5[i])*airmass)*t_efficiency # calculates flux observed at telescope
    counts_final[i] = ((I_final[i]*lmbda[lmbda_n])/(6.62607015e-34*299792458))*t_efficiency # total counts taken in

for i in range(int(fob/1e-3)):
    I_final[i::2*int(fob/1e-3)] = 0
    counts_final[i::2*int(fob/1e-3)] = 0
    
    
I_final = I_final[0:len(t)]
counts_final = counts_final[0:len(t)]

# code for outputting the array of flux values
heading = np.array('Radiant Flux (W)',dtype='str')
heading2 = np.array('Counts')
data = np.genfromtxt('satcoord.csv',dtype='str',delimiter=',') # inputs data file
I_final = np.asarray(I_final,dtype='str')
counts_final = np.asarray(counts_final,dtype='str')
I_final = np.insert(I_final[:],0,heading)
counts_final = np.insert(counts_final[:],0,heading2)
output = np.column_stack((data,I_final))
output = np.column_stack((output,counts_final))
np.savetxt('satcoord.csv', output, fmt='%s', delimiter=',')



