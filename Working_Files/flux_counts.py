import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from scipy.integrate import quad
from settings import parameters


"""
CODE FOR COMPUTING THE EXPECTED COUNTS AT A DETECTOR FROM THE LANDOLT SATELLITE WITH NO ATMOSPHERIC ABSORPTION
"""

data = np.genfromtxt('satcoord.csv',delimiter=',',skip_header=1) # inputs data file
z = data[:,5]
z = z*1e3 # changes units of distance to meters from kilometers
tdelta = parameters.tdelta # increment of time in the file loaded in
t = np.linspace(0,len(z)-1,num=len(z))*tdelta # creates array of times incrementing with t=tdelta
w_z = np.zeros(len(z))
FWHM = np.zeros(len(z))
error_p = np.zeros(len(z))
alt = data[:,4]*(np.pi/180) # altitude of satellite in sky at the center of the beam path
alt_loc = data[0,4]*(np.pi/180) # altitude of satellite in sky at any given location
d0 = float(parameters.d0) # distance of observer from center of beam path
alpha = np.pi/2 - alt # angle a line perpendicular to the center of the beam path makes with a tangent line located at the center of the beam path
fob = 1 # frequency of blinking (in seconds)
t_efficiency = float(parameters.t_eff) # telescope efficiency

### VARIABLES ###

MFD = 1e-5 # mode field diameter of optical fiber
w_0 = MFD/2 # waist radius of the gaussian beam
lmbda = [488e-9, 785e-9, 976e-9, 1550e-9] # wavelength of laser
lmbda_n = int(parameters.n) # determines which laser is being looked at (0 - 488nm, 1 - 785nm, 2 - 976nm, 3 - 1550nm)
P_0 = [0.25, 0.1, 0.5, 0.1] # power of laser
diam_t = float(parameters.t_diam) # diameter of telescope
a_t = np.pi*(diam_t/2)**2 # area that the telescope is able to take in light
aod = 0.15 # aerosol optical depth
airmass = (1/np.cos(alpha)) - 0.0018167*((1/np.cos(alpha))-1) - 0.002875*((1/np.cos(alpha))-1)**2 - 0.0008083*((1/np.cos(alpha))-1)**3 # airmass at a given altitude in the sky

if d0 < diam_t/2:
    d0 = diam_t/2 # fixes error where starting at zero creates invalid variables, sets distance from center of the beam path to the radius of the telescope at the very minimum
    
beta = float(parameters.beta) # angle between the distance from the center of the beam path to the observatory and a line perpendicular to the beam path

### CALCULATIONS ###

I_0 = (2*P_0[lmbda_n])/(np.pi*w_0**2) # incident intensity of the laser
z_r = (np.pi/lmbda[lmbda_n])*w_0**2 # raleigh range

# calculates flux as a gaussian distribution for height above center of beam path given
w_z0 = w_0*np.sqrt(1+(z[0]/z_r)**2) # beam radius at distance z
FWHM = np.sqrt(2*np.log(2))*w_z0 # full width at half maximum of the beam profile for a given distance from the waist
x = np.arange(d0 - diam_t/2, d0 + diam_t/2, 0.0001) # the distance on one direction perpendicular to the laser vector
theta = np.arctan(d0/z) # angle made between the normal of earth's surface and a beam of light landing a given distance away from the normal
z_new = np.zeros(len(t))
w_z = np.zeros(len(t))
flux_z = np.zeros(len(t))
print('Loop Start')
for j in range(len(t)):
    z_new[j] = (z[j]+(0.00008*d0))/np.cos(theta[j]) # amount of distance a given light ray travels factoring in the curvature of the earth
    if alt_loc <= alt[j]: # identifies if observer is closer or further from the satellite using its relative altitude in the sky
        z_new[j] = z_new[j] - d0*np.tan(alpha[j])*np.sin(beta)
    else:
        z_new[j] = z_new[j] + d0*np.tan(alpha[j])*np.sin(beta)
    w_z[j] = w_0*np.sqrt(1+(z_new[j]/z_r)**2) # beam radius observed on earth's surface accounting for the curvature of earth
    flux_z[j] = I_0*((w_0/w_z[j])**2)*np.e**((-2*d0**2)/w_z[j]**2) # flux along one 2D slice of the 3D gaussian beam profile for different distances from the satellite in the center of the beam path
print('Done!')

dis_t = x # distance of telescope from center of beam
error_p[1:] = z_new[1:]*0.00000484813681109536*t[1:] # error in pointing in km

tflux = np.zeros(len(t))
counts = np.zeros(len(t))

print('Loop Start') # finds the total flux over a given detector area
for i in range(len(t)):
    def flux_fn(r):
        return r*I_0*((w_0/w_z[i])**2)*np.e**((-2*r**2)/w_z[i]**2)
    tflux_temp = quad(flux_fn, -(diam_t/2) + d0, (diam_t/2) + d0) # flux taken in by a given telescope
    coeftemp = np.pi*(((diam_t/2) + d0)**2 - (-(diam_t/2) + d0)**2) / a_t # calculates fraction of distribution needed to be swept over to get an area a_t
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

I_final = np.zeros(len(t))
counts_final = np.zeros(len(t))
r_coef1 = 1 - r_coef(cs_n2, z_new, N*0.78084) # scattering coefficient from rayleigh scattering for n2
r_coef2 = 1 - r_coef(cs_o2, z_new, N*0.20946) # scattering coefficient from rayleigh scattering for o2
r_coef3 = 1 - r_coef(cs_ar, z_new, N*0.00934) # scattering coefficient from rayleigh scattering for argon
r_coef4 = 1 - r_coef(cs_co2, z_new, N*0.000397) # scattering coefficient from rayleigh scattering for co2
r_coef5 = 1 - r_coef(cs_ne, z_new, N*1.818e-5) # scattering coefficient from rayleigh scattering for neon
m_coef = 1 - np.ones(len(t))*np.e**(-aod*(1/np.cos(theta))) # transmission coefficient from mie scattering

for i in range(len(t)):
    I_final[i] = (tflux[i] - tflux[i]*(m_coef[i]+r_coef1[i]+r_coef2[i]+r_coef3[i]+r_coef4[i]+r_coef5[i])*airmass[i])*t_efficiency # calculates flux observed at telescope
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
data = data[:,:6]
I_final = np.asarray(I_final,dtype='str')
counts_final = np.asarray(counts_final,dtype='str')
I_final = np.insert(I_final[:],0,heading)
counts_final = np.insert(counts_final[:],0,heading2)
output = np.column_stack((data,I_final))
output = np.column_stack((output,counts_final))
np.savetxt('satcoord.csv', output, fmt='%s', delimiter=',')



