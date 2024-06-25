import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from scipy.integrate import quad
from miepython import mie


"""
CODE FOR COMPUTING THE EXPECTED COUNTS AT A DETECTOR FROM THE LANDOLT SATELLITE WITH NO ATMOSPHERIC ABSORPTION
"""

data = np.genfromtxt('satcoord.csv',delimiter=',',skip_header=1)
z = data[:,5]
z = z*1e3
t = np.linspace(0,len(z)-1,num=len(z))

### VARIABLES ###

MFD = 1e-5 # mode field diameter of optical fiber
w_0 = MFD/2 # waist radius of the gaussian beam
lmbda = [488e-9, 785e-9, 976e-9, 1550e-9] # wavelength of laser
lmbda_n = 0 # determines which laser is being looked at
P_0 = [0.25, 0.1, 0.5, 0.1] # power of laser
diam_t = 0.8128 # diameter of telescope
a_t = np.pi*(0.8128/2)**2 # area that the telescope is able to take in light
current_time = 12 # time from start of heigh data file in units of the incriment value of that data file

### CALCULATIONS ###

I_0 = (2*P_0[lmbda_n])/(np.pi*w_0**2) # incident intensity of the laser
z_r = (np.pi/lmbda[lmbda_n])*w_0**2 # raleigh range

w_z = np.zeros(len(z))
FWHM = np.zeros(len(z))
flux = np.zeros([len(z),10000])

print('Loop 1') # calculates flux as a gaussian distribution per height given
for i in range(len(z)):
    w_z[i] = w_0*np.sqrt(1+(z[i]/z_r)**2) # beam radius at distance z
    FWHM[i] = np.sqrt(2*np.log(2))*w_z[i] # full width at half maximum of the beam profile for a given distance from the waist
    x = np.linspace(-w_z[i], w_z[i],num=int(10000)) # the range of light in one direction perpendicular to the direction of travel
    for j in range(len(x)):   
        flux[i,j] = I_0*((w_0/w_z[i])**2)*np.e**((-2*x[j]**2)/w_z[i]**2) # flux along one 2D slice of the 3D gaussian beam profile
    print('\r' + str(int(i/len(z) * 10000)/100) + "%", end='', flush=True)
print('Done!')

plt.figure()
plt.plot(x,flux[current_time])
plt.xlabel('Displacement from original beam path (m)')
plt.ylabel('Intensity (W/m\u00b2)')

dis_t = np.linspace(1,w_z[current_time],num=10000) # distance of telescope from center of beam
theta = np.linspace(np.arctan(dis_t[0]/z[current_time]),np.arctan(max(dis_t)/z[current_time]),num=10000) # angle made between the normal of earth's surface and a beam of light landing a given distance away from the normal

z_new = np.zeros(len(dis_t))
tflux = np.zeros(len(dis_t))
counts = np.zeros(len(dis_t))
diff_tflux = np.zeros(len(dis_t))
diff_counts = np.zeros(len(dis_t))
diff_alt = np.zeros(len(dis_t))

print('Loop 2') # finds the total flux over a given detector area assuming the detector is in the center of the beam
for i in range(len(dis_t)):
    def flux_fn(r):
        return r*I_0*((w_0/w_z[current_time])**2)*np.e**((-2*r**2)/w_z[current_time]**2)
    z_new[i] = (z[current_time]+(0.00008*dis_t[i]))/np.cos(theta[i])
    tflux_temp = quad(flux_fn, -(diam_t/2) + dis_t[i], (diam_t/2) + dis_t[i]) # flux taken in by a given telescope
    coeftemp = np.pi*(((diam_t/2) + dis_t[i])**2 - (-(diam_t/2) + dis_t[i])**2) / a_t # calculates fraction of distribution needed to be swept over to get an area a_t
    tflux[i] = tflux_temp[0]*(np.pi/coeftemp) # integrating over an angle that gives us an arclength of the diameter of the telescope
    counts[i] = (tflux[i]*lmbda[lmbda_n])/(6.62607015e-34*299792458) # total counts taken in
    print('\r' + str(int(i/len(dis_t) * 10000)/100) + "%", end='', flush=True)
print('Done!')

"""
TELLURIC ABSORPTION
"""

# Functions for calculating the absorption coefficient

def V(x,f):
    """
    Creates a Voigt line shape at x with Lorentzian FWHM alpha and Gaussian FWHM gamma at frequency f
    """
    
    alpha = 7.17e-7*f*np.sqrt(287/28.96) # uses rms velocity of molecules in atmosphere, average temperature of Earth, and g/mol of average atmosphere
    
    gamma = 0.115 # overestimation, taken from princeton source
    
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
    r_coef = cs*d*N
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

# rayleigh scattering cross sections for 488 nm

cs_n2 = 7.26e-31
cs_o2 = 6.50e-31
cs_ar = 7.24e-31
cs_co2 = 23e-31
cs_ne = 0.33e-31

# rayleigh scattering cross sections for 785 nm

#cs_n2 = 2.65e-31
#cs_o2 = 2.20e-31
#cs_ar = 2.38e-31
#cs_co2 = 6.22e-31
#cs_ne = 0.128e-31

I_final = np.zeros(len(z_new))
r_coef = r_coef(cs_n2, z_new, N*0.78084) + r_coef(cs_o2, z_new, N*0.20946) + r_coef(cs_ar, z_new, N*0.00934) + r_coef(cs_co2, z_new, N*0.000397) + r_coef(cs_ne, z_new, N*1.818e-5) # scattering coefficient from rayleigh scattering
m_coef = np.ones(len(dis_t))*np.e**(-0.15*(1/np.cos(theta[i])))

for i in range(len(z_new)):
    I_final[i] = tflux[i]*m_coef[i] - tflux[i]*(r_coef[i]) # calculates flux observed at telescope

plt.figure()
plt.plot(dis_t, I_final)
plt.xlabel('Displacement from original beam path (m)')
plt.ylabel('Intensity (W/m\u00b2)')

