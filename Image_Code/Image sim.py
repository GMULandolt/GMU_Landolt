#Lillehei

#--------------------------------------------------------------------------------------------------------------------
# IMPORTS
#--------------------------------------------------------------------------------------------------------------------


import aplpy
import astropy
import astroquery
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import urllib 
import pandas as pd 

from scipy.stats import norm
from urllib.parse import urlencode
from astropy.io import fits
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.hips2fits import hips2fits
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle
from astropy import wcs as astropy_wcs
from astroquery.hips2fits import conf

#--------------------------------------------------------------------------------------------------------------------
# INPUTS
#--------------------------------------------------------------------------------------------------------------------

Database = 'DSS'                 #2MASS, DSS
queryrad = "0d1m0s"                #X(d)Y(m)Z(s)


height    = 4096                   #Pixel image sizes
width     = 4096

resolution1 = 10 # times arcmin   #zoom amount

#--------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
#--------------------------------------------------------------------------------------------------------------------

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


#--------------------------------------------------------------------------------------------------------------------
# SPACECRAFT DATA
#--------------------------------------------------------------------------------------------------------------------

# 15 "/sec / 0.34 "/px = 44 px/s
# 3 "/sec / 0.34 "/px = 9 px/s
xrate     = 9           # px/s
yrate     = 44          # px/s
countrate = 3554
step      = 0.1          # time step
seeing    = 9            # px FWHM
sunfrac   = 0.0          # intensity of reflected sunlight to laser light from spacecraft
frac      = 0.25         # fraction of laser cycle on
dur       = 2.0          # laser cycle duration in seconds
sigma     = seeing /2.355
times     = np.arange(1,91,step)


y     = yrate * times
x     = xrate * times
yint  = np.round(y).astype(int)
xint  = np.round(x).astype(int)
yfrac = y - yint
xfrac = x - xint

print ("INFO: SPACECRAFT DATA IMPORTED")

#--------------------------------------------------------------------------------------------------------------------
# SIMBAD REGION QUERY
#--------------------------------------------------------------------------------------------------------------------



#      First, RA, h   m  s      Dec,  d  m  s 
# Coord = SkyCoord('0   37  41.1       -33 42 59', unit=(u.hourangle, u.deg), frame='fk5') 
# query_results = Simbad.query_region(Coord, radius =queryrad)


# print(query_results)


# object_main_id = query_results[0]['MAIN_ID']
# object_coords = SkyCoord(ra=query_results['RA'], dec=query_results['DEC'], 
#                           unit=(u.hourangle, u.deg), frame='icrs')



# query_params = { 
#      'hips': Database,
#      'object': object_main_id, 
#      'ra': object_coords[0].ra.value, 
#      'dec': object_coords[0].dec.value, 
#      'fov': (resolution1 * u.arcmin).to(u.deg).value, 
#      'width': width, 
#      'height': height
# }





#--------------------------------------------------------------------------------------------------------------------
# HIPS2FITS QUERY FOR FITS FILE
#--------------------------------------------------------------------------------------------------------------------


w = astropy_wcs.WCS(header={
	'BITPIX': 16,
    'NAXIS1': height,         # Width of the output fits/image
    'NAXIS2': width,         # Height of the output fits/image
    'WCSAXES': 2,           # Number of coordinate axes
    'CRPIX1': 1024.0,       # Pixel coordinate of reference point
    'CRPIX2': 1024.0,        # Pixel coordinate of reference point
    'CDELT1': -0.00001,        # [deg] Coordinate increment at reference point
    'CDELT2': 0.00001,         # [deg] Coordinate increment at reference point
    'CUNIT1': 'deg',        # Units of coordinate increment and value
    'CUNIT2': 'deg',        # Units of coordinate increment and value
    'CTYPE1': 'GLON-MOL',   # galactic longitude, Mollweide's projection
    'CTYPE2': 'GLAT-MOL',   # galactic latitude, Mollweide's projection
    'CRVAL1': 0.0,          # [deg] Coordinate value at reference point
    'CRVAL2': 0.0,          # [deg] Coordinate value at reference point
})

print ("INFO: WCS Header Created")

result = hips2fits.query_with_wcs(
    hips="CDS/P/2MASS/K",
    wcs = w,
    get_query_payload=False,
   
)

print ("INFO: Fits file imported")


#--------------------------------------------------------------------------------------------------------------------
# DOWNLOAD FITS FILE
#--------------------------------------------------------------------------------------------------------------------


result.writeto("2MASS_FITS.fits",overwrite=True)

#--------------------------------------------------------------------------------------------------------------------
# GRAB AND MODIFY FITS FILE
#--------------------------------------------------------------------------------------------------------------------

#hdul3 = fits.getdata("TOI_5463.01_90.000s_R-0018_out.fits")
hdul3 = fits.getdata("2MASS_FITS.fits")

print ("INFO: Fits file grabbed for modification")


data_frame = pd.DataFrame(hdul3)
print(data_frame.shape)

for i in range(len(y)):
	for j in range(-30,30):
		for k in range(-30,30):	
			if ((times[i] % dur) /dur < frac):
				hdul3[1024+xint[i]+j,yint[i]+k] += (1.0+sunfrac)*countrate*step*gaussian(np.sqrt((j+xfrac[i])*(j+xfrac[i])+(k+yfrac[i])*(k+yfrac[i])),0,sigma)
			else: 
				hdul3[1024+xint[i]+j,yint[i]+k] += (0.0+sunfrac)*countrate*step*gaussian(np.sqrt((j+xfrac[i])*(j+xfrac[i])+(k+yfrac[i])*(k+yfrac[i])),0,sigma)

hdu1=fits.PrimaryHDU(hdul3)

hdu1.writeto('test.fits',overwrite=True)

hdu11 = fits.open("test.fits")

print ("INFO: Fits file modification complete")

#--------------------------------------------------------------------------------------------------------------------
# PLOTTING FITS FILES
#--------------------------------------------------------------------------------------------------------------------


gc = aplpy.FITSFigure(result)
gc.show_colorscale(cmap='inferno')
gc.show_contour(data=result,filled=False,cmap='inferno')
gc.show_rgb()
#gc.show_markers(object_coords.ra, object_coords.dec, edgecolor='red',marker='s', s=50**2)              
gc.save('2MASS_UNEDITED.png')


gc1 = aplpy.FITSFigure(hdu11)
gc1.show_colorscale(cmap='inferno')
gc1.save('2MASS_FITS_MOD.png')

print ("INFO: PNGs Created! Closing code.")