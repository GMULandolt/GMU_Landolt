#Elizabeth Lillehei

#--------------------------------------------------------------------------------------------------------------------
# IMPORTS
#--------------------------------------------------------------------------------------------------------------------


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

Database = 'CDS/P/2MASS/K'        #Hips2fits database


height    = 1024                   #Pixel image sizes
width     = 1024
pixel     = 0.0002777777778     # [deg] Coordinate increment at reference point. 0.0002777777778 corresponds to 1px/arcsecond


#--------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
#--------------------------------------------------------------------------------------------------------------------

# 15 "/sec / 0.34 "/px = 44 px/s
# 3 "/sec / 0.34 "/px = 9 px/s

#--------------------------------------------------------------------------------------------------------------------
# SPACECRAFT DATA 
#--------------------------------------------------------------------------------------------------------------------

data = np.genfromtxt('satcoord.csv',delimiter=',',skip_header=1) # inputs data file

TimeEST = data[:,0]
RA      = data[:,1]
Dec     = data[:,2]
Az      = data[:,3]
Alt     = data[:,4]
Dist    = data[:,5]
R_Flux  = data[:,6]
# Count   = data[:,7]
# AirMass = data[:,8]

print(RA)
print(Dec)

def find_middle(lst):
    length = len(lst)  # Get the length of the list
 
    if length % 2 != 0:  # Check if the length is odd
        middle_index = length // 2
        return lst[middle_index]
    else:
        second_middle_index = length // 2
        return lst[second_middle_index]

    

RA_import = find_middle(RA)
Dec_import = find_middle(Dec)

print(len(data))

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
    'WCSAXES': 2,           # Number of coordinate axes
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel,        
    'CRPIX1': 512, 
    'CRVAL1': RA_import,
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel,      
    'CRPIX2': 512, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
})


wcs_input_dict = {
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel, 
    'CRPIX1': 1, 
    'CRVAL1': RA_import, 
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel, 
    'CRPIX2': 1, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
}

print ("INFO: WCS Header Created")

result = hips2fits.query_with_wcs(
    hips=Database,
    wcs = w,
    get_query_payload=False,
   
)

print ("INFO: Fits file imported")


#--------------------------------------------------------------------------------------------------------------------
# DOWNLOAD FITS FILE
#--------------------------------------------------------------------------------------------------------------------


result.writeto("2MASS_FITS.fits",overwrite=True)

#--------------------------------------------------------------------------------------------------------------------
# MODIFY WCS HEADER TO PREPARE FOR EDITING
#--------------------------------------------------------------------------------------------------------------------


wcs_landolt_dict = astropy_wcs.WCS(wcs_input_dict)

header_data_unit_list = fits.open("2MASS_FITS.fits")

header_data_unit_list.info()

image1 = header_data_unit_list[0].data

header1 =header_data_unit_list[0].header

wcs_landolt = astropy_wcs.WCS(header1)






#--------------------------------------------------------------------------------------------------------------------
# GRAB AND MODIFY FITS FILE
#--------------------------------------------------------------------------------------------------------------------
print ("INFO: Fits file grabbed for modification")

fitsfile = fits.open("2MASS_FITS.fits")







# fig = plt.figure(figsize=(10, 10), frameon=False)
# ax = plt.subplot(projection=wcs_landolt)

#          head_width=0, head_length=0, 
#          fc='red', ec='red', width=0.003, 
#          transform=ax.get_transform('icrs'))
# plt.text(RA[-1], -4.075, '0.1 deg', 
#          color='white', rotation=90, 
#          transform=ax.get_transform('icrs'))

def Landoltplot(image,figsize=(15,15),cmap='inferno',scale=0.5,colorbar=False,header=None,wcsplot=None,**kwargs):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(projection=wcsplot)
    mu = np.mean(image)
    s = np.std(image)
    dvmin = mu - scale*s
    dvmax = mu + scale*s
    if all(['vmin','vmax']) in kwargs.keys():
       im = ax.imshow(image,origin='lower',cmap=cmap,vmin=kwargs['vmin'],vmax=kwargs['vmax'])
    elif 'vmin' in kwargs.keys():
        im = ax.imshow(image,origin='lower',cmap=cmap,vmin=kwargs['vmin'],vmax=dvmax)
    elif 'vmax' in kwargs.keys():
        im = ax.imshow(image,origin='lower',cmap=cmap,vmin=dvmin,vmax=kwargs['vmax'])
    else:
        im = ax.imshow(image,origin='lower',cmap=cmap,vmin=dvmin,vmax=dvmax)
    if colorbar:
        cbar = plt.colorbar(im,ax=ax)


    ax.arrow(RA[0], Dec[0]+((1/15)*Dec[0]), (RA[-1]-RA[0]), (Dec[-1]-Dec[0]),
             head_width=0, head_length=0, 
            fc='green', ec='green', width=0.0003, 
            transform=ax.get_transform('icrs')) 
    
    ax.arrow(RA[0], Dec[0]-((1/15)*Dec[0]), (RA[-1]-RA[0]), (Dec[-1]-Dec[0]), 
             head_width=0, head_length=0, 
            fc='green', ec='green', width=0.0003, 
            transform=ax.get_transform('icrs')) 
    

    # ax.arrow(RA[0], Dec[0], (RA[-1]-RA[0]), (Dec[-1]-Dec[0]), 
    #          head_width=0, head_length=0, 
    #         fc='red', ec='red', width=0.0003, 
    #         transform=ax.get_transform('icrs'))

    for i in range(0,len(data)):
        if R_Flux[i] == 0:
            ax.scatter(RA[i],Dec[i], s=0.0001, marker=".", edgecolors=None, alpha= 0, transform=ax.get_transform('icrs'))
        else:
            ax.scatter(RA[i],Dec[i], s=0.0001, marker=".", edgecolors=None, transform=ax.get_transform('icrs'))


    
    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='white', ls='dotted')
    plt.xlabel(r'RA')
    plt.ylabel(r'Dec')
    plt.savefig("Image_sim.png")
    plt.show()
    return fig, ax    



Landoltplot(image1,scale=0.5, colorbar=True,wcsplot=wcs_landolt, vmin=-4.677, vmax=10)


print ("INFO: PNGs Created! Closing code.")