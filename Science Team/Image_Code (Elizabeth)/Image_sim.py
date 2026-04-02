#Elizabeth Lillehei

#GMU Landolt Image simulations
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

# Multiplying this number by 12.5 nets us 3.555556, or 1px/ 12.5 arcseconds, This will be the largest number we will use

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

print("The length of the imported CSV File is {}".format(len(data)))
print("If the CSV file is over 10,000 lines you may encounter long wait times \n")

print ("SPACECRAFT DATA IMPORTED")

#--------------------------------------------------------------------------------------------------------------------
# HIPS2FITS QUERY FOR FITS FILE
#--------------------------------------------------------------------------------------------------------------------

print("Your Imported RA is {} \n".format(abs(RA[-1]-RA[0])))

if abs(RA[-1]-RA[0]) >= 3.55:
    print("RA Tracking too large! Please limit the scale of the data to under 3.55 degrees")
elif 2.84  <= abs(RA[-1]-RA[0]) < 3.55:
    w = astropy_wcs.WCS(header={
	'BITPIX': 16,
    'WCSAXES': 2,           # Number of coordinate axes
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel * 12.5,        
    'CRPIX1': 512, 
    'CRVAL1': RA_import,
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel * 12.5,      
    'CRPIX2': 512, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
    })

    wcs_input_dict = {
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel * 12.5, 
    'CRPIX1': 1, 
    'CRVAL1': RA_import, 
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel * 12.5, 
    'CRPIX2': 1, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
    }
    satzoom = 30
    print("For this image, 1px is 12.5 arcseconds")
elif 2.27  <= abs(RA[-1]-RA[0]) < 2.84:
    w = astropy_wcs.WCS(header={
	'BITPIX': 16,
    'WCSAXES': 2,           # Number of coordinate axes
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel * 10,        
    'CRPIX1': 512, 
    'CRVAL1': RA_import,
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel * 10,      
    'CRPIX2': 512, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
    })
    wcs_input_dict = {
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel * 10, 
    'CRPIX1': 1, 
    'CRVAL1': RA_import, 
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel * 10, 
    'CRPIX2': 1, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
    }
    satzoom = 30
    print("For this image, 1px is 10 arcseconds")
elif 1.70  <= abs(RA[-1]-RA[0]) < 2.27:
    w = astropy_wcs.WCS(header={
	'BITPIX': 16,
    'WCSAXES': 2,           # Number of coordinate axes
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel * 8,        
    'CRPIX1': 512, 
    'CRVAL1': RA_import,
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel * 8,      
    'CRPIX2': 512, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
    })
    wcs_input_dict = {
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel * 8, 
    'CRPIX1': 1, 
    'CRVAL1': RA_import, 
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel * 8, 
    'CRPIX2': 1, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
    }
    satzoom = 30
    print("For this image, 1px is 8 arcseconds")
elif 1.13  <= abs(RA[-1]-RA[0]) < 1.70:
    w = astropy_wcs.WCS(header={
	'BITPIX': 16,
    'WCSAXES': 2,           # Number of coordinate axes
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel * 6,        
    'CRPIX1': 512, 
    'CRVAL1': RA_import,
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel * 6,      
    'CRPIX2': 512, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
    })
    wcs_input_dict = {
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel * 6, 
    'CRPIX1': 1, 
    'CRVAL1': RA_import, 
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel * 6, 
    'CRPIX2': 1, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
    }
    satzoom = 30
    print("For this image, 1px is 6 arcseconds")
elif 0.56  <= abs(RA[-1]-RA[0]) < 1.13:
    w = astropy_wcs.WCS(header={
	'BITPIX': 16,
    'WCSAXES': 2,           # Number of coordinate axes
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel * 4,        
    'CRPIX1': 512, 
    'CRVAL1': RA_import,
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel * 4,      
    'CRPIX2': 512, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
    })
    wcs_input_dict = {
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel * 4, 
    'CRPIX1': 1, 
    'CRVAL1': RA_import, 
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel * 4, 
    'CRPIX2': 1, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
    }
    satzoom = 30
    print("For this image, 1px is 4 arcseconds")
elif 0.28  <= abs(RA[-1]-RA[0]) < 0.56:
    w = astropy_wcs.WCS(header={
	'BITPIX': 16,
    'WCSAXES': 2,           # Number of coordinate axes
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel * 2,        
    'CRPIX1': 512, 
    'CRVAL1': RA_import,
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel * 2,      
    'CRPIX2': 512, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
    })

    wcs_input_dict = {
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': pixel * 2, 
    'CRPIX1': 1, 
    'CRVAL1': RA_import, 
    'NAXIS1': height,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': pixel * 2, 
    'CRPIX2': 1, 
    'CRVAL2': Dec_import, 
    'NAXIS2': width
    }
    satzoom = 35
    print("For this image, 1px is 2 arcseconds")
elif 0.01 <= abs(RA[-1]-RA[0]) < 0.28:
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
    satzoom = 50
    print("For this image, 1px is 1 arcsecond")
elif 0.00  <= abs(RA[-1]-RA[0]) < 0.01:
    print("RA Tracking too small! Please limit the scale of the date to over 0.1 degrees.") 

print ("WCS Header Created")

result = hips2fits.query_with_wcs(
    hips=Database,
    wcs = w,
    get_query_payload=False,
)

print ("Fits file imported")


#--------------------------------------------------------------------------------------------------------------------
# DOWNLOAD FITS FILE
#--------------------------------------------------------------------------------------------------------------------


result.writeto("2MASS_FITS.fits",overwrite=True)

#--------------------------------------------------------------------------------------------------------------------
# MODIFY WCS HEADER TO PREPARE FOR EDITING
#--------------------------------------------------------------------------------------------------------------------


wcs_landolt_dict = astropy_wcs.WCS(wcs_input_dict)

header_data_unit_list = fits.open("2MASS_FITS.fits")

print("Displaying Basic FITS header data")
header_data_unit_list.info()

print()

image1 = header_data_unit_list[0].data

header1 =header_data_unit_list[0].header

wcs_landolt = astropy_wcs.WCS(header1)






#--------------------------------------------------------------------------------------------------------------------
# GRAB AND MODIFY FITS FILE
#--------------------------------------------------------------------------------------------------------------------
print ("\nFile grabbed for modification")

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


    ax.arrow(RA[0], Dec[0]+((1/satzoom)*Dec[0]), (RA[-1]-RA[0]), (Dec[-1]-Dec[0]),
             head_width=0, head_length=0, 
            fc='green', ec='green', width=0.0003, 
            transform=ax.get_transform('icrs')) 
    
    ax.arrow(RA[0], Dec[0]-((1/satzoom)*Dec[0]), (RA[-1]-RA[0]), (Dec[-1]-Dec[0]), 
             head_width=0, head_length=0, 
            fc='green', ec='green', width=0.0003, 
            transform=ax.get_transform('icrs')) 
    
    for i in range(0,len(data)):
        if R_Flux[i] != 0:
            ax.plot(RA[i],Dec[i], marker=".", transform=ax.get_transform('icrs'))


    
    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='white', ls='dotted')
    plt.xlabel(r'RA')
    plt.ylabel(r'Dec')
    plt.savefig("First_Observing_Mode.png")
    return fig, ax    



Landoltplot(image1,scale=0.5, colorbar=True,wcsplot=wcs_landolt, vmin=-4.677, vmax=10)

print("Satellite tracking done!\n")

print ("Secondary Tracking Mode Starting")


def optimize_satellite_points(ra, dec, min_distance=0.001):
    """Select optimal points for querying based on minimum angular separation"""
    points = list(zip(ra, dec))
    selected = [points[0]]  # Always include first point
    
    for point in points[1:]:
        last_selected = selected[-1]
        # Calculate angular separation
        separation = np.sqrt((point[0] - last_selected[0])**2 + 
                           (point[1] - last_selected[1])**2)
        if separation >= min_distance:
            selected.append(point)
    
    return np.array(selected)

def create_tracking_composite(satellite_data, wcs_base, database='CDS/P/2MASS/K'):
    """
    Creates a composite image from multiple hips2fits queries based on tracking mode.
    """
    # Initialize composite image array
    composite = np.zeros((1024, 1024))
    
    # Optimize points for querying
    points = optimize_satellite_points(
        satellite_data['RA'], 
        satellite_data['Dec'],
        min_distance=abs(wcs_base.wcs.cdelt[0]) * 2  # Use twice the pixel scale as minimum separation
    )
    
    print(f"\nGenerating tracking composite image\nThis may take a while...")
    
    
    # Progress bar setup
    bar_width = 50
    total_steps = len(points)
    print("Progress: [" + " " * bar_width + "] 0%", end='\r')
    
    # Preallocate memory for image data
    current_image = np.zeros((1024, 1024))
    
    for i, (center_ra, center_dec) in enumerate(points):
        # Update progress bar
        progress = int((i + 1) * 100 / total_steps)
        filled_width = int(bar_width * (i + 1) / total_steps)
        bar = "=" * filled_width + " " * (bar_width - filled_width)
        print(f"Progress: [{bar}] {progress}%", end='\r')
        
        # Create WCS for this position
        w_header = {
            'WCSAXES': 2,
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
            'CRPIX1': 512,
            'CRPIX2': 512,
            'CRVAL1': center_ra,
            'CRVAL2': center_dec,
            'CDELT1': wcs_base.wcs.cdelt[0],
            'CDELT2': wcs_base.wcs.cdelt[1],
            'NAXIS1': 1024,
            'NAXIS2': 1024,
            'CUNIT1': 'deg',
            'CUNIT2': 'deg'
        }
        
        w = astropy_wcs.WCS(w_header)
        
        try:
            # Query hips2fits
            result = hips2fits.query_with_wcs(
                hips=database,
                wcs=w,
                get_query_payload=False,
            )
            
            # Get image data efficiently
            if isinstance(result, fits.HDUList):
                current_image = result[0].data
                result.close()
            else:
                with fits.open(result) as hdul:
                    current_image = hdul[0].data.copy()
            
            # Normalize using vectorized operations
            current_image = np.nan_to_num(current_image, nan=0.0)
            max_val = np.max(current_image)
            if max_val > 0:
                current_image /= max_val
            
            
            # For satellite tracking, add entire stellar field
            composite += current_image
            
            # Clear memory
            current_image.fill(0)
            
        except Exception as e:
            print(f"\nWarning: Failed to process position {i}: {e}")
            continue
    
    print("\nComposite image generation complete!     ")
    return composite


def plot_composite(composite, wcs, satellite_data):
    """Plot the composite image with appropriate scaling and labels"""
    fig = plt.figure(figsize=(15, 15))
    ax = plt.subplot(projection=wcs)
    

   
    scaled_composite = np.log1p(composite)  # log scaling to handle bright points
    
    # Plot with different color schemes based on mode
    im = ax.imshow(scaled_composite, origin='lower', cmap='inferno')
    # Mark satellite position
    center_idx = len(satellite_data['RA']) // 2
    ax.plot(satellite_data['RA'][center_idx], satellite_data['Dec'][center_idx],
            'r*', markersize=10, transform=ax.get_transform('icrs'))
    
    # Add coordinate grid
    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='white', ls='dotted')
    
    plt.colorbar(im, ax=ax, label='Log(intensity)')
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title(f'Second Observing Mode')
    
    return fig, ax


satellite_data = {
    'RA': RA,
    'Dec': Dec,
    'R_Flux': R_Flux
}

# Generate Image

composite = create_tracking_composite(satellite_data, wcs_landolt)
fig, ax = plot_composite(composite, wcs_landolt, satellite_data)
plt.savefig(f'Second_Observing_mode.png')
plt.close()

print("Secondary Tracking mode finished!")