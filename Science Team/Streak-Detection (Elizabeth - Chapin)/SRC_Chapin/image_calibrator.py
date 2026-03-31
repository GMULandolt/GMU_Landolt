import numpy as np
import sys
import os
from astropy.io import fits
import logging
import json

# Configure logging at module level
def setup_logging(config_file="config.json"):
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
        logging_level = config.get("logging_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, logging_level, logging.INFO),
            format='%(levelname)s: %(message)s'
        )
    except FileNotFoundError:
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Config file '{config_file}' not found. Using default INFO level.")

# Call at startup
setup_logging()

def load_fits_image(file_path, type):
    try:
        with fits.open(file_path) as hdul:
            image = hdul[1].data.astype(np.float32)
            fits_headers = hdul[1].header   
    except:
        with fits.open(file_path) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    if hdu.data.ndim == 3:
                        image = hdu.data[0].astype(np.float32) 
                        fits_headers = hdu.header
                    else:
                        image = hdu.data
                        fits_headers = hdu.header
    
    date_obs = (fits_headers.get("DATE-OBS") or
                fits_headers.get("DATE") or
                fits_headers.get("FRAME"))
    logging.info(f"Loaded {type} dated: {date_obs}")

    save_fits_image(image, f"./loaded_{type}_master.fits", headers=fits_headers)

    return image

def save_fits_image(image, output_path, headers=None):
    """Save a numpy array as a FITS image file."""
    hdu = fits.PrimaryHDU(image.astype(np.float32))
    if headers:
        hdu.header.update(headers)
    hdu.writeto(output_path, overwrite=True)
    logging.info(f"Saved FITS image to {output_path}")
    
def subtract_dark(image, mdark):
    if image.shape != mdark.shape:
        logging.error(f"Image and master dark have different dimensions: {image.shape} vs {mdark.shape}")
        sys.exit(1)
    calibrated_image = image - mdark

    save_fits_image(calibrated_image, "./dark_subtracted_image.fits")
    
    return calibrated_image

def divide_flat(image, mflat):  
    if image.shape != mflat.shape:
        logging.error(f"Image and master flat have different dimensions: {image.shape} vs {mflat.shape}")
        sys.exit(1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        calibrated_image = np.true_divide(image, mflat)
        calibrated_image[~np.isfinite(calibrated_image)] = 0  # set inf and NaN to 0

        save_fits_image(calibrated_image, "./flat_divided_image.fits")

    return calibrated_image

def calibrate_image(image, config_file="config.json"):
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
        cailbration_dir = config.get("calibration_dir", "calibration_frames")
        mdark_filepath = cailbration_dir + "/mdark.fits"
        mflat_filepath = cailbration_dir + "/mflat.fits"
        mdark = load_fits_image(mdark_filepath, type='dark')
        mflat = load_fits_image(mflat_filepath, type='flat')
        calibrated_image = subtract_dark(image, mdark)
        calibrated_image = divide_flat(calibrated_image, mflat)
        logging.info("Image calibration completed successfully.")
        return calibrated_image

    except FileNotFoundError:
        logging.error(f"Configuration file or calibration masters not found.")
        sys.exit(1)

if __name__ == "__main__":
    test_image_path = os.path.join(os.path.dirname(__file__), "testing_shit/test_image.fits")
    image = load_fits_image(test_image_path, type='test') 
    calibrated_image = calibrate_image(image)