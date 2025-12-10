import os
import cv2
import numpy as np
from astropy.io import fits
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import logging
from PIL import Image 
from PIL import PngImagePlugin
from datetime import datetime
from satprocessing import build_satellite, get_ra_dec_rates

def load_fits_image(file_path):
    from astropy.io import fits
    import numpy as np

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


    exposure_time = (fits_headers.get("EXPTIME") or
                     fits_headers.get("EXPOSURE") or
                     fits_headers.get("ACT") or
                     fits_headers.get("KCT"))
    telescope = fits_headers.get("TELESCOP")
    focal_length = fits_headers.get("FOCALLEN")
    camera = fits_headers.get("INSTRUME")
    lat = fits_headers.get("SITELAT")
    lon = fits_headers.get("SITELONG")
    elevation = fits_headers.get("SITEELEV")
    date_obs = (fits_headers.get("DATE-OBS") or
                fits_headers.get("DATE") or
                fits_headers.get("FRAME"))
    
    if lat is None or lon is None or elevation is None:
        logging.error("Missing observatory location in FITS headers (SITELAT, SITELONG, SITEELEV).")
    if camera is None:
        logging.error("Missing camera model in FITS headers (INSTRUME).")
        return image, exposure_time, None, lat, lon, elevation, date_obs
    if fits_headers.get("XBINNING") is not None:
        binning = fits_headers.get("XBINNING")
        logging.info(f"Image binning: {binning}x")
    else: 
        binning = 1
        logging.info("No binning information found, assuming 1x1")
    logging.info(f"Image shape: {image.shape}, dtype: {image.dtype}, min: {np.min(image)}, max: {np.max(image)}")
    logging.info(f"FITS headers: EXPTIME={exposure_time}, TELESCOP={telescope}, FOCALLEN={focal_length}, INSTRUME={camera}, LATITUDE={lat}, LONGITUD={lon}, ELEVATION={elevation}, DATE-OBS={date_obs}")
    
    if camera == "QHY268M" or camera == "QHY600":
        pixel_size = 3.76 * binning  # microns
        pixel_scale = 206.265 * pixel_size / focal_length  # arcsec/pixel
        print(f"Pixel scale: {pixel_scale:.2f} arcsec/pixel")
    else: 
        pixel_size = fits_headers.get("XPIXSZ") * binning  # microns
        pixel_scale = 206.265 * pixel_size / focal_length  # arcsec/pixel
        print(f"Pixel scale: {pixel_scale:.2f} arcsec/pixel")

    return image, exposure_time, pixel_scale, lat, lon, elevation, date_obs

def estimate_steak_length(NORAD_id, exposure_time, pixel_scale, lat, lon, elevation, date_obs):
    sat, OBSERVER = build_satellite(NORAD_id, lat, lon, elevation)
    ra_rate, dec_rate, dec_deg = get_ra_dec_rates(sat, OBSERVER, date_obs)
    angular_velocity = np.hypot(dec_rate, ra_rate * np.cos((dec_deg)))
    logging.info(f"Estimated angular velocity: {angular_velocity:.2f} arcsec/sec")
    max_streak_length = ((angular_velocity * exposure_time) / pixel_scale)
    minLinelength = 0.7 * max_streak_length 
    logging.info(f"Estimated streak length: {minLinelength:.2f} pixels")
    return minLinelength

def detect_streaks(image, minLinelength, enabled_filters, background_dectection_method) -> list:

    p_low, p_high = np.percentile(image, (2, 98))
    data_clipped = np.clip(image, p_low, p_high)

    # Normalize to 0–255
    norm_data = (data_clipped - p_low) / (p_high - p_low) * 255
    save_image_with_metadata("normalized_image.png", norm_data, 
                             tags={"stage": "normalized", "min": str(np.min(image)), "max": str(np.max(image))})
    
    image_display = np.uint8(norm_data)
    print(f"Image dtype: {image.dtype}, min: {np.min(image)}, max: {np.max(image)}")
    save_image_with_metadata("image_display.png", image_display, 
                             tags={"stage": "normalized", "min": str(np.min(image)), "max": str(np.max(image))})


    if background_dectection_method.get("simple_median", True):
        binary = simple_median(image)
        background_dectection_method = "simple_median"
    elif background_dectection_method.get("Guassian_blur", True):  
        binary = gaussian_binarize(image)
        background_dectection_method = "Guassian_blur"
    elif background_dectection_method.get("doublepass_median_to_guassian_blur", True):
        binary = double_threshold(image)
        background_dectection_method = "doublepass_median_to_guassian_blur"
    else:
        logging.error("No valid background detection method selected, defaulting to Gaussian blur")
        binary = gaussian_binarize(image)
        background_dectection_method = "Guassian_blur"

    # Use Hough Line Transform to detect streaks
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=60,
                            minLineLength=minLinelength, maxLineGap=5)
    if lines is None:
        return logging.error("No lines detected"), image_display
    print(f"Detected {len(lines)} lines") 

    enabled_filters_tags = []

    if enabled_filters.get("midpoint_filter", True):
        filtered_lines = midpoint_filter_close_lines(lines, min_distance=10)
        logging.info(f"{len(filtered_lines)} lines after midpoint filtering")
        enabled_filters_tags.append("midpoint_filter")
    
    if enabled_filters.get("line_angle", True):
        filtered_lines = line_angle_filter(filtered_lines, min_angle_diff=10)
        logging.info(f"{len(filtered_lines)} lines after filtering by angle")
        enabled_filters_tags.append("line_angle")

    if enabled_filters.get("colinear_filter", True):
        filtered_lines = add_colinear_segments(filtered_lines)
        print(f"{len(filtered_lines)} lines after merging collinear segments")
        enabled_filters_tags.append("colinear_filter")

    if enabled_filters.get("endpoint_filer", True):
        filtered_lines = endpoint_filer(filtered_lines, min_distance=10)
        logging.info(f"{len(filtered_lines)} lines after endpoint filtering")
        enabled_filters_tags.append("endpoint_filer")
    
    if enabled_filters.get("length_filter", True):
        filtered_lines = line_length_filter(filtered_lines)
        logging.info(f"{len(filtered_lines)} lines after length filtering")
        enabled_filters_tags.append("length_filter")

    tags = ["background_method=" + background_dectection_method,
            "enabled_filters=" + ",".join(enabled_filters_tags), "processed at" + str(datetime.now())]
    labels = [f"{i+1}" for i in range(len(filtered_lines))]

    output = cv2.cvtColor(image_display, cv2.COLOR_GRAY2BGR)  # make color image to draw on
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    filtered_lines = np.array(filtered_lines)
    if filtered_lines is not None:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    

    return filtered_lines, image_display, labels, tags

def endpoint_filer (lines, min_distance=10):
    if lines is None:
        return []
    endpoint_filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        too_close = False
        for fline in endpoint_filtered_lines:
            fx1, fy1, fx2, fy2 = fline[0]
            distances = [
                np.hypot(x1 - fx1, y1 - fy1),
                np.hypot(x1 - fx2, y1 - fy2),
                np.hypot(x2 - fx1, y2 - fy1),
                np.hypot(x2 - fx2, y2 - fy2)
            ]
            if any(d < min_distance for d in distances):
                too_close = True
                break
        if not too_close:
            endpoint_filtered_lines.append(line)
    return endpoint_filtered_lines

def line_length_filter(lines) -> list:
    if lines is None:
        return []
    lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        lengths.append(length)
    max_length = np.max(lengths)
    length_filtered_lines = []
    for line, length in zip(lines, lengths):
        if length >= 0.8 * max_length:
            length_filtered_lines.append(line)
    return length_filtered_lines

def add_colinear_segments(filtered_lines) -> list:
    if filtered_lines is None or len(filtered_lines) == 1:
        return filtered_lines
    else: 
        for i in range(len(filtered_lines)):
            line1 = filtered_lines[i]
            line2 = filtered_lines[i+1] if i + 1 < len(filtered_lines) else None
            if line2 is None:
                break
            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]
            A = np.array([x1, y1])
            B = np.array([x2, y2])
            C = np.array([x3, y3])
            AB = B - A
            AC = C - A

            orientation = np.cross(AB, AC)
            if abs(orientation) < 1:  # collinear
                new_line = np.array([[min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4)]])
                filtered_lines.append(new_line)
                filtered_lines.pop(i)
                filtered_lines.pop(i+1)
                return add_colinear_segments(filtered_lines)  # re-run to catch further merges
        return filtered_lines

def line_angle_filter(lines, min_angle_diff) -> list:
    if lines is None:
        return []
    angle_filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        too_far = False
        for fline in angle_filtered_lines:
            fx1, fy1, fx2, fy2 = fline[0]
            fangle = np.degrees(np.arctan2(fy2 - fy1, fx2 - fx1))
            if abs(angle - fangle) > min_angle_diff:  
                too_far = True
                break
        if not too_far:
            angle_filtered_lines.append(line)
    return angle_filtered_lines
            
def midpoint_filter_close_lines(lines, min_distance=10):
    if lines is None:
        return []
    midpoint_filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        too_close = False
        for fline in midpoint_filtered_lines:
            fx1, fy1, fx2, fy2 = fline[0]
            fmid_x, fmid_y = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            distance = np.hypot(mid_x - fmid_x, mid_y - fmid_y)
            if distance < min_distance:
                too_close = True
                break
        if not too_close:
            midpoint_filtered_lines.append(line)
    return midpoint_filtered_lines

def gaussian_binarize(image):
    # Percentile clip to tame hot pixels & bright stars
    p1, p99 = np.percentile(image, (2, 99.8))
    clipped = np.clip(image, p1, p99)
    plt.imsave("clipped.png", clipped, cmap='gray')

    # Remove low-frequency background (top-hat): blur, subtract
    background = cv2.GaussianBlur(clipped, (51, 51), 0)

    highpass = clipped - background

    save_image_with_metadata("highpass_image.png", highpass, 
                             tags={"stage": "highpass", "method": "gaussian_blur"})
    
    save_image_with_metadata("background.png", background, 
                             tags={"stage": "background", "method": "gaussian_blur"})
    
    

    # Blur slightly to improve SNR for thresholding
    hp_blur = cv2.GaussianBlur(highpass, (5, 5), 0)

    # 2) Gradient
    gy, gx = np.gradient(highpass)
    grad_mag = np.hypot(gx, gy)

    # 3) Visualize
    plt.figure(figsize=(6, 6))
    save_image_with_metadata("image_display1.png", grad_mag,
                             tags={"stage": "gradient_magnitude", "pass": "pre_threshold"},
                             dir="output")
    plt.title("Gradient Magnitude (streaks should be bright)")


    # Robust sigma via MAD
    med = np.median(hp_blur)
    mad = np.median(np.abs(hp_blur - med)) + 1e-6
    sigma = 1.4826 * mad

    # Try a small ladder of k values for faint streaks
    for k in (3.0, 2.5, 2.0, 1.5, 1.2):
        print(f"Trying k={k} for thresholding")
        thr = med + k * sigma
        binary = (hp_blur >= thr).astype(np.uint8) * 255
        # Close tiny gaps
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        
        save_image_with_metadata("binary.png", binary,
                                 tags={"stage": "binarized", "k": str(k), "method": "gaussian_blur"})

        if np.count_nonzero(binary) > 50:
               # enough pixels to attempt Hough
            return binary


    return binary  # best effort

def simple_median(image):
    median = np.median(image)
    stddev = np.std(image)
    threshold1 = median + (2 * stddev)
    binary = (image >= threshold1).astype(np.uint8) * 255  
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    gy1, gx1 = np.gradient(binary)
    grad_mag1 = np.hypot(gx1, gy1)

    save_image_with_metadata("binary_grad_map.png", grad_mag1,
                                 tags={"stage": "gradient_magnitude", "pass": "post_threshold"},
                                 dir="output")
    return binary

def double_threshold(image):
    median = np.median(image)
    stddev = np.std(image)
    threshold1 = median + (2 * stddev)
    masked = image.copy()
    hot_mask = masked > threshold1
    masked[hot_mask] = np.nan

    nan_mask = np.isnan(masked)
    masked_filled = masked.copy()
    masked_filled[nan_mask] = median
    
    background = cv2.GaussianBlur(masked_filled, (51, 51), 0)

    save_image_with_metadata("background.png", background,
                             tags={"stage": "background", "method": "double_threshold"})
    
    highpass = image - background

    save_image_with_metadata("highpass.png", highpass,
                             tags={"stage": "highpass", "method": "double_threshold"})
    
    median2 = np.nanmedian(highpass)
    stddev2 = np.nanstd(highpass)
    threshold2 = median2 + (2 * stddev2)
    binary = (highpass >= threshold2).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    
    save_image_with_metadata("binary.png", binary,
                             tags={"stage": "binarized", "method": "double_threshold"})
    return binary  

def save_streaks_to_csv(lines, labels, filename="streaks.csv"):
    import csv
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label", "x1", "y1", "x2", "y2", "midpoint_x", "midpoint_y"])
        for line, label in zip(lines, labels):
            x1, y1, x2, y2 = line[0]
            minpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2
            writer.writerow([label, x1, y1, x2, y2, minpoint_x, midpoint_y])

def save_image_with_metadata(filename, image, tags=None, dir="output"): 
    os.makedirs(dir, exist_ok=True)
    filepath = os.path.join(dir, filename)
    
    # Convert image to PIL if it's a numpy array
    if isinstance(image, np.ndarray):
        # Convert grayscale or BGR to RGB for PNG
        if len(image.shape) == 2:  # grayscale
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:  # BGR → RGB
           pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:        
        pil_image = image
   
   # Create PngInfo object and add metadata
    metadata = PngImagePlugin.PngInfo()
    for i, tag in enumerate(tags or []):
       key = f"tag_{i+1}"
       metadata.add_text(key, str(tag))
   
    pil_image.save(filepath, pnginfo=metadata)
    logging.info(f"Saved image with metadata to {filepath}")

def draw_bounding_boxes(image_display, lines, labels):
    if lines is not None:
        for line, label in zip(lines, labels):
            x1, y1, x2, y2 = line[0]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            cv2.rectangle(image_display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image_display, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    
    return image_display

def main(config_file="config.json"):
    
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
        logging_level = config.get("logging_level")
        logging.basicConfig(level=getattr(logging, logging_level, logging.INFO))

        images_dir = config.get("images_dir", "images")
        files = [f for f in os.listdir(images_dir) if f.lower().endswith('.fits') or f.lower().endswith('.fit')]
        if not files:
            logging.error("No FITS files found in the images directory.")
            return
        print("Available FITS files:")
        for idx, file in enumerate(files):
            print(f"{idx + 1}: {file}")
        choice = int(input(f"Pick a file by number:  ")) - 1
        selected_file = files[choice]
        logging.info(f"Selected file: {selected_file}")
        file_path: str = os.path.join(images_dir, selected_file)

        image, exposure_time, pixel_scale, lat, lon, elevation, date_obs = load_fits_image(file_path)

        estimated_streak_length_enabled = config.get("estimated_streak_length_enabled")
        if estimated_streak_length_enabled == True:
            NORAD_id = int(input("Please provide the NORAD ID of the satellite to detect streaks: ").strip())
            if lat is None or lon is None or elevation is None or NORAD_id is None:
                logging.error("Incomplete or incompatible fits header schema cannot estimate streak length.")
            else: 
                minlinelength: float = estimate_steak_length(NORAD_id, exposure_time, pixel_scale, lat, lon, elevation, date_obs)
        else: 
            if lat is None or lon is None or elevation is None:
                logging.warning("Incomplete or incompatible fits header schema.")
                logging.info("Estimated streak length disabled, deafulting to general minLinelength")
            minlinelength: int = config.get("default_minlinelength")
        enabled_line_filters = config.get("enabled_line_filters", {})
        background_detection_method = config.get("background_detection_method", {})
        lines, image_display, labels, tags = detect_streaks(image, minlinelength, enabled_line_filters, background_detection_method)
        boxed_image = draw_bounding_boxes(image_display, lines, labels)
        save_streaks_to_csv(lines, labels)
        save_image_with_metadata("detected_streaks.png", boxed_image, tags, dir="output")

    except FileNotFoundError:
        logging.error(f"Config file {config_file} not found please verify file path to continue.")
        exit(1)
    except json.JSONDecodeError:
        logging.error(f"Config file {config_file} is not a valid JSON file please verify file content to continue.")
        exit(1)
                
if __name__ == "__main__":
    main(config_file="config.json")


'''

        ("images/8b5f6576-fb72-45c7-b0e2-83923ff9b233.fit") 
        NORAD_id = 32265 

        ("images/9f2022f0-264a-4701-adf3-1495f64f67d1.fit")
        NORAD_id = 23775

        ("images/Intelsat-40_G200_05s.fits")
        NORAD_id = 56174

'''
