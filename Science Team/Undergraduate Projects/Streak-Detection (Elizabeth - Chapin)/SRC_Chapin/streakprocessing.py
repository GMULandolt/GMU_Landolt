import os
import cv2
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import logging
from satprocessing import build_satellite, get_ra_dec_rates

logging.basicConfig(level=logging.INFO)

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
    print(f"Exposure time: {exposure_time}, Telescope: {telescope}, Focal Length: {focal_length}, Camera: {camera}")

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
    minLinelength = 0.8 * max_streak_length 
    logging.info(f"Estimated streak length: {minLinelength:.2f} pixels")
    return minLinelength

def detect_streaks(image, minLinelength):

    p_low, p_high = np.percentile(image, (2, 98))
    data_clipped = np.clip(image, p_low, p_high)

    # Normalize to 0–255
    norm_data = (data_clipped - p_low) / (p_high - p_low) * 255
    plt.imsave("normalized_image.png", norm_data, cmap='gray')
    image_display = np.uint8(norm_data)
    print(f"Image dtype: {image.dtype}, min: {np.min(image)}, max: {np.max(image)}")
    plt.imsave("image display.png", image_display, cmap='gray')
    plt.title("Pre-blur image")
    plt.show()
    

    binary = robust_binarize(image)

    plt.imshow(binary, cmap='gray')
    plt.title("Binarized Image")    
    plt.show()
    plt.imsave("binarized_image.png", binary, cmap='gray')

    # Use Hough Line Transform to detect streaks
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=60,
                            minLineLength=minLinelength, maxLineGap=5)
    if lines is None:
        return logging.error("No lines detected"), image_display
    print(f"Detected {len(lines)} lines") 

    midpoint_filtered_lines = midpoint_filter_close_lines(lines, min_distance=10)
    print(f"{len(midpoint_filtered_lines)} lines after filtering close ones")

    angle_filtered_lines = line_angle_filter(midpoint_filtered_lines, min_angle_diff=10)
    print(f"{len(angle_filtered_lines)} lines after filtering by angle")

    filtered_lines = add_colinear_segments(angle_filtered_lines)
    print(f"{len(filtered_lines)} lines after merging collinear segments")

    filtered_lines = endpoint_filer(filtered_lines, min_distance=10)
    print(f"{len(filtered_lines)} lines after endpoint filtering")
    
    labels = [f"{i+1}" for i in range(len(filtered_lines))]

    output = cv2.cvtColor(image_display, cv2.COLOR_GRAY2BGR)  # make color image to draw on
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    filtered_lines = np.array(filtered_lines)
    if filtered_lines is not None:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.imshow(output)
    plt.title("Detected Streak Lines")
    plt.axis('off')
    plt.show()
    return filtered_lines, image_display, labels

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

def robust_binarize(image):
    # Percentile clip to tame hot pixels & bright stars
    p1, p99 = np.percentile(image, (2, 99.8))
    clipped = np.clip(image, p1, p99)
    plt.imsave("clipped.png", clipped, cmap='gray')

    # Remove low-frequency background (top-hat): blur, subtract
    background = cv2.GaussianBlur(clipped, (51, 51), 0)
    highpass = clipped - background
    plt.imsave("highpass_image.png", highpass, cmap='gray')
    plt.imsave("background.png", background, cmap='gray')

    # Blur slightly to improve SNR for thresholding
    hp_blur = cv2.GaussianBlur(highpass, (5, 5), 0)
    plt.imsave("hp_blur_image.png", hp_blur, cmap='gray')

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
        if np.count_nonzero(binary) > 50:   # enough pixels to attempt Hough
            return binary

    return binary  # best effort

def save_streaks_to_csv(lines, labels, filename="streaks.csv"):
    import csv
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label", "x1", "y1", "x2", "y2"])
        for line, label in zip(lines, labels):
            x1, y1, x2, y2 = line[0]
            writer.writerow([label, x1, y1, x2, y2])

def draw_bounding_boxes(image_display, lines, labels):
    if lines is not None:
        for line, label in zip(lines, labels):
            x1, y1, x2, y2 = line[0]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            cv2.rectangle(image_display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image_display, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    
    return image_display

def main():
    #testing UI
    images_dir = "images"
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
    file_path = os.path.join(images_dir, selected_file)
    print ("Please provide the NORAD ID of the satellite to detect streaks: ")
    NORAD_id = int(input("NORAD ID: ").strip())

    image, exposure_time, pixel_scale, lat, lon, elevation, date_obs = load_fits_image(file_path)
    if lat is None or lon is None or elevation is None:
        logging.info("Proceeding without estimated streak length")
        minLinelength = 25  # default minimum length
        lines, image_display, labels = detect_streaks(image, minLinelength)
        boxed_image = draw_bounding_boxes(image_display, lines, labels)
        save_streaks_to_csv(lines, labels)
    else: 
        minLinelength = estimate_steak_length(NORAD_id, exposure_time, pixel_scale, lat, lon, elevation, date_obs)
        lines, image_display, labels = detect_streaks(image, minLinelength)
        boxed_image = draw_bounding_boxes(image_display, lines, labels)
        save_streaks_to_csv(lines, labels)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(boxed_image)
    plt.title("Detected Streaks")
    plt.axis("off")
    plt.show()
    plt.imsave("detected_streaks.png", boxed_image, cmap='gray')

if __name__ == "__main__":
    main()


'''

        ("images/8b5f6576-fb72-45c7-b0e2-83923ff9b233.fit") 
        NORAD_id = 32265 

        ("images/9f2022f0-264a-4701-adf3-1495f64f67d1.fit")
        NORAD_id = 23775

        ("images/Intelsat-40_G200_05s.fits")
        NORAD_id = 56174

'''
