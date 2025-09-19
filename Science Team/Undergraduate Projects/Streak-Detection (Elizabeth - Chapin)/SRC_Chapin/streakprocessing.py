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
                    image = hdu.data
                    break
            fits_headers = hdul[0].header
            if image.ndim == 3:
                image = image[0].astype(np.float32)

    exposure_time = fits_headers.get("EXPTIME")
    telescope = fits_headers.get("TELESCOP")
    focal_length = fits_headers.get("FOCALLEN")
    camera = fits_headers.get("INSTRUME")
    print(f"Exposure time: {exposure_time}, Telescope: {telescope}, Focal Length: {focal_length}, Camera: {camera}")
    if camera == "QHY268M" or camera == "QHY600":
        pixel_size = 3.76  # microns
        pixel_scale = 206.265 * pixel_size / focal_length  # arcsec/pixel
        print(f"Pixel scale: {pixel_scale:.2f} arcsec/pixel")
    else: raise ValueError(f"Unknown camera model cannot determine pixel scale, please update known camera database: {camera}")
    
    return image, exposure_time, pixel_scale

def estimate_steak_length(NORAD_id, exposure_time, pixel_scale):
    sat = build_satellite(NORAD_id)
    ra_rate, dec_rate = get_ra_dec_rates(sat)
    angular_velocity = np.hypot(ra_rate, dec_rate) * 3600.0  # arcsec/sec
    logging.info(f"Estimated angular velocity: {angular_velocity:.2f} arcsec/sec")
    max_streak_length = ((angular_velocity * exposure_time) / pixel_scale) * 1.1 # 10% safty margin
    minLinelength = 0.9 * max_streak_length 
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
        raise ValueError("No streaks detected.")
    print(f"Detected {len(lines)} lines") 

    filtered_lines = filter_close_lines(lines, min_distance=10)
    print(f"{len(filtered_lines)} lines after filtering close ones")

        # Draw detected lines on the original image for visualization

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
    return filtered_lines, image_display

def filter_close_lines(lines, min_distance=10):
    if lines is None:
        return []
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        too_close = False
        for fline in filtered_lines:
            fx1, fy1, fx2, fy2 = fline[0]
            fmid_x, fmid_y = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            distance = np.hypot(mid_x - fmid_x, mid_y - fmid_y)
            if distance < min_distance:
                too_close = True
                break
        if not too_close:
            filtered_lines.append(line)
    return filtered_lines

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

def draw_bounding_boxes(image_display, lines):

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            cv2.rectangle(image_display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    return image_display

def main():
    print("Sideral (1), Rate Tracked (2), Testing (3)")
    mode = input("Enter mode (1, 2 or 3): ").strip()
    if mode == '1':
        image, exposure_time, pixel_scale = load_fits_image("images/8b5f6576-fb72-45c7-b0e2-83923ff9b233.fit") 
        NORAD_id = 32265 
    elif mode == '2':
        image, exposure_time, pixel_scale = load_fits_image("images/9f2022f0-264a-4701-adf3-1495f64f67d1.fit")
        NORAD_id = 23775
    elif mode == '3':
        image, exposure_time, pixel_scale = load_fits_image("images/Intelsat-40_G200_05s.fits")
        NORAD_id = 56174
    else:
        print("Invalid mode selected. Please enter 1 or 2.")
        return
    minLinelength = estimate_steak_length(NORAD_id, exposure_time, pixel_scale)
    lines, image_display = detect_streaks(image, minLinelength)
    boxed_image = draw_bounding_boxes(image_display, lines)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(boxed_image)
    plt.title("Detected Streaks")
    plt.axis("off")
    plt.show()
    plt.imsave("detected_streaks.png", boxed_image)

if __name__ == "__main__":
    main()
