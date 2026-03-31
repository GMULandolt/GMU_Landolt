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
from image_calibrator import calibrate_image

def load_fits_image(file_path):
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

    # collect snapshots for visualization: list of (stage_name, lines)
    stages = []
    stages.append(("detected", lines))
    current_lines = lines

    if enabled_filters.get("midpoint_filter", True):
        new_lines = midpoint_filter_close_lines(current_lines, min_distance=10)
        logging.info(f"{len(new_lines)} lines after midpoint filtering")
        enabled_filters_tags.append("midpoint_filter")
        stages.append(("midpoint_filter", new_lines))
        current_lines = new_lines

    if enabled_filters.get("line_angle", True):
        new_lines = line_angle_filter(current_lines, min_angle_diff=10)
        logging.info(f"{len(new_lines)} lines after filtering by angle")
        enabled_filters_tags.append("line_angle")
        stages.append(("line_angle", new_lines))
        current_lines = new_lines

    if enabled_filters.get("colinear_filter", True):
        new_lines = add_colinear_segments(current_lines)
        print(f"{len(new_lines)} lines after merging collinear segments")
        enabled_filters_tags.append("colinear_filter")
        stages.append(("colinear_filter", new_lines))
        current_lines = new_lines

    if enabled_filters.get("endpoint_filer", True):
        new_lines = endpoint_filer(current_lines, min_distance=10)
        logging.info(f"{len(new_lines)} lines after endpoint filtering")
        enabled_filters_tags.append("endpoint_filer")
        stages.append(("endpoint_filer", new_lines))
        current_lines = new_lines

    if enabled_filters.get("length_filter", True):
        new_lines = line_length_filter(current_lines)
        logging.info(f"{len(new_lines)} lines after length filtering")
        enabled_filters_tags.append("length_filter")
        stages.append(("length_filter", new_lines))
        current_lines = new_lines

    # final filtered lines
    filtered_lines = current_lines

    # create a visualization overlay showing evolution across stages
    try:
        annotated = draw_filter_stage_overlays(binary, stages)
        save_image_with_metadata("filter_stage_overlays.png", annotated, tags={"stage": "filter_evolution"}, dir="output")
    except Exception as e:
        logging.warning(f"Could not create filter-stage overlay: {e}")

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
    """
    Background detection using Gaussian blur high-pass filtering.
    
    This method isolates streaks by creating a high-pass filter that removes
    low-frequency background components while preserving sharp features like
    satellites. The approach:
    
    1. Clips outliers using percentiles (2% to 99.8%) to handle hot pixels and
       bright stars that would otherwise dominate the background estimation.
    2. Estimates background via Gaussian blur (51x51 kernel), capturing the
       smooth low-frequency component of the image.
    3. Subtracts background from clipped image to isolate high-frequency features
       (streaks, stars, cosmic rays).
    4. Applies gradient magnitude to enhance edges and improve SNR.
    5. Uses robust sigma estimation (MAD-based) to adaptively threshold across
       different k-values, finding the best threshold that captures streak
       signal without excessive noise.
    
    This method works well for images with smooth background gradients and is
    computationally efficient, though it may struggle with highly variable
    backgrounds or crowded star fields.
    """
    # Percentile clip to tame hot pixels & bright stars
    p1, p99 = np.percentile(image, (2, 99.8))
    clipped = np.clip(image, p1, p99)

    # Remove low-frequency background (top-hat): blur, subtract
    background = cv2.GaussianBlur(clipped, (51, 51), 0)

    highpass = clipped - background

    save_image_with_metadata("highpass.png", highpass, 
                             tags={"stage": "highpass", "method": "gaussian_blur"},
                             normalize=True)
    
    save_image_with_metadata("background.png", background, 
                             tags={"stage": "background", "method": "gaussian_blur"},
                             normalize=True)
    
    

    # Blur slightly to improve SNR for thresholding
    hp_blur = cv2.GaussianBlur(highpass, (5, 5), 0)

    gy, gx = np.gradient(highpass)
    grad_mag = np.hypot(gx, gy)

    save_image_with_metadata("image_display1.png", grad_mag,
                             tags={"stage": "gradient_magnitude", "pass": "pre_threshold"},
                             dir="output", normalize=True)
    # Gradient magnitude visualization


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
    """
    Background detection using simple median and standard deviation thresholding.
    
    This is the simplest background detection approach, assuming the image has
    a relatively uniform background with streaks as outliers. The method:
    
    1. Computes global median and standard deviation of the entire image.
    2. Sets threshold at median + 2*stddev, which corresponds approximately to
       the 95th percentile for normally distributed data.
    3. Creates binary mask where pixels above threshold (likely streaks/stars)
       are marked as foreground.
    4. Applies morphological closing (3x3 kernel) to fill small holes in detected
       regions, helping maintain streak continuity.
    5. Computes gradient magnitude for visualization.
    
    Assumptions & Limitations:
    - Assumes image statistics are dominated by background pixels (valid if
      foreground objects occupy <5% of image area).
    - Fails when background is non-uniform or has strong gradients.
    - Doesn't account for local variations in noise level.
    - Works well for simple, clean images but can miss faint streaks or over-detect
      in noisy regions.
    
    Best used as a quick baseline or for well-behaved astronomical data with
    relatively constant background.
    """
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
    """
    Background detection using dual-stage thresholding with inpainting.
    
    This is the most sophisticated approach, designed to handle images with
    significant foreground contamination (stars, cosmic rays, streaks). The
    two-pass strategy:
    
    PASS 1 - Initial Masking:
    1. Identifies hot pixels/stars using threshold = median + 2*stddev.
    2. Masks these high-intensity pixels to prevent them from biasing background
       estimation (sets masked regions to NaN initially).
    3. Uses Telea inpainting algorithm to intelligently fill masked regions,
       reconstructing what the background would be without foreground objects.
    4. This produces a cleaner background estimate than global median.
    
    PASS 2 - Streak Detection:
    1. Subtracts inpainted background from hot pixel mask to isolate the streak
       signal (difference between foreground and reconstructed background).
    2. Recomputes median and stddev on the refined highpass data (ignoring NaN).
    3. Applies second threshold (median2 + 2*stddev2) for final binarization.
    4. Applies morphological closing (5x5 kernel) to maintain streak continuity.
    
    Advantages:
    - Handles non-uniform backgrounds better than simple_median.
    - Reduces false positives from scattered bright pixels.
    - Two-stage approach refines signal and improves SNR significantly.
    
    Disadvantages:
    - Computationally more expensive (inpainting is iterative).
    - Inpainting quality depends on kernel size and algorithm choice.
    - Can over-smooth if inpainting erases large streaks or artifacts.
    
    Best for: Complex scenes with variable backgrounds, crowded fields, or
    images where stars and cosmic rays contaminate the background.
    """
    median = np.median(image)
    stddev = np.std(image)
    threshold1 = median + (2 * stddev)
    masked = image.copy()
    hot_mask = masked > threshold1
    masked[hot_mask] = np.nan

    nan_mask = np.isnan(masked).astype(np.uint8)
    masked_filled = np.nan_to_num(masked, nan=median)
    background = cv2.inpaint(masked_filled.astype(np.float32), nan_mask, 3, cv2.INPAINT_TELEA)

    save_image_with_metadata("background.png", background,
                             tags={"stage": "background", "method": "double_threshold"},
                             normalize=True)
    
    highpass = background - hot_mask

    save_image_with_metadata("highpass.png", highpass,
                             tags={"stage": "highpass", "method": "double_threshold"},
                             normalize=True)
    
    median2 = np.nanmedian(highpass)
    stddev2 = np.nanstd(highpass)
    threshold2 = median2 + (2 * stddev2)
    binary = (highpass >= threshold2).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    
    save_image_with_metadata("binary.png", binary,
                             tags={"stage": "binarized", "method": "double_threshold"})
    return binary  

def save_streaks_to_csv(lines, labels, filename="streaks.csv", dir="output"):
    import csv
    os.makedirs(dir, exist_ok=True)
    filepath = os.path.join(dir, filename)
    with open(filepath, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label", "x1", "y1", "x2", "y2", "midpoint_x", "midpoint_y"])
        for line, label in zip(lines, labels):
            x1, y1, x2, y2 = line[0]
            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2
            writer.writerow([label, x1, y1, x2, y2, midpoint_x, midpoint_y])

def save_image_with_metadata(filename, image, tags=None, dir="output", normalize=False): 
    """
    Save image faithfully to disk with optional metadata.
    - normalize=False: save raw data as-is (preserve full dynamic range).
    - normalize=True: rescale to 0-255 for visualization.
    Uses cv2.imwrite + PIL metadata for lossless storage.
    """
    os.makedirs(dir, exist_ok=True)
    filepath = os.path.join(dir, filename)
    
    if not isinstance(image, np.ndarray):
        logging.warning(f"Image is not numpy array, skipping save to {filepath}")
        return
    
    # Prepare image for saving
    img_to_save = image.copy()
    
    # Handle normalization if requested
    if normalize and img_to_save.dtype in [np.float32, np.float64]:
        # Normalize float to 0-255 preserving full range
        img_min, img_max = np.min(img_to_save), np.max(img_to_save)
        if img_max > img_min:
            img_to_save = ((img_to_save - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_to_save = img_to_save.astype(np.uint8)
    elif img_to_save.dtype == np.float32 or img_to_save.dtype == np.float64:
        # Save float as-is (cv2.imwrite will handle)
        img_to_save = img_to_save.astype(np.float32)
    else:
        # Already uint8 or other integer type
        pass
    
    # Handle 3-channel images (BGR for OpenCV consistency)
    if len(img_to_save.shape) == 3 and img_to_save.shape[2] == 3:
        # Already BGR, cv2.imwrite handles this correctly
        pass
    elif len(img_to_save.shape) == 2:
        # Grayscale, cv2.imwrite handles this
        pass
    
    # Save with cv2 (faithful, lossless PNG)
    success = cv2.imwrite(filepath, img_to_save)
    if not success:
        logging.error(f"Failed to save image to {filepath}")
        return
    
    # Add metadata via PIL if tags provided
    if tags:
        try:
            pil_image = Image.open(filepath)
            metadata = PngImagePlugin.PngInfo()
            for i, tag in enumerate(tags or []):
                key = f"tag_{i+1}"
                metadata.add_text(key, str(tag))
            pil_image.save(filepath, pnginfo=metadata)
        except Exception as e:
            logging.warning(f"Could not add metadata to {filepath}: {e}")
    
    logging.info(f"Saved image (dtype={image.dtype}, range=[{np.min(image):.2f}, {np.max(image):.2f}]) to {filepath}")

def draw_bounding_boxes(image_display, lines, labels):
    if lines is not None:
        for line, label in zip(lines, labels):
            x1, y1, x2, y2 = line[0]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            cv2.rectangle(image_display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image_display, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    
    return image_display

def draw_filter_stage_overlays(base_image, stages, cmap_name='tab10', thickness=2, annotate_counts=True):
    """
    Draw overlays showing line counts at each filtering stage.
    - base_image: grayscale or BGR numpy image
    - stages: list of (stage_name, lines_array) in chronological order (first is 'detected')
    Returns a BGR image with semi-transparent overlays and a legend.
    """
    import math
    # prepare base image as BGR
    out = base_image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    else:
        out = out.copy()

    h, w = out.shape[:2]

    # colormap
    cmap = plt.get_cmap(cmap_name)
    n = max(1, len(stages))

    # draw each stage with increasing prominence
    for idx, (name, lines) in enumerate(stages):
        if lines is None:
            continue
        # color from cmap (RGB float 0..1) -> BGR int 0..255
        c = cmap(idx / max(1, n - 1))[:3]
        color = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))

        # create overlay and draw lines there
        overlay = out.copy()
        for line in lines:
            # handle shape variations
            l = np.asarray(line).reshape(-1)
            if l.size < 4:
                continue
            x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
            cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        # alpha blending: earlier stages lighter, later stages stronger
        alpha = 0.35 + 0.55 * (idx / max(1, n - 1))
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    # draw legend box
    pad = 50
    legend_x, legend_y = pad, pad
    box_w = 350  # increased width for larger text
    box_h = 35 * len(stages) + 12  # increased height per entry
    # semi-transparent background for legend
    legend_bg = out.copy()
    cv2.rectangle(legend_bg, (legend_x - 4, legend_y - 4), (legend_x + box_w, legend_y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(legend_bg, 0.45, out, 0.55, 0, out)

    # draw legend entries
    for idx, (name, lines) in enumerate(stages):
        c = cmap(idx / max(1, n - 1))[:3]
        color = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
        ty = legend_y + 6 + idx * 35  # increased spacing
        # small color swatch
        cv2.rectangle(out, (legend_x, ty), (legend_x + 20, ty + 20), color, -1)  # larger swatch
        count = 0 if lines is None else (len(lines) if hasattr(lines, '__len__') else 0)
        text = f"{name}: {count}"
        cv2.putText(out, text, (legend_x + 30, ty + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)  # larger font and thickness

    # optional annotation of total at top-right
    total = 0 if len(stages) == 0 or stages[0][1] is None else (len(stages[0][1]) if hasattr(stages[0][1], '__len__') else 0)
    summary = f"Initial detected: {total} lines"
    cv2.putText(out, summary, (w - 10 - 6 * len(summary), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return out

def save_config_and_results(config, num_streaks, filename="processing_results.txt", dir="output"):
    """
    Save configuration methods used and number of streaks extracted to a text file.
    """
    os.makedirs(dir, exist_ok=True)
    filepath = os.path.join(dir, filename)
    
    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("STREAK DETECTION PROCESSING RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        # Background Detection Method
        f.write("BACKGROUND DETECTION METHOD:\n")
        f.write("-" * 40 + "\n")
        background_method = config.get("background_detection_method", {})
        for method, enabled in background_method.items():
            status = "ENABLED" if enabled else "disabled"
            f.write(f"  {method}: {status}\n")
        
        f.write("\n")
        
        # Line Filters
        f.write("LINE FILTERS ENABLED:\n")
        f.write("-" * 40 + "\n")
        enabled_filters = config.get("enabled_line_filters", {})
        for filter_name, enabled in enabled_filters.items():
            status = "ENABLED" if enabled else "disabled"
            f.write(f"  {filter_name}: {status}\n")
        
        f.write("\n")
        
        # Other Configuration
        f.write("OTHER CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Default Min Line Length: {config.get('default_minlinelength')}\n")
        f.write(f"  Estimated Streak Length Enabled: {config.get('estimated_streak_length_enabled')}\n")
        f.write(f"  Logging Level: {config.get('logging_level')}\n")
        
        f.write("\n")
        
        # Results
        f.write("RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total Streaks Extracted: {num_streaks}\n")
        f.write("=" * 60 + "\n")
    
    logging.info(f"Saved processing results to {filepath}")
    return filepath

def hotpixel_removal(image, threshold=5000):
    """
    Simple hot pixel removal by thresholding.
    Pixels above the threshold are set to the median of their 3x3 neighborhood.
    """
    cleaned_image = image.copy()
    hot_pixels = np.where(cleaned_image > threshold)
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        # Get 3x3 neighborhood
        y_min = max(y - 1, 0)
        y_max = min(y + 2, cleaned_image.shape[0])
        x_min = max(x - 1, 0)
        x_max = min(x + 2, cleaned_image.shape[1])
        neighborhood = cleaned_image[y_min:y_max, x_min:x_max]
        median_value = np.median(neighborhood)
        cleaned_image[y, x] = median_value
    logging.info(f"Removed {len(hot_pixels[0])} hot pixels using threshold {threshold}")
    return cleaned_image

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

        #hotpixel removal or image calibration, depending on config
        if config.get("image_calibration") == True:
            image = calibrate_image(image, file_path)
        else: image = hotpixel_removal(image)

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
        
        num_streaks = len(lines) if lines is not None else 0
        save_config_and_results(config, num_streaks)

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
