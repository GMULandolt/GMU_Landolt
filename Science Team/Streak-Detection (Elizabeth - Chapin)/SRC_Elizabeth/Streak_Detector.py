import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import rotate, gaussian_filter, label, map_coordinates, find_objects
from skimage.feature import match_template, peak_local_max
from skimage.filters import threshold_otsu
from sklearn.neighbors import BallTree
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import fftconvolve
import os

class FourierStreakDetector:
    """
    A class to automatically detect and extract satellite or orbital debris streaks 
    in astronomical FITS images.
    
    The pipeline works by:
    1. Isolating the brightest pixels to find the longest, most prominent streak in the image.
    2. Using Principal Component Analysis (PCA) on the streak's shape to determine its exact angle, length, and width.
    3. Extracting that streak to use as a "Master Template".
    4. Using Fast Fourier Transform (FFT) cross-correlation to find all other regions in the image that match the template.
    5. Filtering out false positives (too close to edge, too close to another streak, or too circular).
    """
    
    def __init__(self, fits_path, min_box_separation=0):
        """
        Parameters
        ----------
        fits_path : str
            Path to the FITS file to be analyzed.
        min_box_separation : float
            Minimum fractional overlap allowed between the template bounding
            box and any *other* streak bounding box (0 = no overlap required).
        """
        self.fits_path = fits_path
        self.min_box_separation = min_box_separation

        # Core image data
        self.image_data = None
        self.header = None
        self.streaks = []

        # Master Template attributes
        self.best_template = None
        self.best_template_angle = 0
        self.best_template_bbox = None   # (y_start, y_end, x_start, x_end)
        self.template_coords = None

        # Rejection bookkeeping and feature storage
        self._rejection_map = {}
        self._all_features = []          # List of dictionaries holding metadata for every detected object

        # Intermediates for FFT/Correlation math and diagnostics
        self.correlation_map = None
        self.detected_peaks_coords = None
        self.fft_image_log = None
        self.fft_template_log = None
        self.fft_product_log = None

        # Telemetry / Physical attributes extracted from FITS
        self.exposure_time_s = None      # Exposure time in seconds
        self.pixel_scale_arcsec = None   # Plate scale in arcseconds per pixel
        self.expected_streak_length_px = None # Length derived mathematically from the master template
        
        # --- variables for future exposure-based length calculations ---
        self.theoretical_streak_length_px = None 
        self.use_exposure_length = False # Keep False to preserve the template-driven behavior by default

    # ------------------------------------------------------------------
    # 1.  LOADING & PREPROCESSING
    # ------------------------------------------------------------------
    def load_image(self):
        """
        Loads the FITS image, extracts telemetry from the header, and normalizes the image data.
        Normalization (zero-mean, unit-variance) is critical for cross-correlation math to work reliably.
        """
        if not os.path.exists(self.fits_path):
            raise FileNotFoundError(f"FITS file not found: {self.fits_path}")

        # Open the FITS file and extract the primary 2D image data
        with fits.open(self.fits_path) as hdul:
            data_found = False
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    # If it's a 3D data cube, just take the first slice
                    if data.ndim == 3:
                        print("3-D FITS cube detected – using first slice.")
                        data = data[0]
                    self.image_data = data.astype(np.float32)
                    self.header = hdu.header
                    data_found = True
                    break
            if not data_found:
                raise ValueError("No image data found in the FITS file.")

        # Try to pull exposure time and plate scale from the header
        self._parse_exposure_info()

        # NORMALIZE THE IMAGE: 
        # By subtracting the median and dividing by the standard deviation, the background sky 
        # centers around ~0.0, and bright streaks become positive outliers. 
        # This prevents the cross-correlation algorithm from heavily weighting bright, noisy sky backgrounds.
        self.image_data = ((self.image_data - np.median(self.image_data))
                           / np.std(self.image_data))
        print("FITS image loaded and normalised.")
        
        # If possible, calculate what the theoretical streak length *should* be based on physics
        self._estimate_streak_length_from_exposure()

    def _parse_exposure_info(self):
        """
        Attempts to read the exposure time and the pixel scale (arcseconds/pixel) from the FITS header.
        If standard keywords fail, it uses Astropy's World Coordinate System (WCS) to derive it mathematically.
        """
        h = self.header
        
        # Look for standard exposure time keys
        for key in ('EXPTIME', 'EXPOSURE', 'EXPTIMED', 'EXP_TIME'):
            if key in h:
                self.exposure_time_s = float(h[key])
                print(f"  Exposure time: {self.exposure_time_s} s  (key={key})")
                break
        if self.exposure_time_s is None:
            print("  WARNING: No exposure-time keyword found in header.")

        # Look for standard plate scale keys (CD matrices or CDELT keys)
        if 'CDELT2' in h:
            self.pixel_scale_arcsec = abs(float(h['CDELT2'])) * 3600.0
        elif 'CD2_2' in h:
            self.pixel_scale_arcsec = abs(float(h['CD2_2'])) * 3600.0
        else:
            # Fallback: If the image is plate-solved, Astropy can calculate the scale using WCS
            try:
                w = WCS(h)
                if w.has_celestial:
                    # proj_plane_pixel_scales returns degrees/pixel. Multiply by 3600 to convert to arcsec/px
                    scales = proj_plane_pixel_scales(w) * 3600.0
                    self.pixel_scale_arcsec = float(np.mean(scales))
            except Exception as e:
                print(f"  WARNING: Astropy WCS could not extract pixel scale: {e}")

        if self.pixel_scale_arcsec:
            print(f"  Pixel scale: {self.pixel_scale_arcsec:.4f} arcsec/px")

    def _estimate_streak_length_from_exposure(self, expected_angular_velocity_arcsec_s=15.0):
        """
        Calculates the  streak length based on the tracking speed and exposure time.
        
        Formula: 
        Length [px] = (Velocity [arcsec/s] * Exposure [s]) / Pixel Scale [arcsec/px]
        
        Note: The default velocity (15 arcsec/s) is the sidereal tracking rate. This means 
        if the telescope is stationary, stars will trail at exactly this length.
        """
        if self.exposure_time_s is None or self.pixel_scale_arcsec is None:
            print("  Cannot calculate length: Missing exposure time or pixel scale.")
            self.theoretical_streak_length_px = None
            return

        self.theoretical_streak_length_px = (expected_angular_velocity_arcsec_s * self.exposure_time_s) / self.pixel_scale_arcsec
        print(f"  Streak length calculated: {self.theoretical_streak_length_px:.1f} px "
              f"(@ {expected_angular_velocity_arcsec_s} arcsec/s)")

    # ------------------------------------------------------------------
    # 2.  MASTER TEMPLATE CREATION
    # ------------------------------------------------------------------
    def _create_template_from_image(self, min_area=40, padding=10, max_width_std=4.0):
        """
        This function identifies the most perfect, beautiful streak in the image to use as a template.
        It converts the image to a binary mask, measures the mathematical shape (covariance) of every 
        bright point, filters out the bad ones, and saves the best one.
        """
        if self.image_data is None:
            raise RuntimeError("Image data not loaded.")

        print("Creating binary mask to find best streak template...")
        
        # Standard thresholding (like Otsu) assumes a 50/50 mix of light and dark pixels. 
        # Astronomical images are 99% dark sky, so Otsu fails. 
        # Instead, we aggressively threshold everything below the 99th percentile to 0 (black). 
        # This perfectly isolates the 1% of bright signal (stars, streaks, galaxies).
        # We sample every 4th pixel ([::4, ::4]) to make this percentile math run 16x faster!
        thresh = np.percentile(self.image_data[::4, ::4], 99.0)
        binary_mask = self.image_data > thresh
        
        # 'binary_closing' acts like a morphological glue. It fills in tiny 1-pixel gaps in our streaks 
        # so that a single streak doesn't accidentally get counted as two separate pieces.
        binary_mask = binary_closing(binary_mask, structure=np.ones((3,3)))
        
        # Label assigns a unique integer ID to every connected component of bright pixels
        labeled_image, num_features = label(binary_mask)

        if num_features == 0:
            raise ValueError("No features found in binary mask to create template.")

        ih, iw = self.image_data.shape
        features = []

        # `find_objects` is a massive optimization. It returns the bounding box for every feature instantly, 
        # rather than forcing numpy to scan the entire 16-megapixel image thousands of times.
        slices = find_objects(labeled_image)

        # ---- Measure the physical characteristics of every object ----
        for i, slc in enumerate(slices):
            if slc is None:
                continue
                
            label_id = i + 1
            
            # Extract just the tiny stamp of the feature to save memory and time
            sub_labeled = labeled_image[slc]
            
            # Find the local (y, x) coordinates of every pixel that belongs to this specific feature
            local_coords = np.argwhere(sub_labeled == label_id)
            
            # Throw away tiny specks of noise
            if len(local_coords) < min_area:
                continue
                
            # Shift the local coordinates back to their real positions in the global image
            y_start, x_start = slc[0].start, slc[1].start
            coords = local_coords + np.array([y_start, x_start])

            # --- PRINCIPAL COMPONENT ANALYSIS (PCA) ON FEATURE SHAPE ---
            # By calculating the spatial covariance matrix of the pixel coordinates, we can mathematically 
            # determine exactly how long, how wide, and at what angle the object is sitting at!
            cov = np.cov(coords.T)
            if np.isnan(cov).any():
                continue
                
            # Eigenvalues tell us the variance (size) along the major and minor axes.
            # eigvals[1] is the length variance. eigvals[0] is the width variance.
            # Eigenvectors tell us the orientation (angle) of those axes.
            eigvals, eigvecs = np.linalg.eigh(cov)
            if eigvals[0] <= 1e-6: # Prevent division by zero for 1-pixel-wide artifacts
                continue

            # Elongation is the ratio of Length to Width
            elongation = np.sqrt(eigvals[1]) / np.sqrt(eigvals[0])

            # Calculate a basic bounding box
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = (y_min, y_max, x_min, x_max)

            # Measure how bright the feature is in the original un-labeled image
            mean_intensity = self.image_data[y_min:y_max+1, x_min:x_max+1].mean()
            max_intensity  = self.image_data[y_min:y_max+1, x_min:x_max+1].max()

            # Save all this math for later filtering and visualization
            features.append({
                'label':        label_id,
                'coords':       coords,
                'elongation':   elongation,
                'eigvals':      eigvals,
                'eigvecs':      eigvecs,
                'bbox':         bbox,
                'mean_int':     mean_intensity,
                'max_int':      max_intensity,
                'rejection':    None,
            })

        if not features:
            raise ValueError("No valid features after initial filtering.")

        # SORTING: We want the longest streak to act as our master template.
        # We sort descending by PHYSICAL LENGTH (variance along the principal axis -> eigvals[1]).
        # (We don't sort by elongation, because a tiny 3x1 pixel noise artifact has an almost infinite elongation!)
        features.sort(key=lambda f: f['eigvals'][1], reverse=True)

        # ---- FILTER OUT BAD CANDIDATES ----
        valid_features = []
        edge_margin = 15

        for f in features:
            y_min, y_max, x_min, x_max = f['bbox']

            # 1. Edge Rejection: Don't build templates from streaks that bleed off the edge of the sensor
            if (x_min < edge_margin or x_max > iw - edge_margin or
                y_min < edge_margin or y_max > ih - edge_margin):
                f['rejection'] = 'edge'
                continue

            # 2. Circularity Rejection: Streaks are lines. A perfect circle has an elongation of 1.0. 
            # If it is less than 3.0, it is probably just a slightly warped star.
            if f['elongation'] < 3.0: 
                f['rejection'] = 'circular'
                continue
                
            
            # Since sqrt(eigvals[0]) is essentially the 1-sigma width of the object in pixels, 
            # capping it at max_width_std ensures we strictly keep thin lines and get rid larger features.
            width_std = np.sqrt(f['eigvals'][0])
            if width_std > max_width_std: 
                f['rejection'] = 'too_wide'
                continue

            valid_features.append(f)

        if not valid_features:
            raise ValueError("All features were rejected (edge, circular, etc.")

        # The #1 feature remaining in the list is our champion Master Template!
        chosen = valid_features[0]
        cy1, cy2, cx1, cx2 = chosen['bbox']

        # Now, mark any features whose bounding box touches the Master Template as 'overlap'
        for f in valid_features[1:]:
            fy1, fy2, fx1, fx2 = f['bbox']
            # Rectangle intersection math
            inter_y = max(0, min(cy2, fy2) - max(cy1, fy1))
            inter_x = max(0, min(cx2, fx2) - max(cx1, fx1))
            if inter_y * inter_x > 0:
                f['rejection'] = 'overlap'

        # Any feature that survived all the filters, but wasn't the #1 longest one, gets tagged as 'prominence'.
        # These are usually perfectly good streaks, they just weren't selected to be the master template!
        for f in valid_features[1:]:
            if f['rejection'] is None:
                f['rejection'] = 'prominence'

        self._all_features = features

        # ---- EXTRACT AND FINALIZE THE TEMPLATE IMAGE ----
        self.template_coords = chosen['coords']
        
        # Calculate the mathematical length of the streak in pixels using the variance
        primary_variance = chosen['eigvals'][1]
        self.expected_streak_length_px = np.sqrt(12 * primary_variance)
        print(f"Calculated universal streak length from template: {self.expected_streak_length_px:.1f} px")
        
        # Extract the angle in degrees using the principal eigenvector (y vs x)
        eigvecs = chosen['eigvecs']
        v = eigvecs[:, -1]
        self.best_template_angle = np.degrees(np.arctan2(v[0], v[1]))

        # Cut out the template from the original image, padding it slightly so it captures the background context
        y_min, y_max, x_min, x_max = chosen['bbox']
        y_start = max(0, y_min - padding)
        y_end   = min(ih, y_max + padding)
        x_start = max(0, x_min - padding)
        x_end   = min(iw, x_max + padding)
        self.best_template_bbox = (y_start, y_end, x_start, x_end)

        self.best_template = self.image_data[y_start:y_end, x_start:x_end].copy()
        if self.best_template.size == 0:
            raise ValueError("Failed to create a valid template.")

        # Re-normalize just the template cutout to zero-mean. 
        # This is strictly required for the cross-correlation math in the next step to work properly.
        if np.std(self.best_template) > 1e-6:
            self.best_template = ((self.best_template - np.mean(self.best_template))
                                  / np.std(self.best_template))
        else:
            self.best_template = self.best_template - np.mean(self.best_template)

        print(f"Template created from streak at angle {self.best_template_angle:.2f}°  "
              f"(elongation={chosen['elongation']:.2f})")

    # ------------------------------------------------------------------
    # 3.  STREAK DETECTION (CROSS-CORRELATION)
    # ------------------------------------------------------------------
    def detect_streaks_by_template(self,
                                   threshold_sigma=0.75,
                                   streak_length=None,
                                   min_distance=10):
        """
        Uses the Master Template to scan the entire image and detect identical streaks.
        It uses Fast Fourier Transforms (FFT) to perform the scanning algorithm in a fraction of a second.
        """
        if self.image_data is None:
            raise RuntimeError("Image has not been loaded.")

        # Step 1: Guarantee the template exists
        self._create_template_from_image()

        # Step 2: Determine which length metric to use when drawing the final lines
        if streak_length is None:
            if self.use_exposure_length and self.theoretical_streak_length_px is not None:
                streak_length = self.theoretical_streak_length_px
            else:
                streak_length = (self.expected_streak_length_px
                                 if self.expected_streak_length_px
                                 else 50)
        
        print(f"Using streak_length={streak_length:.1f} px for endpoint calculation.")

        # Step 3: FFT Cross-Correlation
        # Standard cross-correlation slides the template pixel-by-pixel across the whole image. 
        # This takes $O(N \cdot M)$ time (often minutes!).
        # By convolving in the frequency domain using FFTs, the math drops to $O(N \log N)$ and finishes in < 1 second.
        # NOTE: Mathematical convolution flips the template. To simulate a true cross-correlation using convolution, 
        # we must physically flip the template 180 degrees first! `[::-1, ::-1]` accomplishes this.
        print("Performing template match (FFT-accelerated)...")
        flipped_template = self.best_template[::-1, ::-1]
        
        # `mode='same'` ensures the output correlation map is the exact same size as the input image
        self.correlation_map = fftconvolve(self.image_data, flipped_template, mode='same')
        print("Correlation map generated.")

        # Step 4: Extract the peaks
        # The correlation map contains bright "hotspots" wherever the template matched perfectly.
        # We set a threshold dynamically based on the median and standard deviation of the correlation map.
        mean_c  = np.mean(self.correlation_map)
        med_c   = np.median(self.correlation_map)
        std_c   = np.std(self.correlation_map)
        threshold_abs = med_c + threshold_sigma * std_c

        # `peak_local_max` finds the (y,x) coordinates of the hotspots
        peak_coords_yx = peak_local_max(
            self.correlation_map,
            min_distance=min_distance,
            threshold_abs=threshold_abs)

        if peak_coords_yx.size == 0:
            print("No peaks found.")
            self.detected_peaks_coords = np.empty((0, 2), dtype=int)
            return

        # Convert back to (x, y) coordinates for standard math
        self.detected_peaks_coords = peak_coords_yx[:, ::-1]
        print(f"Found {len(self.detected_peaks_coords)} peaks.")

        # Perform nearest-neighbor analysis for diagnostic purposes
        self._nearest_neighbour_analysis()

        # Step 5: Convert the center points into start and end points using trigonometry
        half_len  = streak_length / 2
        angle_rad = np.deg2rad(self.best_template_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        for (x_c, y_c) in self.detected_peaks_coords:
            # dx, dy represent the horizontal and vertical distance from the center to the edge of the streak
            dx = half_len * cos_a
            dy = half_len * sin_a
            
            # Start Point (x1, y1) and End Point (x2, y2)
            x1, y1 = x_c - dx, y_c - dy
            x2, y2 = x_c + dx, y_c + dy
            
            # Save the peak correlation score so we can rank the streaks later
            corr_val = self.correlation_map[int(y_c), int(x_c)]
            
            # Store the final streak as a dictionary containing its geometric data
            self.streaks.append({
                'endpoints': ((x1, y1), (x2, y2)),
                'center': (x_c, y_c),
                'bbox': (min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2)),
                'corr_val': corr_val,
                'rejection': None # Default is accepted until the filter runs
            })
            
        # Run the final pass filter to reject invalid detections
        self._filter_detected_streaks()

    # ------------------------------------------------------------------
    # 4.  FILTERING & NEAREST-NEIGHBOUR MATH
    # ------------------------------------------------------------------
    def _filter_detected_streaks(self, edge_margin=5, proximity_margin=20):
        """
        Runs through the final list of detected streaks and rejects ones that are invalid.
        Critically, it rejects streaks that are too close to each other, ensuring that downstream 
        tasks (like Pill Photometry) have a clean, uncontaminated background to measure against.
        """
        ih, iw = self.image_data.shape

        # Sort the streaks by their correlation score descending, so we always process the 
        # highest confidence (brightest) streaks first.
        self.streaks.sort(key=lambda s: s['corr_val'], reverse=True)
        max_corr = self.streaks[0]['corr_val']

        for i, streak in enumerate(self.streaks):
            y_min, y_max, x_min, x_max = streak['bbox']
            x_c, y_c = streak['center']

            # 1. Edge Rejection (Streaks too close to the sensor edge)
            if (x_min < edge_margin or x_max > iw - edge_margin or
                y_min < edge_margin or y_max > ih - edge_margin):
                streak['rejection'] = 'edge'
                continue

            # 2. Proximity Rejection (Required for clean Pill Photometry)
            # Checks the current streak against all *previously accepted* (brighter) streaks.
            for j in range(i):
                prev_streak = self.streaks[j]
                if prev_streak['rejection'] is None:
                    py_min, py_max, px_min, px_max = prev_streak['bbox']
                    
                    # Expand the bounding boxes by `proximity_margin` (e.g. 20 pixels). 
                    # If the expanded boxes intersect, the streaks are too close to safely measure flux!
                    inter_y = max(0, min(y_max, py_max + proximity_margin) - max(y_min, py_min - proximity_margin))
                    inter_x = max(0, min(x_max, px_max + proximity_margin) - max(x_min, px_min - proximity_margin))
                    if inter_y * inter_x > 0:
                        streak['rejection'] = 'too_close'
                        break

            # 3. Prominence Rejection 
            # If the streak is less than 50% as "confident" as the best streak in the image, toss it.
            if streak['rejection'] is None and streak['corr_val'] < (max_corr * 0.5):
                streak['rejection'] = 'prominence'


    def _nearest_neighbour_analysis(self):
        """
        Uses a highly optimized spatial BallTree (from scikit-learn) to calculate the distance 
        from every detected streak to its absolute closest neighbor.
        """
        if (self.detected_peaks_coords is None or
                len(self.detected_peaks_coords) < 2):
            self.nn_distances = np.array([])
            self.nn_indices   = np.array([], dtype=int)
            return

        coords_rad = np.deg2rad(self.detected_peaks_coords.astype(float))
        tree = BallTree(coords_rad, metric='euclidean')

        # Query the tree for k=2 nearest neighbors. 
        # (The closest point to point A is always point A itself, so we need k=2 to get the actual neighbor)
        distances, indices = tree.query(coords_rad, k=2)
        
        # Store the distance and index of the true nearest neighbor
        self.nn_distances = distances[:, 1]   
        self.nn_indices   = indices[:, 1]

        print(f"Nearest-neighbour stats: "
              f"min={self.nn_distances.min():.1f} px, "
              f"mean={self.nn_distances.mean():.1f} px, "
              f"max={self.nn_distances.max():.1f} px")


    def _calculate_fft_intermediates(self):
        """
        Computes the Fast Fourier Transform (FFT) magnitudes for the original image and the template.
        This is strictly used for the `display_fourier_correlation_steps` visualization plot.
        """
        print("Calculating FFT intermediates...")
        # Image FFT
        fft_img         = np.fft.fft2(self.image_data)
        fft_img_shifted = np.fft.fftshift(fft_img) # Shift low frequencies to the center of the image
        self.fft_image_log = np.log(np.abs(fft_img_shifted) + 1e-9)

        # Template FFT (padded to match image size)
        padded = np.zeros_like(self.image_data)
        th, tw = self.best_template.shape
        ih, iw = self.image_data.shape
        cy, cx = ih // 2, iw // 2
        padded[cy - th//2: cy + th - th//2, cx - tw//2: cx + tw - tw//2] = self.best_template

        fft_tpl         = np.fft.fft2(padded)
        fft_tpl_shifted = np.fft.fftshift(fft_tpl)
        self.fft_template_log = np.log(np.abs(fft_tpl_shifted) + 1e-9)

        # Spectral Product (Image * Conjugate of Template) - This represents the correlation in frequency space!
        fft_product         = fft_img * np.conj(fft_tpl)
        fft_product_shifted = np.fft.fftshift(fft_product)
        self.fft_product_log = np.log(np.abs(fft_product_shifted) + 1e-9)


    # ------------------------------------------------------------------
    # 5.  DIAGNOSTIC PLOTTING MODULES
    # ------------------------------------------------------------------
    
    # Master dictionary associating rejection reasons with plot colors
    _REJECTION_COLORS = {
        'prominence': 'yellow',
        'overlap':    'blue',
        'too_close':  'magenta',
        'edge':       'red',
        'saturated':  'purple',
        'circular':   'orange',
        'too_wide':   'cyan',
    }

    def display_diagnostics_plot_1(self):
        """
        Plots the original image and illustrates exactly which features were rejected 
        from becoming the Master Template, and which one won.
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        # ── Panel 1: Original Image ──────────────────────────────────
        # Display with 1st/99th percentiles for good contrast
        axs[0].imshow(self.image_data, cmap='gray', origin='lower',
                      vmin=np.percentile(self.image_data, 1),
                      vmax=np.percentile(self.image_data, 99))

        legend_handles = []

        # Draw all the candidate features that were evaluated
        for f in self._all_features:
            reason = f['rejection']
            if reason is None:
                continue
            color  = self._REJECTION_COLORS.get(reason, 'white')
            coords = f['coords']
            
            label_text = f'Rejected: {reason}'
            
            handle = axs[0].plot(coords[:, 1], coords[:, 0], '.',
                                 markersize=1, color=color, alpha=0.4,
                                 label=label_text)[0]
            legend_handles.append(handle)

        # Draw the champion Master Template in green
        if self.template_coords is not None:
            h = axs[0].plot(self.template_coords[:, 1],
                            self.template_coords[:, 0],
                            'g.', markersize=1, alpha=0.4,
                            label='Selected template')[0]
            legend_handles.append(h)

        # Draw a bounding box around the chosen template for visibility
        if self.best_template_bbox is not None:
            y_start, y_end, x_start, x_end = self.best_template_bbox
            rect = patches.Rectangle(
                (x_start, y_start),
                x_end - x_start, y_end - y_start,
                linewidth=2, edgecolor='lime', facecolor='none',
                linestyle='--', label='Template box')
            axs[0].add_patch(rect)

        # Generate custom key symbols for the legend
        for reason, color in self._REJECTION_COLORS.items():
            label_text = f'Rejected: {reason}'
            axs[0].plot([], [], 's', color=color, label=label_text)
            
        # Deduplicate the legend! Without this, matplotlib will create like 500 legend entries.
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[0].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=7, markerscale=4)

        axs[0].set_title("1. Original Image – Template Selection")

        # ── Panel 2: Chosen Template ─────────────────────────────────
        if self.best_template is not None:
            axs[1].imshow(self.best_template, cmap='gray', origin='lower')
            axs[1].set_title(
                f"2. Best Template  (angle: {self.best_template_angle:.2f}°)")
        else:
            axs[1].set_title("2. Template not created")

        plt.tight_layout()
        plt.show()

    def display_diagnostics_plot_2(self):
        """
        Plots the Correlation Map (Heatmap) output by the FFT algorithm, alongside 
        the extracted peaks and nearest-neighbor linkages.
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        if self.correlation_map is not None:
            # ── Panel 1: Nearest Neighbors ───────────────────────────────
            axs[0].imshow(self.correlation_map, cmap='viridis', origin='lower')
            if (self.detected_peaks_coords is not None and len(self.detected_peaks_coords) > 0):
                axs[0].plot(self.detected_peaks_coords[:, 0],
                            self.detected_peaks_coords[:, 1],
                            'r+', markersize=10, alpha=0.6,
                            label='Detected peaks')

                # Draw thin lines connecting every peak to its closest neighbor
                if hasattr(self, 'nn_indices') and self.nn_distances.size > 0:
                    plotted_nn = set()
                    for i, j in enumerate(self.nn_indices):
                        # Use a sorted tuple to ensure line A->B is treated the same as B->A (prevent duplicate drawing)
                        key = tuple(sorted((i, int(j))))
                        if key in plotted_nn:
                            continue
                        plotted_nn.add(key)
                        xi, yi = self.detected_peaks_coords[i]
                        xj, yj = self.detected_peaks_coords[int(j)]
                        axs[0].plot([xi, xj], [yi, yj], 'c-', lw=0.8, alpha=0.5)
                        
            axs[0].set_title("3. Correlation Map, Peaks & Nearest-Neighbour Links")
            axs[0].legend(loc='upper right', fontsize=8)

            # ── Panel 2: Detected Streaks ────────────────────────────────
            axs[1].imshow(self.correlation_map, cmap='viridis', origin='lower')
            for streak in self.streaks:
                (x1, y1), (x2, y2) = streak['endpoints']
                reason = streak['rejection']
                
                # In the final detection context, 'prominence' just means it wasn't the absolute brightest, 
                # but it is still fully accepted.
                if reason is None or reason == 'prominence':
                    color = 'lime' if reason is None else 'yellow'
                    label = 'Accepted'
                else:
                    color = self._REJECTION_COLORS.get(reason, 'red')
                    label = f'Rejected: {reason}'
                    
                axs[1].plot([x1, x2], [y1, y2], color=color, lw=1.5, alpha=0.8, label=label)
            
            # Deduplicate the legend!
            handles, labels = axs[1].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axs[1].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=7)
            axs[1].set_title("4. Correlation Map & Final Detections")
            axs[1].set_xlim(0, self.image_data.shape[1])
            axs[1].set_ylim(0, self.image_data.shape[0])
        else:
            axs[0].set_title("3. Correlation map not calculated")
            axs[1].set_title("4. Correlation map not calculated")

        plt.tight_layout()
        plt.show()

    def display_fourier_correlation_steps(self):
        """Displays the internal spatial frequencies used by the FFT correlation algorithm."""
        if self.fft_image_log is None:
            return
        fig, axs = plt.subplots(1, 3, figsize=(14, 5))
        axs[0].imshow(self.fft_image_log,    cmap='gray',   origin='lower')
        axs[0].set_title("A. FFT Magnitude of Image")
        axs[0].axis('off')
        axs[1].imshow(self.fft_template_log, cmap='gray',   origin='lower')
        axs[1].set_title("B. FFT Magnitude of Template")
        axs[1].axis('off')
        axs[2].imshow(self.fft_product_log,  cmap='inferno', origin='lower')
        axs[2].set_title("C. Spectral Product (Image × Template*)")
        axs[2].axis('off')
        plt.tight_layout()
        plt.show()

    def display_nearest_neighbour_histogram(self):
        """Plots a histogram showing the distribution of distance between streaks."""
        if not hasattr(self, 'nn_distances') or self.nn_distances.size == 0:
            print("No NN data to plot.")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(self.nn_distances, bins=20, color='steelblue', edgecolor='white')
        ax.set_xlabel("Distance to nearest-neighbour peak (pixels)")
        ax.set_ylabel("Count")
        ax.set_title("Nearest-Neighbour Distance Distribution")
        plt.tight_layout()
        plt.show()

    def display_line_profiles(self, profile_length=200):
        """
        Draws an exact 1D line slice straight down the mathematical center of the primary streak,
        plotting its brightness across the image vs. its correlation confidence score.
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        if not self.streaks:
            for ax in axs:
                ax.set_title("No streaks detected")
            plt.tight_layout()
            plt.show()
            return

        (x1, y1), (x2, y2) = self.streaks[0]['endpoints']
        
        # Linearly interpolate points between the start and end of the streak
        t = np.linspace(0, 1, profile_length)
        x_idx = x1 + (x2 - x1) * t
        y_idx = y1 + (y2 - y1) * t
        coords = np.vstack((y_idx, x_idx))

        # Map the coordinates back onto the heatmap and original image arrays to extract values
        if self.correlation_map is not None:
            cp = map_coordinates(self.correlation_map, coords, order=1)
            axs[0].plot(t, cp, 'c-')
            axs[0].set_title("5. Correlation Profile Along Streak")
            axs[0].set_xlabel("Normalised distance")
            axs[0].set_ylabel("Correlation value")

        bp = map_coordinates(self.image_data, coords, order=1)
        axs[1].plot(t, bp, 'm-')
        axs[1].set_title("6. Base Image Profile Along Streak")
        axs[1].set_xlabel("Normalised distance")
        axs[1].set_ylabel("Pixel intensity")

        plt.tight_layout()
        plt.show()

    def display_final_results(self):
        """
        The final master plot overlaying every accepted and rejected streak line directly 
        onto the original FITS image.
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(self.image_data, cmap='gray', origin='lower',
                  vmin=np.percentile(self.image_data, 1),
                  vmax=np.percentile(self.image_data, 99))
        
        accepted_count = 0
        for streak in self.streaks:
            (x1, y1), (x2, y2) = streak['endpoints']
            reason = streak['rejection']
            
            # Draw accepted streaks bold and green, and rejected streaks faint and color-coded
            if reason is None or reason == 'prominence':
                accepted_count += 1
                color = 'lime' if reason is None else 'yellow'
                alpha, lw = 0.9, 2.0 
                label = 'Accepted'
            else:
                color, alpha, lw = self._REJECTION_COLORS.get(reason, 'red'), 0.4, 1.0 
                label = f'Rejected: {reason}'
            
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, label=label)

        title = f"Final Detections ({accepted_count} valid, {len(self.streaks) - accepted_count} rejected)"
        
        # Display the length used for calculations in the title dynamically
        length_to_display = self.theoretical_streak_length_px if (self.use_exposure_length and self.theoretical_streak_length_px) else self.expected_streak_length_px
        
        if length_to_display:
            title += (f"\n(Expected streak length ≈ "
                      f"{length_to_display:.0f} px"
                      f" | Exposure = {self.exposure_time_s} s)")
                      
        ax.set_title(title)
        ax.set_xlim(0, self.image_data.shape[1])
        ax.set_ylim(0, self.image_data.shape[0])
        
        # Deduplicate Legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 6.  FULL PIPELINE RUNNER
    # ------------------------------------------------------------------
    def run(self):
        """Sequentially executes the loading, calculation, and plotting steps."""
        try:
            self.load_image()
            self.detect_streaks_by_template()
            self.display_diagnostics_plot_1()
            self.display_diagnostics_plot_2()
            
            # Uncomment below if you want to see the Fourier components (requires _calculate_fft_intermediates)
            # self._calculate_fft_intermediates()
            # self.display_fourier_correlation_steps()
            
            self.display_nearest_neighbour_histogram()
            self.display_line_profiles()
            self.display_final_results()
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    # Initialize and execute the detector
    detector = FourierStreakDetector("Palomar_Himage3_3sec.fits")
    detector.run()