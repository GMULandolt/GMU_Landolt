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
    def __init__(self, fits_path,
                min_box_separation=0):
        """
        Parameters
        ----------
        fits_path : str
            Path to the FITS file.
        min_box_separation : float
            Minimum fractional overlap allowed between the template bounding
            box and any *other* streak bounding box (0 = no overlap required).
        """
        self.fits_path = fits_path
      
        self.min_box_separation = min_box_separation

        self.image_data = None
        self.header = None
        self.streaks = []

        # Template attributes
        self.best_template = None
        self.best_template_angle = 0
        self.best_template_bbox = None   # (y_start, y_end, x_start, x_end)
        self.template_coords = None

        # Rejection bookkeeping  {label_id: reason_string}
        self._rejection_map = {}
        # All candidate feature info for plotting
        self._all_features = []          # list of dicts

        # Fourier / correlation intermediates
        self.correlation_map = None
        self.detected_peaks_coords = None
        self.fft_image_log = None
        self.fft_template_log = None
        self.fft_product_log = None

        # Exposure / streak-length estimate
        self.exposure_time_s = None      # from FITS header (seconds)
        self.pixel_scale_arcsec = None   # arcsec/pixel from FITS header
        self.expected_streak_length_px = None # Derived from template shape
        
        # --- exposure-based length ---
        self.theoretical_streak_length_px = None 
        self.use_exposure_length = False # Keep False to preserve current behavior!

    # ------------------------------------------------------------------
    # 1.  LOADING
    # ------------------------------------------------------------------
    def load_image(self):
        """Loads and normalises the FITS image.  Also extracts exposure info."""
        if not os.path.exists(self.fits_path):
            raise FileNotFoundError(f"FITS file not found: {self.fits_path}")

        with fits.open(self.fits_path) as hdul:
            data_found = False
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    if data.ndim == 3:
                        print("3-D FITS cube detected – using first slice.")
                        data = data[0]
                    self.image_data = data.astype(np.float32)
                    self.header = hdu.header
                    data_found = True
                    break
            if not data_found:
                raise ValueError("No image data found in the FITS file.")

        # --- Extract exposure time & pixel scale ---
        self._parse_exposure_info()

        # Normalise
        self.image_data = ((self.image_data - np.median(self.image_data))
                           / np.std(self.image_data))
        print("FITS image loaded and normalised.")
        
        # Now that we have exposure time and pixel scale, calculate the theoretical length!
        self._estimate_streak_length_from_exposure()

    def _parse_exposure_info(self):
        """Pulls exposure time and pixel scale from the FITS header using WCS if plate-solved."""
        h = self.header
        
        # Common exposure-time keywords
        for key in ('EXPTIME', 'EXPOSURE', 'EXPTIMED', 'EXP_TIME'):
            if key in h:
                self.exposure_time_s = float(h[key])
                print(f"  Exposure time: {self.exposure_time_s} s  (key={key})")
                break
        if self.exposure_time_s is None:
            print("  WARNING: No exposure-time keyword found in header.")

        # Pixel scale – Try hardcoded keys first
        if 'CDELT2' in h:
            self.pixel_scale_arcsec = abs(float(h['CDELT2'])) * 3600.0
        elif 'CD2_2' in h:
            self.pixel_scale_arcsec = abs(float(h['CD2_2'])) * 3600.0
        else:
            # Fallback: Let Astropy's WCS solve for the pixel scale matrix natively
            try:
                w = WCS(h)
                if w.has_celestial:
                    # proj_plane_pixel_scales returns degrees/pixel. Multiply by 3600 to get arcsec/px
                    scales = proj_plane_pixel_scales(w) * 3600.0
                    self.pixel_scale_arcsec = float(np.mean(scales))
            except Exception as e:
                print(f"  WARNING: Astropy WCS could not extract pixel scale: {e}")

        if self.pixel_scale_arcsec:
            print(f"  Pixel scale: {self.pixel_scale_arcsec:.4f} arcsec/px")

    def _estimate_streak_length_from_exposure(self, expected_angular_velocity_arcsec_s=15.0):
        """
        Calculates theoretical streak length from exposure time and pixel scale.
        
        Formula: 
        Length [px] = (Velocity [arcsec/s] * Exposure [s]) / Pixel Scale [arcsec/px]
        
        Parameters
        ----------
        expected_angular_velocity_arcsec_s : float
            Expected angular velocity of the streak in arcseconds per second.
            Defaults to 15.0 arcsec/s (Typical for GEO satellites on a sidereal 
            tracking mount, or stars on a stationary mount).
        """
        if self.exposure_time_s is None or self.pixel_scale_arcsec is None:
            print("  Cannot calculate theoretical length: Missing exposure time or pixel scale.")
            self.theoretical_streak_length_px = None
            return

        self.theoretical_streak_length_px = (expected_angular_velocity_arcsec_s * self.exposure_time_s) / self.pixel_scale_arcsec
        print(f"  Theoretical streak length calculated: {self.theoretical_streak_length_px:.1f} px "
              f"(@ {expected_angular_velocity_arcsec_s} arcsec/s)")

    # ------------------------------------------------------------------
    # 2.  TEMPLATE CREATION  (with rejection colour-coding)
    # ------------------------------------------------------------------
    def _create_template_from_image(self, min_area=40, padding=10, max_width_std=4.0):
        """
        Finds the best (most elongated) streak for use as a template.
        """
        if self.image_data is None:
            raise RuntimeError("Image data not loaded.")

        print("Creating binary mask to find best streak template...")
        # Otsu thresholding fails on sparse astronomical images. 
        # Using a high percentile strictly isolates the bright signal (stars/streaks).
        
        thresh = np.percentile(self.image_data, 99.0)
        binary_mask = self.image_data > thresh
        binary_mask = binary_closing(binary_mask, structure=np.ones((3,3)))
        labeled_image, num_features = label(binary_mask)

        if num_features == 0:
            raise ValueError("No features found in binary mask to create template.")

        ih, iw = self.image_data.shape
        features = []

        # Extract slices for each label
        slices = find_objects(labeled_image)

        # ---- collect all candidate features ----
        for i, slc in enumerate(slices):
            if slc is None:
                continue
                
            label_id = i + 1
            
            # Extract just the tiny bounding box of the feature from the labeled image
            sub_labeled = labeled_image[slc]
            
            # Find coordinates within this tiny bounding box
            local_coords = np.argwhere(sub_labeled == label_id)
            if len(local_coords) < min_area:
                continue
                
            # Shift coordinates back up to global image space
            y_start, x_start = slc[0].start, slc[1].start
            coords = local_coords + np.array([y_start, x_start])

            cov = np.cov(coords.T)
            if np.isnan(cov).any():
                continue
            eigvals, eigvecs = np.linalg.eigh(cov)
            if eigvals[0] <= 1e-6:
                continue

            elongation = np.sqrt(eigvals[1]) / np.sqrt(eigvals[0])

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = (y_min, y_max, x_min, x_max)

            mean_intensity = self.image_data[y_min:y_max+1, x_min:x_max+1].mean()
            max_intensity  = self.image_data[y_min:y_max+1, x_min:x_max+1].max()

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

        # Sort descending by PHYSICAL LENGTH (variance along principal axis) to find the longest, most robust streak!
        # (Elongation is too vulnerable to tiny 3x1 pixel noise artifacts having near-infinite elongation)
        features.sort(key=lambda f: f['eigvals'][1], reverse=True)

        # ---- apply rejection criteria ----
        valid_features = []
        
        edge_margin = 15

        for f in features:
            y_min, y_max, x_min, x_max = f['bbox']

            # 1. Edge Rejection (Don't build templates from edge artifacts)
            if (x_min < edge_margin or x_max > iw - edge_margin or
                y_min < edge_margin or y_max > ih - edge_margin):
                f['rejection'] = 'edge'
                continue

            # 2. Circular Object Rejection (Must be stretched)
            if f['elongation'] < 3.0: 
                f['rejection'] = 'circular'
                continue
                
            # 3. Blob / "Too Wide" Rejection (Nukes massive galaxies and bright blobs)
            width_std = np.sqrt(f['eigvals'][0])
            if width_std > max_width_std: 
                f['rejection'] = 'too_wide'
                continue

            valid_features.append(f)

        if not valid_features:
            raise ValueError("All features were rejected (edge, circular, or too blobby).")

        # Overlap check – mark features whose box overlaps the chosen best box
        chosen = valid_features[0]
        cy1, cy2, cx1, cx2 = chosen['bbox']

        for f in valid_features[1:]:
            fy1, fy2, fx1, fx2 = f['bbox']
            # Check intersection
            inter_y = max(0, min(cy2, fy2) - max(cy1, fy1))
            inter_x = max(0, min(cx2, fx2) - max(cx1, fx1))
            if inter_y * inter_x > 0:
                f['rejection'] = 'overlap'

        # Remaining valid (not overlapping) non-chosen → 'prominence'
        # (It is NORMAL for the rest of the good streaks to be yellow in Plot 1, 
        # because Plot 1 is only picking the SINGLE best template!)
        for f in valid_features[1:]:
            if f['rejection'] is None:
                f['rejection'] = 'prominence'

        self._all_features = features

        # ---- build the template from 'chosen' ----
        self.template_coords = chosen['coords']
        primary_variance = chosen['eigvals'][1]
        self.expected_streak_length_px = np.sqrt(12 * primary_variance)
        print(f"Calculated universal streak length from template: {self.expected_streak_length_px:.1f} px")
        
        # Get angle from eigenvector
        eigvecs = chosen['eigvecs']
        v = eigvecs[:, -1]
        self.best_template_angle = np.degrees(np.arctan2(v[0], v[1]))

        y_min, y_max, x_min, x_max = chosen['bbox']
        y_start = max(0, y_min - padding)
        y_end   = min(ih, y_max + padding)
        x_start = max(0, x_min - padding)
        x_end   = min(iw, x_max + padding)
        self.best_template_bbox = (y_start, y_end, x_start, x_end)

        self.best_template = self.image_data[y_start:y_end, x_start:x_end].copy()
        if self.best_template.size == 0:
            raise ValueError("Failed to create a valid template.")

        if np.std(self.best_template) > 1e-6:
            self.best_template = ((self.best_template - np.mean(self.best_template))
                                  / np.std(self.best_template))
        else:
            self.best_template = self.best_template - np.mean(self.best_template)

        print(f"Template created from streak at angle {self.best_template_angle:.2f}°  "
              f"(elongation={chosen['elongation']:.2f})")

    # ------------------------------------------------------------------
    # 3.  STREAK DETECTION
    # ------------------------------------------------------------------
    def detect_streaks_by_template(self,
                                   threshold_sigma=2.5,
                                   streak_length=None,
                                   min_distance=10):
        """
        Detect streaks via template matching + peak_local_max.
        Uses estimated streak length if available.
        """
        if self.image_data is None:
            raise RuntimeError("Image has not been loaded.")

        self._create_template_from_image()

        #Use theoretical exposure length if toggled ON
        if streak_length is None:
            if self.use_exposure_length and self.theoretical_streak_length_px is not None:
                streak_length = self.theoretical_streak_length_px
            else:
                streak_length = (self.expected_streak_length_px
                                 if self.expected_streak_length_px
                                 else 50)
        
        print(f"Using streak_length={streak_length:.1f} px for endpoint calculation.")

        print("Performing template match...")
        flipped_template = self.best_template[::-1, ::-1]
        self.correlation_map = fftconvolve(
            self.image_data, flipped_template, mode='same')
        print("Correlation map generated.")

        mean_c  = np.mean(self.correlation_map)
        med_c   = np.median(self.correlation_map)
        std_c   = np.std(self.correlation_map)
        threshold_abs = med_c + threshold_sigma * std_c

        peak_coords_yx = peak_local_max(
            self.correlation_map,
            min_distance=min_distance,
            threshold_abs=threshold_abs)

        if peak_coords_yx.size == 0:
            print("No peaks found.")
            self.detected_peaks_coords = np.empty((0, 2), dtype=int)
            return

        self.detected_peaks_coords = peak_coords_yx[:, ::-1]   # (x, y)
        print(f"Found {len(self.detected_peaks_coords)} peaks.")

        # ---- nearest-neighbour pairing among peaks ----
        self._nearest_neighbour_analysis()

        # ---- streak endpoints ----
        half_len  = streak_length / 2
        angle_rad = np.deg2rad(self.best_template_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        for (x_c, y_c) in self.detected_peaks_coords:
            dx = half_len * cos_a
            dy = half_len * sin_a
            
            x1, y1 = x_c - dx, y_c - dy
            x2, y2 = x_c + dx, y_c + dy
            
            # Grab the correlation value for prominence filtering
            corr_val = self.correlation_map[int(y_c), int(x_c)]
            
            self.streaks.append({
                'endpoints': ((x1, y1), (x2, y2)),
                'center': (x_c, y_c),
                'bbox': (min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2)),
                'corr_val': corr_val,
                'rejection': None # Default is accepted
            })
            
        self._filter_detected_streaks()

    # ------------------------------------------------------------------
    # 4.  NEAREST-NEIGHBOUR ANALYSIS ON PEAKS
    # ------------------------------------------------------------------
    def _nearest_neighbour_analysis(self):
        """
        Builds a BallTree on detected peaks and stores pairwise distances
        for use in diagnostics.  Stores:
          self.nn_distances  – (N,) array of distance to nearest neighbour
          self.nn_indices    – (N,) array of nearest-neighbour index
        """
        if (self.detected_peaks_coords is None or
                len(self.detected_peaks_coords) < 2):
            self.nn_distances = np.array([])
            self.nn_indices   = np.array([], dtype=int)
            return

        coords_rad = np.deg2rad(self.detected_peaks_coords.astype(float))
        tree = BallTree(coords_rad, metric='euclidean')

        # k=2: first result is the point itself
        distances, indices = tree.query(coords_rad, k=2)
        self.nn_distances = distances[:, 1]   # in pixel units (approx)
        self.nn_indices   = indices[:, 1]

        print(f"Nearest-neighbour stats: "
              f"min={self.nn_distances.min():.1f} px, "
              f"mean={self.nn_distances.mean():.1f} px, "
              f"max={self.nn_distances.max():.1f} px")

    # ------------------------------------------------------------------
    # 5.  FFT INTERMEDIATES
    # ------------------------------------------------------------------
    def _calculate_fft_intermediates(self):
        print("Calculating FFT intermediates...")
        fft_img         = np.fft.fft2(self.image_data)
        fft_img_shifted = np.fft.fftshift(fft_img)
        self.fft_image_log = np.log(np.abs(fft_img_shifted) + 1e-9)

        padded = np.zeros_like(self.image_data)
        th, tw = self.best_template.shape
        ih, iw = self.image_data.shape
        cy, cx = ih // 2, iw // 2
        padded[cy - th//2: cy + th - th//2,
               cx - tw//2: cx + tw - tw//2] = self.best_template

        fft_tpl         = np.fft.fft2(padded)
        fft_tpl_shifted = np.fft.fftshift(fft_tpl)
        self.fft_template_log = np.log(np.abs(fft_tpl_shifted) + 1e-9)

        fft_product         = fft_img * np.conj(fft_tpl)
        fft_product_shifted = np.fft.fftshift(fft_product)
        self.fft_product_log = np.log(np.abs(fft_product_shifted) + 1e-9)

    def _filter_detected_streaks(self, edge_margin=5, saturation_percentile=99.0):
        """
        Filters the final streak list based on edge proximity, 
        saturation, and overlaps.
        """
        ih, iw = self.image_data.shape
                # Define saturation as anything above the 99.99th percentile.
                # This is not a perfect method, you need to do photometric calibration to be sure, but it should catch the worst offenders.
        saturation_limit = np.percentile(self.image_data, 99.99999)

        # Sort by correlation value descending (process best matches first)
        self.streaks.sort(key=lambda s: s['corr_val'], reverse=True)
        max_corr = self.streaks[0]['corr_val']

        for i, streak in enumerate(self.streaks):
            y_min, y_max, x_min, x_max = streak['bbox']
            x_c, y_c = streak['center']

            # 1. Edge Rejection
            if (x_min < edge_margin or x_max > iw - edge_margin or
                y_min < edge_margin or y_max > ih - edge_margin):
                streak['rejection'] = 'edge'
                continue

            # 2. Saturated Rejection
            # Check maximum pixel intensity inside the streak's bounding box
            box_max = self.image_data[int(max(0, y_min)):int(min(ih, y_max+1)),
                                      int(max(0, x_min)):int(min(iw, x_max+1))].max()
            if box_max > saturation_limit:
                streak['rejection'] = 'saturated'
                continue

            # 3. Overlap Rejection
            # Check against all PREVIOUSLY ACCEPTED streaks (since we sorted by best first)
            for j in range(i):
                prev_streak = self.streaks[j]
                if prev_streak['rejection'] is None:
                    py_min, py_max, px_min, px_max = prev_streak['bbox']
                    inter_y = max(0, min(y_max, py_max) - max(y_min, py_min))
                    inter_x = max(0, min(x_max, px_max) - max(x_min, px_min))
                    if inter_y * inter_x > 0:
                        streak['rejection'] = 'overlap'
                        break

            # 4. Prominence Rejection
            # If it survived the above but its correlation is less than 50% of the absolute best streak
            if streak['rejection'] is None and streak['corr_val'] < (max_corr * 0.5):
                streak['rejection'] = 'prominence'


    # ------------------------------------------------------------------
    # 6.  DIAGNOSTIC PLOTS
    # ------------------------------------------------------------------
    _REJECTION_COLORS = {
        'prominence': 'yellow',
        'overlap':    'blue',
        'edge':       'red',
        'saturated':  'purple',
        'circular':   'orange',
    }
    def display_diagnostics_plot_1(self):
        """Original image + template, with colour-coded rejected streaks."""
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        # ── Panel 1: original image ──────────────────────────────────
        axs[0].imshow(self.image_data, cmap='gray', origin='lower',
                      vmin=np.percentile(self.image_data, 1),
                      vmax=np.percentile(self.image_data, 99))

        legend_handles = []

        # Draw rejected features
        for f in self._all_features:
            reason = f['rejection']
            if reason is None:
                continue
            color  = self._REJECTION_COLORS.get(reason, 'white')
            coords = f['coords']
            handle = axs[0].plot(coords[:, 1], coords[:, 0], '.',
                                 markersize=1, color=color, alpha=0.4,
                                 label=reason)[0]
            legend_handles.append(handle)

        # Draw chosen template pixels in green
        if self.template_coords is not None:
            h = axs[0].plot(self.template_coords[:, 1],
                            self.template_coords[:, 0],
                            'g.', markersize=1, alpha=0.4,
                            label='Selected template')[0]
            legend_handles.append(h)

        # Draw bounding box around selected template
        if self.best_template_bbox is not None:
            y_start, y_end, x_start, x_end = self.best_template_bbox
            rect = patches.Rectangle(
                (x_start, y_start),
                x_end - x_start, y_end - y_start,
                linewidth=2, edgecolor='lime', facecolor='none',
                linestyle='--', label='Template box')
            axs[0].add_patch(rect)

        # Add rejection legend key
        for reason, color in self._REJECTION_COLORS.items():
            axs[0].plot([], [], 's', color=color,
                        label=f'Rejected: {reason}')
            
        # Deduplicate legend labels (prevents overflow from hundreds of points)
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[0].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=7, markerscale=4)

        axs[0].set_title("1. Original Image – Template Selection")

        # ── Panel 2: chosen template ─────────────────────────────────
        if self.best_template is not None:
            axs[1].imshow(self.best_template, cmap='gray', origin='lower')
            axs[1].set_title(
                f"2. Best Template  (angle: {self.best_template_angle:.2f}°)")
        else:
            axs[1].set_title("2. Template not created")

        plt.tight_layout()
        plt.show()
        
    def display_diagnostics_plot_2(self):
        """Correlation map with peaks, and with final streak overlays."""
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        if self.correlation_map is not None:
            axs[0].imshow(self.correlation_map, cmap='viridis', origin='lower')
            if (self.detected_peaks_coords is not None and
                    len(self.detected_peaks_coords) > 0):
                axs[0].plot(self.detected_peaks_coords[:, 0],
                            self.detected_peaks_coords[:, 1],
                            'r+', markersize=10, alpha=0.6,
                            label='Detected peaks')

                # ---- nearest-neighbour lines ----
                if hasattr(self, 'nn_indices') and self.nn_distances.size > 0:
                    plotted_nn = set()
                    for i, j in enumerate(self.nn_indices):
                        key = tuple(sorted((i, int(j))))
                        if key in plotted_nn:
                            continue
                        plotted_nn.add(key)
                        xi, yi = self.detected_peaks_coords[i]
                        xj, yj = self.detected_peaks_coords[int(j)]
                        axs[0].plot([xi, xj], [yi, yj],
                                    'c-', lw=0.8, alpha=0.5)
            axs[0].set_title("3. Correlation Map, Peaks & Nearest-Neighbour Links")
            axs[0].legend(loc='upper right', fontsize=8)

            axs[1].imshow(self.correlation_map, cmap='viridis', origin='lower')
            for streak in self.streaks:
                (x1, y1), (x2, y2) = streak['endpoints']
                reason = streak['rejection']
                
                if reason is None or reason == 'prominence':
                    color = 'lime' if reason is None else 'yellow'
                    label = 'Accepted'
                else:
                    color = self._REJECTION_COLORS.get(reason, 'red')
                    label = f'Rejected: {reason}'
                    
                axs[1].plot([x1, x2], [y1, y2], color=color, lw=1.5, alpha=0.8, label=label)
            
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
        """FFT magnitude of image, template, and their spectral product."""
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
        """Histogram of nearest-neighbour distances between detected peaks."""
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
        """Line profiles through the first detected streak."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        if not self.streaks:
            for ax in axs:
                ax.set_title("No streaks detected")
            plt.tight_layout()
            plt.show()
            return

        (x1, y1), (x2, y2) = self.streaks[0]['endpoints']
        t = np.linspace(0, 1, profile_length)
        x_idx = x1 + (x2 - x1) * t
        y_idx = y1 + (y2 - y1) * t
        coords = np.vstack((y_idx, x_idx))

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
        """Final detections overlaid on the original image."""
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(self.image_data, cmap='gray', origin='lower',
                  vmin=np.percentile(self.image_data, 1),
                  vmax=np.percentile(self.image_data, 99))
        
        accepted_count = 0
        for streak in self.streaks:
            (x1, y1), (x2, y2) = streak['endpoints']
            reason = streak['rejection']
            if reason is None:
                accepted_count += 1
                color, alpha, lw = 'lime', 0.9, 2.0 
                label = 'Accepted'
            else:
                color, alpha, lw = self._REJECTION_COLORS.get(reason, 'red'), 0.4, 1.0 
                label = f'Rejected: {reason}'
            
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, label=label)

        title = f"Final Detections ({accepted_count} valid, {len(self.streaks) - accepted_count} rejected)"
        
        # Pull whichever length metric is actively being used for the title
        length_to_display = self.theoretical_streak_length_px if (self.use_exposure_length and self.theoretical_streak_length_px) else self.expected_streak_length_px
        
        if length_to_display:
            title += (f"\n(Expected streak length ≈ "
                      f"{length_to_display:.0f} px"
                      f" | Exposure = {self.exposure_time_s} s)")
                      
        ax.set_title(title)
        ax.set_xlim(0, self.image_data.shape[1])
        ax.set_ylim(0, self.image_data.shape[0])
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 7.  FULL PIPELINE
    # ------------------------------------------------------------------
    def run(self):
        try:
            self.load_image()
            self.detect_streaks_by_template()
            self.display_diagnostics_plot_1()
            self.display_diagnostics_plot_2()
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
    # Choose a FITS file to test on. Make sure to update the path as needed.
    detector = FourierStreakDetector("PalomarArtemis_Himage20sec.fits")
    detector.run()