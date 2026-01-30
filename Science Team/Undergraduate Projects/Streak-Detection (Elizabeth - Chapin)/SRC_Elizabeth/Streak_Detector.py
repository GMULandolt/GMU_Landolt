import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import rotate, gaussian_filter, label, map_coordinates
from skimage.feature import match_template, peak_local_max
from skimage.filters import threshold_otsu
import os

class FourierStreakDetector:
    def __init__(self, fits_path):
        self.fits_path = fits_path
        self.image_data = None
        self.header = None
        self.streaks = []
        # --- Attributes for Template Matching ---
        self.best_template = None
        self.best_template_angle = 0
        self.correlation_map = None
        self.detected_peaks_coords = None
        self.template_coords = None # Pixels of the chosen template streak
        self.fft_image_log = None
        self.fft_template_log = None
        self.fft_product_log = None

    def load_image(self):
        """Loads the FITS image data and header."""
        if not os.path.exists(self.fits_path):
            raise FileNotFoundError(f"FITS file not found: {self.fits_path}")

        with fits.open(self.fits_path) as hdul:
            data_found = False
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    if data.ndim == 3:
                        print("3D FITS cube detected. Using the first slice.")
                        data = data[0]
                    self.image_data = data.astype(np.float32)
                    self.header = hdu.header
                    data_found = True
                    break
            if not data_found:
                raise ValueError("No image data found in the FITS file.")
        
        # Normalize image data for stable correlation
        self.image_data = (self.image_data - np.median(self.image_data)) / np.std(self.image_data)
        print("FITS image loaded and normalized.")
                
    def _create_template_from_image(self, min_length=100, padding=10):
        """
        Creates a template by finding the "best" (most elongated)
        streak in a binarized mask of the original image.
        """
        if self.image_data is None: raise RuntimeError("Image data not loaded.")

        print("Creating binary mask to find best streak...")
        # 1. Binarize the original (un-rotated) image
        thresh = threshold_otsu(self.image_data)
        binary_mask = self.image_data > thresh
        
        # 2. Label all distinct features
        labeled_image, num_features = label(binary_mask)
        
        if num_features == 0:
            raise ValueError("No features found in binary mask to create template.")

        # 3. Find the "best" (most elongated) streak

        # TODO: Find a better metric than just elongation. Brightest streak>
        best_feature_label = -1
        max_elongation = 0.0
        best_feature_coords = None
        best_feature_eigvecs = None

        for i in range(1, num_features + 1):
            coords = np.argwhere(labeled_image == i) # (y, x)
            if len(coords) < min_length:
                continue
            
            # Use covariance matrix to find elongation
            # We need (x, y) order, so transpose
            cov = np.cov(coords.T) 
            if np.isnan(cov).any(): continue
                
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # Avoid division by zero
            if eigvals[0] <= 1e-6: continue
                
            elongation = np.sqrt(eigvals[1]) / np.sqrt(eigvals[0])
            
            if elongation > max_elongation:
                max_elongation = elongation
                best_feature_label = i
                self.template_coords = coords # Save for highlighting
                best_feature_eigvecs = eigvecs

        if best_feature_label == -1:
            raise ValueError(f"No streaks found with elongation > 1.0 and min length {min_length}.")
            
        print(f"Found best streak (label {best_feature_label}) with elongation {max_elongation:.2f}.")

        # 4. Calculate the precise angle from the eigenvector
        # The eigenvector for the largest eigenvalue is the last one
        v = best_feature_eigvecs[:, -1] # (dx, dy)
        self.best_template_angle = np.degrees(np.arctan2(v[0], v[1])) # (y, x) -> dy, dx -> arctan2(dy, dx)

        # 5. Get the bounding box of the best streak
        y_min, x_min = np.min(self.template_coords, axis=0)
        y_max, x_max = np.max(self.template_coords, axis=0)
        
        # 6. "Cut out" the template from the original grayscale image with padding
        y_start = max(0, y_min - padding)
        y_end = min(self.image_data.shape[0], y_max + padding)
        x_start = max(0, x_min - padding)
        x_end = min(self.image_data.shape[1], x_max + padding)
        
        self.best_template = self.image_data[y_start:y_end, x_start:x_end]
        
        if self.best_template.size == 0:
             raise ValueError("Failed to create a valid template.")
        
        # 7. Normalize the template
        if np.std(self.best_template) > 1e-6:
            self.best_template = (self.best_template - np.mean(self.best_template)) / np.std(self.best_template)
        else:
            self.best_template = self.best_template - np.mean(self.best_template)
             
        print(f"Template created from real streak at angle {self.best_template_angle:.2f}°")

    def detect_streaks_by_template(self, threshold_sigma=3.5, streak_length=50, min_distance=10):
        """
        Finds the best template from the image, then finds peaks using
        peak_local_max (no centroiding).
        """
        if self.image_data is None: raise RuntimeError("Image has not been loaded.")
        
        # 1. Create the template *from* the image
        self._create_template_from_image()
        
        # 2. Generate the final correlation map (one time)
        print("Performing single template match...")
        self.correlation_map = match_template(self.image_data, self.best_template, pad_input=True)
        print("Final correlation map generated.")

        # 3. Find peaks using peak_local_max (replaces DAOStarFinder)
        mean, median, std = np.mean(self.correlation_map), np.median(self.correlation_map), np.std(self.correlation_map)
        threshold_abs = median + (threshold_sigma * std)
        
        # Find all coordinates (y, x) that are local maxima
        peak_coords_yx = peak_local_max(self.correlation_map, 
                                        min_distance=min_distance, 
                                        threshold_abs=threshold_abs)
        
        if peak_coords_yx.size == 0:
            print("No peaks found with peak_local_max.")
            self.detected_peaks_coords = []
            return
        
        # peak_local_max returns (y, x). We want (x, y) for consistency.
        self.detected_peaks_coords = peak_coords_yx[:, ::-1] # Flip (y, x) to (x, y)
        
        print(f"Found {len(self.detected_peaks_coords)} peaks using peak_local_max.")
        
        # 4. Calculate streak endpoints
        half_len = streak_length / 2
        
        # Use the angle we found for the template
        angle_rad = np.deg2rad(self.best_template_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        for (x_center, y_center) in self.detected_peaks_coords:
            # Calculate endpoints from the center point
            # dx (change in x) uses cosine
            # dy (change in y) uses sine
            dx = half_len * cos_a
            dy = half_len * sin_a
            
            x1 = x_center - dx
            y1 = y_center - dy
            x2 = x_center + dx
            y2 = y_center + dy
            
            self.streaks.append(((x1, y1), (x2, y2)))

    def display_diagnostics_plot_1(self):
        """Displays the base image and the chosen template."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 7))
        
        # Panel 1: Original Image
        axs[0].imshow(self.image_data, cmap='gray', origin='lower',
                         vmin=np.percentile(self.image_data, 1),
                         vmax=np.percentile(self.image_data, 99))
        if self.template_coords is not None:
            # Highlight the pixels of the chosen template streak in green
            axs[0].plot(self.template_coords[:, 1], self.template_coords[:, 0], 
                           'g.', markersize=1, alpha=0.3, label='Selected Template')
            axs[0].legend(loc='upper right')
        axs[0].set_title("1. Original Image (No Rotation)")

        # Panel 2: Best Template
        if self.best_template is not None:
            axs[1].imshow(self.best_template, cmap='gray', origin='lower')
            axs[1].set_title(f"2. Best Template (Angle: {self.best_template_angle:.2f}°)")
        else:
            axs[1].set_title("2. Template Not Created")
        
        plt.tight_layout()
        plt.show()

    def _calculate_fft_intermediates(self):
        """
        Manually computes the FFTs of the image and template for visualization.
        """
        print("Calculating intermediate FFTs for visualization...")
        
        # 1. FFT of Image
        fft_img = np.fft.fft2(self.image_data)
        fft_img_shifted = np.fft.fftshift(fft_img)
        self.fft_image_log = np.log(np.abs(fft_img_shifted) + 1e-9)

        # 2. FFT of Template (Padded to image size)
        padded_template = np.zeros_like(self.image_data)
        
        # Place template in the center
        th, tw = self.best_template.shape
        ih, iw = self.image_data.shape
        cy, cx = ih // 2, iw // 2
        padded_template[cy - th//2 : cy + th - th//2, cx - tw//2 : cx + tw - tw//2] = self.best_template
        
        fft_tpl = np.fft.fft2(padded_template)
        fft_tpl_shifted = np.fft.fftshift(fft_tpl)
        self.fft_template_log = np.log(np.abs(fft_tpl_shifted) + 1e-9)

        # 3. FFT Product (The "Correlation" in Frequency Domain)
        # Correlation = Image * Conjugate(Template)
        fft_product = fft_img * np.conj(fft_tpl)
        fft_product_shifted = np.fft.fftshift(fft_product)
        self.fft_product_log = np.log(np.abs(fft_product_shifted) + 1e-9)


    def display_diagnostics_plot_2(self):
        """
        Displays the correlation map (clean) and the correlation map (with streaks).
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 7))

        # Panel 1: Correlation Map (Clean, with Centroids)
        if self.correlation_map is not None:
            axs[0].imshow(self.correlation_map, cmap='viridis', origin='lower')
            if self.detected_peaks_coords is not None and len(self.detected_peaks_coords) > 0:
                axs[0].plot(self.detected_peaks_coords[:, 0], self.detected_peaks_coords[:, 1], 
                               'r+', markersize=10, alpha=0.6, label='Detected Peaks')
            axs[0].set_title("3. Correlation Map & Peaks")
            axs[0].legend()
        else:
            axs[0].set_title("3. Correlation Map Not Calculated")
        
        # Panel 2: Correlation Map (With Final Streak Lines)
        if self.correlation_map is not None:
            axs[1].imshow(self.correlation_map, cmap='viridis', origin='lower')
            for (x1, y1), (x2, y2) in self.streaks:
                axs[1].plot([x1, x2], [y1, y2], 'r-', lw=1.5, alpha=0.5)
            axs[1].set_title("4. Correlation Map & Overlain Detections")
            axs[1].set_xlim(0, self.image_data.shape[1])
            axs[1].set_ylim(0, self.image_data.shape[0])
        else:
            axs[1].set_title("4. Correlation Map Not Calculated")
        
        plt.tight_layout()
        plt.show()

    def display_fourier_correlation_steps(self):
        """
        Displays the specific Fourier steps of the correlation process.
        """
        if self.fft_image_log is None: return

        fig, axs = plt.subplots(1, 3, figsize=(12, 7))
        
        # 1. FFT of Image
        axs[0].imshow(self.fft_image_log, cmap='gray', origin='lower')
        axs[0].set_title("A. FFT Magnitude of Image")
        axs[0].set_xticks([]); axs[0].set_yticks([])
    
        # 2. FFT of Template
        axs[1].imshow(self.fft_template_log, cmap='gray', origin='lower')
        axs[1].set_title("B. FFT Magnitude of Template")
        axs[1].set_xticks([]); axs[1].set_yticks([])

        # 3. Product (Correlation Spectrum)
        axs[2].imshow(self.fft_product_log, cmap='inferno', origin='lower')
        axs[2].set_title("C. Spectral Product (Image × Template*)")
        axs[2].set_xticks([]); axs[2].set_yticks([])

        plt.tight_layout()
        plt.show()

    def display_line_profiles(self, profile_length=200):
        """
        Displays line profiles of the correlation map and base image
        for the first detected streak.
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 7))
        
        if not self.streaks:
            axs[0].set_title("5. Correlation Profile (No Streaks Found)")
            axs[1].set_title("6. Base Image Profile (No Streaks Found)")
            plt.tight_layout()
            plt.show()
            return

        # Get the first detected streak
        (x1, y1), (x2, y2) = self.streaks[0]
        
        # Generate coordinates along this line
        t = np.linspace(0, 1, profile_length)
        x_indices = x1 + (x2 - x1) * t
        y_indices = y1 + (y2 - y1) * t
        coords = np.vstack((y_indices, x_indices))

        # 1. Plot Correlation Map Profile
        if self.correlation_map is not None:
            corr_profile = map_coordinates(self.correlation_map, coords, order=1)
            axs[0].plot(t, corr_profile, 'c-')
            axs[0].set_title("5. Correlation Profile Along Streak")
            axs[0].set_xlabel("Normalized Distance Along Streak")
            axs[0].set_ylabel("Correlation Value")
        else:
            axs[0].set_title("5. Correlation Profile Not Calculated")

        # 2. Plot Base Image Profile
        base_profile = map_coordinates(self.image_data, coords, order=1)
        axs[1].plot(t, base_profile, 'm-')
        axs[1].set_title("6. Base Image Profile Along Streak")
        axs[1].set_xlabel("Normalized Distance Along Streak")
        axs[1].set_ylabel("Normalized Pixel Intensity")
        
        plt.tight_layout()
        plt.show()


    def display_final_results(self):
        """Displays the final detections on the original image."""
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(self.image_data, cmap='gray', origin='lower',
                      vmin=np.percentile(self.image_data, 1),
                      vmax=np.percentile(self.image_data, 99))
        for (x1, y1), (x2, y2) in self.streaks:
            ax.plot([x1, x2], [y1, y2], 'r-', lw=1.5, alpha=0.8)
        ax.set_title(f"Final Detections ({len(self.streaks)} found)")
        ax.set_xlim(0, self.image_data.shape[1])
        ax.set_ylim(0, self.image_data.shape[0])
        
        plt.tight_layout()
        plt.show()

    def run(self):
        """Executes the full detection and plotting pipeline."""
        try:
            self.load_image()
            self.detect_streaks_by_template()
            self.display_diagnostics_plot_1() 
            self.display_diagnostics_plot_2() 
            self._calculate_fft_intermediates()
            self.display_fourier_correlation_steps()
            self.display_line_profiles()
            self.display_final_results()
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    detector = FourierStreakDetector("9f2022f0-264a-4701-adf3-1495f64f67d1.fit")
    detector.run()