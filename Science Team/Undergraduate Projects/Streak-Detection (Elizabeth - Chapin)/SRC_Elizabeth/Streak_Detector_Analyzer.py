import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import label
from JG_Streaktools import streak_interface
from numba import njit
import os
import csv

#---------- STREAK DETECTOR ----------#

class FitsStreakDetector:
    def __init__(self, fits_path):
        self.fits_path = fits_path
        self.image_data = None
        self.streaks = []
        self.streak_interface = None
        self.results = []

    def load_image(self):
        """Loads FITS image data and initializes the streak interface."""
        if not os.path.exists(self.fits_path):
            raise FileNotFoundError(f"FITS file not found: {self.fits_path}")

        with fits.open(self.fits_path) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    self.header = hdu.header
                    break

        if data.ndim == 3:
            data = data[0]  # Use first slice if 3D

        self.image_data = data.astype(np.float32)
        self.streak_interface = streak_interface(self.image_data)


    def detect_streaks(self, sigma_threshold=0, min_length_pixels=20):
        """Detects linear streaks in the image using statistical thresholding.
        
        Args:
            sigma_threshold (float): The number of standard deviations above the median
                                     to set the pixel brightness threshold.
            min_length_pixels (int): The minimum end-to-end length for a feature
                                     to be considered a streak.
                                     
        """
        median = np.median(self.image_data)
        std = np.std(self.image_data)
        threshold = median + sigma_threshold * std

        mask = self.image_data > threshold
        labeled, num_features = label(mask)

        for label_id in range(1, num_features + 1):
            coords = np.argwhere(labeled == label_id)
            if len(coords) < 10:
                continue

            coords = coords[:, [1, 0]]  # swap to (x, y)
            center = np.mean(coords, axis=0)
            coords_centered = coords - center
            cov = np.cov(coords_centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eig(cov)
            direction = eigvecs[:, np.argmax(eigvals)]

            elongation = np.sqrt(eigvals.max()) / np.sqrt(eigvals.min() + 1e-6)
            if elongation < 5.0:
                continue  # Skip nearly circular features (likely satellites)

            projections = np.dot(coords_centered, direction)
            min_proj = np.min(projections)
            max_proj = np.max(projections)

            endpoint1 = center + direction * min_proj
            endpoint2 = center + direction * max_proj

            x1, y1 = endpoint1
            x2, y2 = endpoint2

            length = np.hypot(x2 - x1, y2 - y1)
            if length < min_length_pixels:
                continue

            self.streaks.append(((x1, y1), (x2, y2)))

        #plot binary mask with detected streaks
        print(f"Detected {len(self.streaks)} streak(s) in the image.")
        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap='gray', origin='lower')
        for ((x1, y1), (x2, y2)) in self.streaks:
            plt.plot([x1, x2], [y1, y2], 'g-', lw=1.5)
        plt.title("Detected Streaks")
        plt.show()


    def analyze_streak(self):
        """Analyzes each detected streak using pill aperture photometry, then plots results."""
        
        # Set up plots
        fig1, ax1 = plt.subplots(figsize=(12, 12))
        ax1.imshow(self.image_data, cmap='gray', origin='lower', vmin=np.percentile(self.image_data, 1), vmax=np.percentile(self.image_data, 99))
        ax1.set_title("Detected Streaks and Apertures")

        # Get image dimensions for the boundary check
        height, width = self.image_data.shape

        # --- START: STREAK ANALYSIS ---

        for i, ((x1, y1), (x2, y2)) in enumerate(self.streaks):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            length = np.hypot(x2 - x1, y2 - y1)
            angle_rad = np.arctan2(y2 - y1, x2 - x1)

            
            # Define a safety margin based on the pill aperture's radius.
            # The 'r' parameter passed to simpill is 6.0.
            margin = 6.0

            # Check if the streak's bounding box is too close to any edge.
            if (min(x1, x2) < margin or max(x1, x2) >= width - margin or
                min(y1, y2) < margin or max(y1, y2) >= height - margin):
                print(f"⚠️ Streak {i+1}: Skipped analysis because it is too close to the image edge.")
                # Use 'continue' to skip the rest of the loop for this streak
                continue
            

            ax1.plot([x1, x2], [y1, y2], 'r-', lw=1.5)
            ax1.plot(cx, cy, 'rx')

            image = self.image_data
            perp_angle = angle_rad + np.pi / 2
            offset_range = int(length / 10)
            offsets = np.linspace(-offset_range, offset_range, 2 * offset_range + 1)

            x_perp = cx + offsets * np.cos(perp_angle)
            y_perp = cy + offsets * np.sin(perp_angle)
            profile_perp = self.sample_flux(image, x_perp, y_perp)

            x_along = np.linspace(x1, x2, int(length))
            y_along = np.linspace(y1, y2, int(length))
            profile_along = self.sample_flux(image, x_along, y_along)

            pixel_values = self.sample_flux(image, x_along, y_along)
            max_pixel = np.max(pixel_values)
            bscale = self.header.get('BSCALE', 1)
            bzero = self.header.get('BZERO', 0)
            saturation_level = 65535 * bscale + bzero

            saturation_warning = max_pixel >= saturation_level

            num_slices = 5  # Number of perpendicular slices to take along each streak
            fwhm_values = []
            slice_profiles = []
            
            # Generate points along the streak to take slices from
            slice_points_x = np.linspace(x1, x2, num_slices)
            slice_points_y = np.linspace(y1, y2, num_slices)

            perp_angle = angle_rad + np.pi / 2
            offset_range = 15 # Use a fixed pixel range for all slices
            offsets = np.linspace(-offset_range, offset_range, 2 * offset_range + 1)

            for slice_cx, slice_cy in zip(slice_points_x, slice_points_y):
                # Calculate coordinates for the perpendicular slice
                x_perp = np.clip(slice_cx + offsets * np.cos(perp_angle), 0, width - 1)
                y_perp = np.clip(slice_cy + offsets * np.sin(perp_angle), 0, height - 1)
                
                # Get the flux profile and calculate its FWHM
                profile = self.sample_flux(self.image_data, x_perp, y_perp)
                fwhm = self.calculate_fwhm(offsets, profile)
                
                slice_profiles.append(profile)
                if fwhm > 0:
                    fwhm_values.append(fwhm)
            
            avg_fwhm = np.mean(fwhm_values) if fwhm_values else 0

            class StreakObj: pass
            streak = StreakObj()
            streak.cheatx = cx
            streak.cheaty = cy
            streak.L = length
            streak.theta = angle_rad
            streak.totalmag = 0  # Needed by simpill

            try:
                self.streak_interface.simpill(cy, cx, r=6.0, L=length, angle=angle_rad, streak=streak, visout=False)
                flux, flux_err = streak.pill_flux
                magzero, magzero_err = streak.pill_magzero
                
                #Plot the pill aperture
                ax1.contour(self.streak_interface.pill, levels=[0.5], colors='Maroon', linestyles='--', linewidths=1.5)
                ax1.plot(cx, cy, 'rx', markersize=4)

                # Create sky annulus mask
                photometry_mask = self.streak_interface.pill
                annulus_inner_r, annulus_outer_r = 6.0, 10.0
                
                inner_annulus_mask = self.create_pill_mask(self.image_data.shape, cy, cx, length, annulus_inner_r, angle_rad)
                outer_annulus_mask = self.create_pill_mask(self.image_data.shape, cy, cx, length, annulus_outer_r, angle_rad)
                sky_mask = outer_annulus_mask & ~inner_annulus_mask

                # Calculate median sky background
                sky_pixels = self.image_data[sky_mask]
                sky_median = np.median(sky_pixels)
                num_phot_pixels = np.sum(photometry_mask)

                # Background-subtracted flux
                raw_flux, flux_err = streak.pill_flux
                bkg_subtracted_flux = raw_flux - (num_phot_pixels * sky_median)

                # Signal-to-noise ratio
                snr = bkg_subtracted_flux / flux_err if flux_err > 0 else 0

                # Overlay the photometry aperture and sky annulus
                ax1.contour(photometry_mask, levels=[0.5], colors='Maroon', linestyles='--', linewidths=1.5,)
                ax1.contour(sky_mask, levels=[0.5], colors='Cyan', linestyles=':', linewidths=1.5,)
    

                self.results.append({
                    'index': i+1,
                    'center': (cx, cy),
                    'length': length,
                    'angle_rad': angle_rad,
                    'flux': flux,
                    'flux_err': flux_err, 
                    'sky_median': sky_median,
                    'bkg_subtracted_flux': bkg_subtracted_flux,
                    'avg_fwhm': avg_fwhm,
                    'magzero': magzero,
                    'magzero_err': magzero_err,
                    'snr': snr,
                    'saturated': saturation_warning
                })
            except Exception as e:
                print(f"Streak {i+1}: Simpill failed - {e}")

        if not self.results:
            print("No valid streaks were successfully analyzed.")
            return
        
        # --- END: STREAK ANALYSIS ---

        # --- BEGIN: PROFILE PLOTTING ---

        fig2, axs = plt.subplots(len(self.results), 2, figsize=(10, 5 * len(self.results)))
        if len(self.results) == 1:
            axs = np.expand_dims(axs, axis=0)

        
        for idx, result in enumerate(self.results):
            # Plot the multiple perpendicular slices

            ((x1, y1), (x2, y2)) = self.streaks[result['index'] - 1]
            angle_rad = result['angle_rad']
            perp_angle = angle_rad + np.pi / 2
            slice_points_x = np.linspace(x1, x2, num_slices)
            slice_points_y = np.linspace(y1, y2, num_slices)

            for i, (slice_cx, slice_cy) in enumerate(zip(slice_points_x, slice_points_y)):
                x_perp = np.clip(slice_cx + offsets * np.cos(perp_angle), 0, width - 1)
                y_perp = np.clip(slice_cy + offsets * np.sin(perp_angle), 0, height - 1)
                profile = self.sample_flux(self.image_data, x_perp, y_perp)

                # The center slice (index 2 for 5 slices) is plotted as a heavier line
                if i == num_slices // 2:
                    axs[idx][0].plot(offsets, profile, color='green', alpha=1.0, lw=2.0, label='Center Slice')
                else: # Other slices are lighter
                    axs[idx][0].plot(offsets, profile, color='green', alpha=0.4, lw=1.0)
            
            axs[idx][0].set_title(f"Streak {result['index']} ⊥ Profiles (Avg FWHM: {result['avg_fwhm']:.2f})")
            axs[idx][0].set_xlabel("Offset from Centerline (pixels)")
            axs[idx][0].legend()
            
            # Plot the parallel profile
            x_along = np.clip(np.linspace(x1, x2, int(result['length'])), 0, width - 1)
            y_along = np.clip(np.linspace(y1, y2, int(result['length'])), 0, height - 1)
            profile_along = self.sample_flux(self.image_data, x_along, y_along)
            axs[idx][1].plot(profile_along, color='blue')
            axs[idx][1].set_title(f"Streak {result['index']} // Profile")
       
        
        fig1.tight_layout()
        fig2.tight_layout()
        plt.show()
     # --- END: PROFILE PLOTTING ---


    def save_results_to_csv(self, filename="streak_analysis_results.csv"):
        """Saves the collected streak analysis data to a CSV file."""
        if not self.results:
            print("No results to save to CSV.")
            return

        # Process results to format float values
        processed_results = []
        for res in self.results:
            row = res.copy()
            row['center_x'] = f"{row['center'][0]:.4f}"
            row['center_y'] = f"{row['center'][1]:.4f}"
            row['length'] = f"{row['length']:.4f}"
            row['angle_rad'] = f"{row['angle_rad']:.4f}"
            row['flux'] = f"{row['flux']:.4f}"
            row['flux_err'] = f"{row['flux_err']:.4f}"
            row['avg_fwhm'] = f"{row['avg_fwhm']:.4f}"
            row['sky_median'] = f"{row['sky_median']:.4f}"
            row['bkg_subtracted_flux'] = f"{row['bkg_subtracted_flux']:.4f}"
            row['magzero'] = f"{row['magzero']:.4f}"
            row['snr'] = f"{row['snr']:.2f}"
            row['magzero_err'] = f"{row['magzero_err']:.4f}"
            del row['center'] 
            processed_results.append(row)
            
        # Define the exact order of columns for the CSV file
        fieldnames = [
            'index', 'center_x', 'center_y', 'length', 'angle_rad',
            'flux', 'flux_err', 'avg_fwhm','sky_median','bkg_subtracted_flux','magzero',
            'magzero_err', 'snr', 'saturated'
        ]
        # Write to CSV
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                # DictWriter handles writing list of dictionaries easily
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(processed_results)
            print(f"\n✅ Analysis results successfully saved to '{filename}'")
        except IOError as e:
            print(f"Error: Could not write results to CSV file. {e}")


#---------- PHYSICS/ASTRONOMY UTILITIES ----------#


    @staticmethod
    def sample_flux(image, x_coords, y_coords):
        """Samples pixel values at given (x, y) coordinates using bilinear interpolation."""
        from scipy.ndimage import map_coordinates
        return map_coordinates(image, [y_coords, x_coords], order=1, mode='nearest')


    @staticmethod
    def calculate_fwhm(x_coords, profile):
        """Calculates the Full Width at Half Maximum of a 1D profile."""
        try:
            # Normalize the profile by subtracting the baseline
            baseline = np.min(profile)
            data = profile - baseline
            peak = np.max(data)
            if peak <= 0: return 0

            half_max = peak / 2.0
            
            # Find all indices where the data is above the half-max value
            above_indices = np.where(data > half_max)[0]
            if len(above_indices) < 2:
                return 0  # Peak is too narrow or ill-defined

            # Interpolate to find the left and right crossing points for sub-pixel accuracy
            left_index = above_indices[0]
            right_index = above_indices[-1]

            # Left side interpolation
            p1 = data[left_index - 1]
            p2 = data[left_index]
            x1 = x_coords[left_index - 1]
            x2 = x_coords[left_index]
            left_cross = x1 + (x2 - x1) * (half_max - p1) / (p2 - p1)
            
            # Right side interpolation
            p1 = data[right_index]
            p2 = data[right_index + 1]
            x1 = x_coords[right_index]
            x2 = x_coords[right_index + 1]
            right_cross = x1 + (x2 - x1) * (half_max - p1) / (p2 - p1)

            return abs(right_cross - left_cross)
        except (ValueError, IndexError):
            # Handle cases where the peak is at the very edge of the profile
            return 0


    @staticmethod
    def create_pill_mask(image_shape, cx, cy, L, r, angle_rad):
        """
        Creates a boolean mask for a pill-shaped aperture, replicating the
        conventions of the JG_Streaktools library for consistency.
        """
        height, width = image_shape
        y_coords, x_coords = np.ogrid[:height, :width]

        # Replicate the conventions from JG_Streaktools to ensure alignment
        xc_swapped, yc_swapped = cy, cx
        angle_cw = angle_rad
        
        # Replicate the internal angle transformation
        cos_transformed = np.cos(np.pi / 2 - angle_cw)
        sin_transformed = -np.sin(np.pi / 2 - angle_cw)

        # Perform the coordinate rotation (replicating the x/y swap bug)
        xp = (y_coords - yc_swapped) * cos_transformed - (x_coords - xc_swapped) * sin_transformed
        yp = (x_coords - xc_swapped) * cos_transformed + (y_coords - yc_swapped) * sin_transformed

        # Check if pixels are within the pill's rectangular body or its circular ends
        in_rectangle = (np.abs(yp) <= r) & (np.abs(xp) <= L / 2)
        in_endcap1 = np.sqrt((xp + L / 2)**2 + yp**2) <= r
        in_endcap2 = np.sqrt((xp - L / 2)**2 + yp**2) <= r
        
        return in_rectangle | in_endcap1 | in_endcap2



# ----------- RUNNING THE MODULE ----------- #

    # Run the Module
    def run(self):
        self.load_image()
        self.detect_streaks(sigma_threshold=3.0, min_length_pixels=50)
        #self.analyze_streak()
        #self.save_results_to_csv("streak_analysis_results.csv")


    # Starting point
if __name__ == '__main__':
    detector = FitsStreakDetector("9f2022f0-264a-4701-adf3-1495f64f67d1.fit")
    detector.run()
