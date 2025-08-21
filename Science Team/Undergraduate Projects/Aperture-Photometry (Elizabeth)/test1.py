import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import label
from JG_Streaktools import streak_interface
import os
import csv

class FitsStreakDetector:
    def __init__(self, fits_path):
        self.fits_path = fits_path
        self.image_data = None
        self.streaks = []
        self.streak_interface = None
        self.results = []

    def load_image(self):
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

    def detect_streaks(self, sigma_threshold=0):
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
                continue  # Skip nearly circular features (likely stars)

            projections = np.dot(coords_centered, direction)
            min_proj = np.min(projections)
            max_proj = np.max(projections)

            endpoint1 = center + direction * min_proj
            endpoint2 = center + direction * max_proj

            x1, y1 = endpoint1
            x2, y2 = endpoint2
            self.streaks.append(((x1, y1), (x2, y2)))

    def analyze_and_plot(self):
        fig1, ax1 = plt.subplots(figsize=(12, 12))
        ax1.imshow(self.image_data, cmap='gray', origin='lower', vmin=np.percentile(self.image_data, 1), vmax=np.percentile(self.image_data, 99))
        ax1.set_title("Detected Streaks")

        # Get image dimensions for the boundary check
        height, width = self.image_data.shape

        for i, ((x1, y1), (x2, y2)) in enumerate(self.streaks):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            length = np.hypot(x2 - x1, y2 - y1)
            angle_rad = np.arctan2(y2 - y1, y2 - y1)

            
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
            profile_perp = self._sample_flux(image, x_perp, y_perp)

            x_along = np.linspace(x1, x2, int(length))
            y_along = np.linspace(y1, y2, int(length))
            profile_along = self._sample_flux(image, x_along, y_along)

            pixel_values = self._sample_flux(image, x_along, y_along)
            max_pixel = np.max(pixel_values)
            bscale = self.header.get('BSCALE', 1)
            bzero = self.header.get('BZERO', 0)
            saturation_level = 65535 * bscale + bzero

            saturation_warning = max_pixel >= saturation_level

            class StreakObj: pass
            streak = StreakObj()
            streak.cheatx = cx
            streak.cheaty = cy
            streak.L = length
            streak.theta = angle_rad
            streak.totalmag = 0  # Needed by simpill

            try:
                self.streak_interface.simpill(cx, cy, r=6.0, L=length, angle=angle_rad, streak=streak, visout=False)
                flux, flux_err = streak.pill_flux
                magzero, magzero_err = streak.pill_magzero
                self.results.append({
                    'index': i+1,
                    'center': (cx, cy),
                    'length': length,
                    'angle_rad': angle_rad,
                    'flux': flux,
                    'flux_err': flux_err,
                    'magzero': magzero,
                    'magzero_err': magzero_err,
                    'saturated': saturation_warning
                })
            except Exception as e:
                print(f"Streak {i+1}: Simpill failed - {e}")

        if not self.results:
            print("No valid streaks were successfully analyzed.")
            return

        fig2, axs = plt.subplots(len(self.results), 2, figsize=(10, 4 * len(self.results)))
        if len(self.results) == 1:
            axs = np.expand_dims(axs, axis=0)

        for idx, result in enumerate(self.results):
            ((x1, y1), (x2, y2)) = self.streaks[result['index'] - 1]
            cx, cy = result['center']
            length = result['length']
            angle_rad = result['angle_rad']
            perp_angle = angle_rad + np.pi / 2

            offset_range = int(length / 10)
            offsets = np.linspace(-offset_range, offset_range, 2 * offset_range + 1)
            x_perp = np.clip(cx + offsets * np.cos(perp_angle), 0, self.image_data.shape[1] - 1)
            y_perp = np.clip(cy + offsets * np.sin(perp_angle), 0, self.image_data.shape[0] - 1)
            profile_perp = self._sample_flux(self.image_data, x_perp, y_perp)

            x_along = np.clip(np.linspace(x1, x2, int(length)), 0, self.image_data.shape[1] - 1)
            y_along = np.clip(np.linspace(y1, y2, int(length)), 0, self.image_data.shape[0] - 1)
            profile_along = self._sample_flux(self.image_data, x_along, y_along)

            axs[idx][0].plot(offsets, profile_perp, color='green')
            axs[idx][0].set_title(f"Streak {result['index']} ⊥ Profile")
            axs[idx][0].set_xlabel("pixels")
            axs[idx][0].set_ylabel("Flux/Pixel Value")
            axs[idx][0].grid(True)

            axs[idx][1].plot(np.arange(len(profile_along)), profile_along, color='blue')
            axs[idx][1].set_title(f"Streak {result['index']} // Profile")
            axs[idx][1].set_xlabel("pixels")
            axs[idx][1].set_ylabel("Flux/Pixel Value")
            axs[idx][1].grid(True)

        fig2.tight_layout()
        plt.show()

    def save_results_to_csv(self, filename="streak_analysis_results.csv"):
        """Saves the collected streak analysis data to a CSV file."""
        if not self.results:
            print("No results to save to CSV.")
            return

        # Prepare the data for CSV writing by flattening the 'center' tuple
        processed_results = []
        for res in self.results:
            row = res.copy()
            row['center_x'] = f"{row['center'][0]:.2f}"
            row['center_y'] = f"{row['center'][1]:.2f}"
            # Round other values for cleaner output
            row['length'] = f"{row['length']:.2f}"
            row['angle_rad'] = f"{row['angle_rad']:.4f}"
            row['flux'] = f"{row['flux']:.2f}"
            row['flux_err'] = f"{row['flux_err']:.2f}"
            row['magzero'] = f"{row['magzero']:.4f}"
            row['magzero_err'] = f"{row['magzero_err']:.4f}"
            del row['center']  # Remove the original tuple field
            processed_results.append(row)
            
        # Define the exact order of columns for the CSV file
        fieldnames = [
            'index', 'center_x', 'center_y', 'length', 'angle_rad',
            'flux', 'flux_err', 'magzero', 'magzero_err', 'saturated'
        ]

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                # DictWriter handles writing list of dictionaries easily
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(processed_results)
            print(f"\n✅ Analysis results successfully saved to '{filename}'")
        except IOError as e:
            print(f"Error: Could not write results to CSV file. {e}")

    @staticmethod
    def _sample_flux(image, x_coords, y_coords):
        from scipy.ndimage import map_coordinates
        return map_coordinates(image, [y_coords, x_coords], order=1, mode='nearest')

    def run(self):
        self.load_image()
        self.detect_streaks(sigma_threshold=5.0)
        self.analyze_and_plot()
        self.save_results_to_csv("streak_analysis_results.csv")

if __name__ == '__main__':
    detector = FitsStreakDetector("Intelsat-40_G200_05s.fits")
    detector.run()
