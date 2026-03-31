import numpy as np
import pandas as pd
import math
import TLEconstructor   # This module must define the function func(inc, eccen, node)

def orbit_score(raan, inc, eccen=0, airmass=2.0):
    """
    Evaluate how "good" a geosynchronous orbit is based on its visibility metrics 
    at multiple telescope locations (Rubin, Mason, Palomar, and SNIFS).
    
    The function runs the orbit simulation with TLEconstructor.func(inc, eccen, raan)
    and then calculates:
      - The percentage of time the satellite is above the altitude corresponding to
        an airmass of `airmass` (nighttime condition is indicated by TIME < 2).
      - The number of unique eclipse nights (i.e. distinct dates when the satellite is in eclipse)
        at each telescope.

    Parameters:
      raan   : Right Ascension of Ascending Node in radians.
      inc    : Inclination in radians.
      eccen  : Eccentricity (default 0 for a circular orbit).
      airmass: Desired maximum airmass threshold (default is 1.6).

    Returns:
      A dictionary containing:
         - Input parameters both in radians and degrees.
         - The threshold altitude (in degrees) corresponding to the airmass.
         - For each telescope, the visibility percentage and number of unique eclipse nights.
         - Average visibility percentage and average eclipse nights across all telescopes.
         - Raw counts of visible points vs. total night time points used.
    """
    # Calculate the altitude threshold corresponding to the desired airmass.
    # For airmass X, cos(z) = 1 / X, so the zenith angle is z, and the altitude = 90 - z (in degrees).
    z = math.acos(1/airmass)
    threshold_alt = 90 - np.rad2deg(z)
    
    # Run the orbit simulation to get a DataFrame of the satellite's behavior.
    df = TLEconstructor.func(inc, eccen, raan)
    
    # Prepare to accumulate results for each telescope.
    telescopes = ["Rubin", "Mason", "Palomar", "SNIFS"]
    results = {}
    
    for tel in telescopes:
        alt_col = f"{tel} Alt (Deg)"
        time_col = f"{tel} TIME"
        
        # Filter to times when the telescope is in "night" (TIME < 2).
        # Match runbaby.py logic: use NumPy size counting on conditions
        time_array = df[time_col].to_numpy()
        alt_array = df[alt_col].to_numpy()
        eclipse_array = df["Eclipse %"].to_numpy()

        # Nighttime condition
        night_mask = time_array < 2
        total_night_points = np.count_nonzero(night_mask)

        # Visible during night and above altitude threshold
        vis_mask = (time_array < 2) & (alt_array > threshold_alt)
        visible_points = np.count_nonzero(vis_mask)
        vis_percentage = (visible_points / total_night_points * 100) if total_night_points > 0 else 0

        # Eclipse condition within visible points
        eclipse_mask = vis_mask & (eclipse_array != "0%")
        eclipse_indices = np.where(eclipse_mask)[0]

        if "Time (EST)" in df.columns:
            eclipse_dates = pd.Series(df["Time (EST)"].iloc[eclipse_indices]).astype(str).str.slice(0, 10)
            eclipse_nights = eclipse_dates.drop_duplicates().shape[0]
        else:
            eclipse_nights = 0

        results[tel] = {
            "visibility_percentage": vis_percentage,
            "eclipse_nights": eclipse_nights,
            "visible_points": visible_points,
            "total_night_points": total_night_points
        }

    
    # Compute average metrics across telescopes.
    avg_visibility = np.mean([results[tel]["visibility_percentage"] for tel in telescopes])
    avg_eclipse_nights = np.mean([results[tel]["eclipse_nights"] for tel in telescopes])
    
    # Build the output dictionary.
    out = {
        "RAAN_rad": raan,
        "Inclination_rad": inc,
        "RAAN_deg": np.degrees(raan),
        "Inclination_deg": np.degrees(inc),
        "threshold_altitude_deg": threshold_alt,
        "telescopes": results,
        "avg_visibility_percentage": avg_visibility,
        "avg_eclipse_nights": avg_eclipse_nights
    }
    
    return out

# Example usage if running score.py directly:
if __name__ == "__main__":
    # For example, input RAAN and inclination in radians.
    # Adjust these values as needed.
    raan = np.deg2rad(226) 
    inc = np.deg2rad(-1) 
    	
    score = orbit_score(raan, inc)
    # Pretty-print the output.
    print("Orbit Score Results:")
    print(f"RAAN: {score['RAAN_deg']:.2f} deg, Inclination: {score['Inclination_deg']:.2f} deg")
    print(f"Threshold Altitude (for airmass 2.0): {score['threshold_altitude_deg']:.2f} deg")
    for tel, metrics in score["telescopes"].items():
        print(f"\nTelescopes: {tel}")
        print(f"  Visibility: {metrics['visibility_percentage']:.1f}% "
              f"({metrics['visible_points']} / {metrics['total_night_points']})")
        print(f"  Eclipse Nights: {metrics['eclipse_nights']}")
    print(f"\nAverage Visibility: {score['avg_visibility_percentage']:.1f}%")
    print(f"Average Eclipse Nights: {score['avg_eclipse_nights']:.2f}")
