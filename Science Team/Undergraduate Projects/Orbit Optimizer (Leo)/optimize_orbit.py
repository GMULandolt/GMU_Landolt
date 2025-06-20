import numpy as np
import pandas as pd
import TLEconstructor   # must provide func(inc, eccen, raan)
import time

import STRAIGHTRUN as sr
from importlib import import_module, reload 

from settings import parameters
import numpy as np

def _count_unique_eclipse_nights(df_visible):
    """Return how many distinct UTC dates appear while Eclipse % != '0%'."""
    if "Time (EST)" not in df_visible.columns:
        return 0
    eclipse_rows = df_visible[df_visible["Eclipse %"] != "0%"]
    if eclipse_rows.empty:
        return 0
    dates = eclipse_rows["Time (EST)"].astype(str).str.slice(0, 10)
    return dates.drop_duplicates().shape[0]

def optimize_orbit(
        airmass=1.6,
        res=30,
        weight_per_night=20,
        w_rubin=0.25,
        w_mason=0.25,
        w_palomar=0.25,
        w_snifs=0.25,
        default_night_code=2,
        snifs_night_code=2
    ):
    """
    Grid‑search RAAN & inclination to maximize weighted visibility quality.

    Parameters
    ----------
    * airmass : float
        Airmass limit (default 1.6 → altitude ≈ 38.2°).
    * res : int
        Number of grid steps for RAAN & inclination each.
    * weight_per_night : float
        Bonus added for each unique eclipse night (average across telescopes).
    * w_rubin, w_mason, w_palomar, w_snifs : floats
        Weights (0‑1) assigned to each telescope’s visibility percentage.
        **Must sum to 1**, else ValueError.
    * default_night_code : int
        Maximum Skyfield `dark_twilight_day` code counted as “night”
        for Rubin, Mason, Palomar (default 2 = nautical twilight).
    * snifs_night_code : int
        Maximum code for SNIFS (default 3 = civil twilight).
    """
    # ----- validate weights -----
    weight_sum = w_rubin + w_mason + w_palomar + w_snifs
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Visibility weights must sum to 1.  Got {weight_sum:.3f}"
        )

    # ----- inclination grid −30° … +10° -----
    incl_list = np.linspace(-30*np.pi/180, 10*np.pi/180, res) # RES IS INCLUDED HERE
    incl_list = np.where(incl_list < 0, 2*np.pi + incl_list, incl_list)  # wrap
    # ----- RAAN grid (0–360°, full circle) -----
    raan_list = np.linspace(0, 2*np.pi, res)   # adjust if you want a narrower band

    # Altitude threshold from airmass
    z_lim = np.arccos(1/airmass)
    alt_threshold = 90 - np.degrees(z_lim)

    telescopes = ["Rubin", "Mason", "Palomar", "SNIFFS"]
    night_limits = {
        "Rubin": default_night_code,
        "Mason": default_night_code,
        "Palomar": default_night_code,
        "SNIFFS": snifs_night_code
    }
    weights = {
        "Rubin": w_rubin,
        "Mason": w_mason,
        "Palomar": w_palomar,
        "SNIFFS": w_snifs
    }

    best = None
    best_quality = -np.inf
    total = len(raan_list) * len(incl_list)
    processed = 0
    start = time.time()

    print("Scanning RAAN × inclination grid…")
    for raan in raan_list:
        for inc in incl_list:
            df = TLEconstructor.func(inc, 0, raan)

            vis_pct = {}
            nights  = {}
            for tel in telescopes:
                alt_col  = f"{tel} Alt (Deg)"
                time_col = f"{tel} TIME"

                # rows with telescope night ≤ night_limit
                night_mask = df[time_col] < night_limits[tel]
                total_night_pts = np.count_nonzero(night_mask)

                # visible: night_mask & altitude > threshold
                vis_mask = night_mask & (df[alt_col] > alt_threshold)
                visible_pts = np.count_nonzero(vis_mask)
                vis_pct[tel] = (visible_pts / total_night_pts * 100) if total_night_pts else 0

                nights[tel] = _count_unique_eclipse_nights(df.loc[vis_mask])

            # weighted mean visibility
            weighted_vis = sum(weights[t] * vis_pct[t] for t in telescopes)
            avg_nights   = np.mean([nights[t] for t in telescopes])

            quality = weighted_vis + weight_per_night * avg_nights
            if quality > best_quality:
                best_quality = quality
                best = dict(
                    RAAN_rad      = raan,
                    Incl_rad      = inc,
                    RAAN_deg      = np.degrees(raan),
                    Incl_deg      = np.degrees(inc if inc <= np.pi else inc - 2*np.pi),
                    weighted_vis  = weighted_vis,
                    avg_nights    = avg_nights,
                    vis_pct       = vis_pct,
                    eclipse_nights= nights,
                    quality       = quality
                )

            processed += 1
            if processed % 20 == 0:
                pct = processed / total * 100
                print(f"\r{pct:5.1f}% completed  best Q={best_quality:.2f}", end="", flush=True)

    print(f"\nDone in {time.time() - start:.1f} s")
    return best

if __name__ == "__main__":
    result = optimize_orbit(
        airmass=1.6,
        res=30, # EDIT
        weight_per_night=0, # EDIT
        w_rubin=0, w_mason=0, w_palomar=0, w_snifs=1
    )
    print("\nBest orbit found:")
    print(f"  RAAN  : {result['RAAN_deg']:.2f}°")
    print(f"  Incl. : {result['Incl_deg']:.2f}°")
    print(f"  Quality score : {result['quality']:.2f}")
    print(f"  Weighted vis  : {result['weighted_vis']:.2f}%")
    print(f"  Avg eclipse nights (all scopes): {result['avg_nights']:.2f}")
    print("  Per‑telescope visibility (%):", result["vis_pct"])
    print("  Per‑telescope eclipse nights :", result["eclipse_nights"])

    # 1) apply the optimal values to settings
    # parameters.inclo = np.deg2rad(result["Incl_deg"])
    # parameters.nodeo = np.deg2rad(result["RAAN_deg"])   # treated as desired longitude
    # parameters.ecco  = 0                                # match grid search
    inclo = np.deg2rad(result["Incl_deg"])
    nodeo = np.deg2rad(result["RAAN_deg"])   # treated as desired longitude
    ecco  = 0  
    
    # 2) convert desired longitude → inertial RAAN (same math as in STRAIGHTRUN)
    # gst_hours = parameters.start.gmst
    # gst_rad   = np.deg2rad(gst_hours * 15.0)
    # raan_inertial = (parameters.nodeo + gst_rad) % (2 * np.pi)
    
    # 3) run the plotting routine
    print("\nGenerating ground-track image with optimal orbit...")
    sr.plot_ground_track(inclo, ecco, nodeo)
