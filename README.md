# NASA Landolt — Script Repository

This repository contains scripts developed, maintained, and used by the GMU Landolt team for the **NASA Landolt mission**.

---

## Repository Organization

```
GMU_Landolt/
├── Science Team/          # All active code and projects
└── Deprecated/            # Code no longer in active use
```

Active work lives in `Science Team/`. Code that is no longer maintained should be moved to the mirrored structure in `Deprecated/`.

---

## Science Team — Project Folders

### `LandoltTLEGenerator.py`
Top-level TLE (Two-Line Element) generation script for the Landolt satellite. Used to construct TLEs from orbital parameters.

---

### `Aperture-Photometry (Elizabeth)/`
Scripts and notebooks for performing aperture photometry on satellite streak images to extract flux measurements.

| File | Description |
|------|-------------|
| `flux_counts.py` | Computes flux counts from satellite images |
| `flux_counts_jn.ipynb` | Jupyter notebook walkthrough of flux count pipeline |
| `image-sim-updated.py` | Updated image simulation script |
| `image-sim-updated-jn.ipynb` | Jupyter notebook for image simulation |
| `fullrun.ipynb` | End-to-end pipeline notebook |
| `Streak_Detector_Analyzer.py` | Detects and analyzes satellite streaks in images |
| `JG_Streaktools.py` | Utility functions for streak detection (John Gizis tools) |
| `TLEconstructor.py` | Constructs TLEs for use in photometry pipeline |
| `observatories.py` | Observatory site definitions and parameters |
| `convenience_functions.py` | Shared utility functions |
| `trippy_utils.py` | Utilities wrapping the TRIPPY photometry package |
| `settings.py` / `settings.json` | Configuration parameters |

---

### `ETC (Angelle-Dawn-Daniel)/`
Exposure Time Calculator (ETC) for the Landolt mission. In development.

---

### `Half_Sidereal_Rate_TLE (Alan)/`
Tools for generating and understanding half-sidereal-rate TLEs for telescope tracking.

| File | Description |
|------|-------------|
| `half_rate_tle.py` | Generates half-sidereal-rate TLEs |
| `half_rate_tle_tutorial.ipynb` | Notebook tutorial and documentation |

---

### `Image_Code (Elizabeth)/`
Early image simulation code for modeling satellite observations.

| File | Description |
|------|-------------|
| `Image_sim.py` | Image simulation script |

---

### `JPL_Horizons (Aiden)/`
Scripts for querying and processing JPL Horizons ephemeris data.

| File | Description |
|------|-------------|
| `SPK/horizons.py` | Queries JPL Horizons API and processes SPK kernel output |

---

### `Orbit Optimizer (Leo)/`
Orbit optimization tools for finding ideal orbital parameters to maximize observatory observability.

| File | Description |
|------|-------------|
| `optimize_orbit.py` | Main orbit optimization script |
| `score.py` | Scoring function for evaluating orbit quality |
| `STRAIGHTRUN.py` | Entry point for running the full optimization pipeline |
| `runbaby.py` | Lightweight run script |
| `TLEconstructor.py` | TLE construction utility |
| `2dplotter.py` | 2D ground track plotter |
| `settings.py` / `settings.json` | Configuration parameters |

---

### `Orbit Propagation (Aiden-Dawn)/`
Core orbit propagation pipeline used to compute satellite visibility windows from ground-based observatories.

| File | Description |
|------|-------------|
| `STRAIGHTRUN.py` | Entry point to run the full propagation pipeline |
| `runbaby.py` | Standard propagation run script |
| `runbaby-subgeo.py` | Propagation run script for sub-GEO orbits |
| `TLEconstructor.py` | Constructs TLEs for propagation |
| `TLEconstructor2.py` | Updated TLE constructor |
| `plotter.py` | Generates visibility and parameter plots |
| `2dplotter.py` | 2D sky / ground track plotter |
| `2dplotter-subgeo.py` | 2D plotter for sub-GEO orbits |
| `ra-dec-vis.py` | Plots RA/Dec visibility windows |
| `settings.py` / `settings.json` | Configuration parameters |
| `orbits/` | Pre-computed orbit output CSV files for GMU, Palomar, Rubin, and SNIFS observatories |
| `output/` | Observation parameter output files |

---

### `Orbit-Flux-Image Sim (Aiden-Dawn)/`
Combined pipeline integrating orbit propagation, flux modeling, and image simulation.

| File | Description |
|------|-------------|
| `flux_counts.py` | Flux count calculation from orbit propagation output |
| `flux_counts_jn.ipynb` | Notebook walkthrough of flux pipeline |
| `image-sim-updated.py` | Image simulation using orbit and flux data |
| `image-sim-updated-jn.ipynb` | Notebook for image simulation |
| `fullrun.ipynb` | End-to-end combined pipeline notebook |
| `TLEconstructor.py` | TLE construction utility |
| `convenience_functions.py` | Shared utility functions |
| `settings.py` / `settings.json` | Configuration parameters |

---

### `OrbitSIm (Aiden)/`
Earlier orbit simulation code developed during initial mission planning phases. Retained for historical reference.

---

### `Streak-Detection (Elizabeth - Chapin)/`
Satellite streak detection in telescope images, with separate implementations by Elizabeth and Chapin.

**`SRC_Chapin/`**

| File | Description |
|------|-------------|
| `Streak_Detector_Analyzer.py` | Main streak detection and analysis script |
| `image_calibrator.py` | Calibrates raw images before processing |
| `satprocessing.py` | Satellite image processing utilities |
| `streakprocessing.py` | Streak-specific processing routines |
| `config.json` | Configuration file |

**`SRC_Elizabeth/`**

| File | Description |
|------|-------------|
| `Streak_Detector.py` | Elizabeth's streak detection implementation |
| `JG_Streaktools.py` | John Gizis streak utility functions |

---

### `scheduling_algorithm (Alex)/`
Observatory scheduling algorithm for planning Landolt satellite observations across multiple ground stations.

| File | Description |
|------|-------------|
| `scheduler.py` | Main scheduling entry point |
| `long_term_algorithm.py` | Long-term (multi-night) scheduling logic |
| `short_term_algorithm.py` | Short-term (single-night) scheduling logic |
| `simulator.py` | Simulates scheduled observation sessions |
| `priority_calculator.py` | Calculates observation priority scores |
| `observatory.py` | Observatory object definition |
| `observatory_availability.py` | Checks observatory availability windows |
| `observatory_calibrations.py` | Handles calibration constraints |
| `observatory_characteristics.py` | Stores observatory site properties |
| `observatory_forecast.py` | Integrates weather forecasts into scheduling |
| `observatory_init.py` | Initializes observatory objects |
| `observatory_repository.py` | Reads/writes observatory data from CSV files |
| `satellitetracker.py` | Tracks satellite position for scheduling |
| `weather_conditions_checking.py` | Weather condition evaluation |
| `observatoryFileRepository/` | CSV files for observatory availability, characteristics, and schedules |

---

## Documentation

For detailed documentation on a specific project, refer to the associated Jupyter Notebook or any README files within that project's folder. If no documentation exists for a project, please create some for future contributors.

---

## Git Workflow

```bash
# Before starting work, always pull latest changes
git pull origin main --allow-unrelated-histories

# Stage all changes
git add -A

# Commit
git commit -m "description of changes"

# Push when finished
git push origin main
```

> Note: `.fits` files are excluded from version control via `.gitignore` due to their large size. Store large data files locally or use an external data store.

---

## Team

This repository is shared across the entire Landolt team — undergraduates, graduates, senior engineers, and scientists. Please respect the work of all contributors and help maintain a clean and organized codebase.

*Thank you for contributing to the success of the Landolt mission!*
