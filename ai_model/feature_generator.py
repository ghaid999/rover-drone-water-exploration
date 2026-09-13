import rasterio
import numpy as np

# Real study area bounds (EPSG:32636), from the Land-based_Surveyed_area shapefile
REAL_MINX, REAL_MINY = 650205.58792499, 2560948.94033352
REAL_MAXX, REAL_MAXY = 705647.6370919, 2608246.34668777

# Shrink inward by this fraction on each side to avoid real raster no-data gaps
# near the boundary polygon's edges (empirically found: 15% eliminates ~100
# random-point failures observed at 0% margin down to 0).
MARGIN_FRAC = 0.15
_shrink_x = (REAL_MAXX - REAL_MINX) * MARGIN_FRAC
_shrink_y = (REAL_MAXY - REAL_MINY) * MARGIN_FRAC
MAPPED_MINX, MAPPED_MAXX = REAL_MINX + _shrink_x, REAL_MAXX - _shrink_x
MAPPED_MINY, MAPPED_MAXY = REAL_MINY + _shrink_y, REAL_MAXY - _shrink_y

# Simulated world bounds (matches the Gazebo world's 50x50m area)
SIM_MIN, SIM_MAX = -25.0, 25.0

RASTER_BASE = "/home/majed/rover_drone_project/data/groundwater_dataset/Shalatein Paper"
RASTERS = {
    "sentinel1_vv":      f"{RASTER_BASE}/Satelite raw data/Sentinel-1 VV band.tif",
    "sentinel1_vh":      f"{RASTER_BASE}/Satelite raw data/Sentinel-1 VH band.tif",
    "dem":               f"{RASTER_BASE}/Satelite raw data/ALOS DEM.tif",
    "slope":             f"{RASTER_BASE}/Satelite raw data/Slope.tif",
    "rainfall":          f"{RASTER_BASE}/Factors Raw maps/Rainfall.tif",
    "twi":               f"{RASTER_BASE}/Factors Raw maps/TWI.tif",
    "curvature":         f"{RASTER_BASE}/Factors Raw maps/Curvature.tif",
    "stream_density":    f"{RASTER_BASE}/Factors Raw maps/ Stream Density.tif",
    "lineament_density": f"{RASTER_BASE}/Factors Raw maps/LD.tif",
}
FEATURE_ORDER = list(RASTERS.keys())

_opened_rasters = None


def _get_opened_rasters():
    global _opened_rasters
    if _opened_rasters is None:
        _opened_rasters = {name: rasterio.open(path) for name, path in RASTERS.items()}
    return _opened_rasters


def sim_to_real(x, y):
    """Map a simulated-world coordinate to a real-world (EPSG:32636) coordinate,
    within a 15%-margin-shrunk version of the real surveyed area (see MARGIN_FRAC)."""
    real_x = MAPPED_MINX + (x - SIM_MIN) / (SIM_MAX - SIM_MIN) * (MAPPED_MAXX - MAPPED_MINX)
    real_y = MAPPED_MINY + (y - SIM_MIN) / (SIM_MAX - SIM_MIN) * (MAPPED_MAXY - MAPPED_MINY)
    return real_x, real_y


def real_to_sim(real_x, real_y):
    """Inverse mapping: real-world coordinate to simulated-world coordinate."""
    x = (real_x - MAPPED_MINX) / (MAPPED_MAXX - MAPPED_MINX) * (SIM_MAX - SIM_MIN) + SIM_MIN
    y = (real_y - MAPPED_MINY) / (MAPPED_MAXY - MAPPED_MINY) * (SIM_MAX - SIM_MIN) + SIM_MIN
    return x, y


def get_features_for_location(x, y, water_locations_path=None, rng=None):
    """
    Given a simulated (x, y) coordinate, maps it to a real geographic location
    within the (margin-shrunk) Shalatein study area and samples all nine
    trained-model features from the real raster data at that exact point.

    Signature kept identical to the earlier synthetic (Option 1) version so
    nothing in water_potential_scorer.py, drone_node.py, or rover_node.py
    needs to change.

    Returns (features_dict, valid). valid=False should now be rare (0% in a
    100-point test after the margin fix) but is still checked defensively.
    """
    real_x, real_y = sim_to_real(x, y)
    rasters = _get_opened_rasters()

    features = {}
    for name, src in rasters.items():
        val = float(next(src.sample([(real_x, real_y)]))[0])
        if src.nodata is not None and (val == src.nodata or val < -1e30):
            return None, False
        features[name] = val

    return features, True


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(1)
    n_invalid = 0
    n_test = 100
    for _ in range(n_test):
        x, y = rng.uniform(-25, 25), rng.uniform(-25, 25)
        _, valid = get_features_for_location(x, y)
        if not valid:
            n_invalid += 1
    print(f"Validation re-check: {n_invalid}/{n_test} invalid ({n_invalid}%)")

    print()
    print("--- Sim center (0, 0) ---")
    feats, valid = get_features_for_location(0.0, 0.0)
    print("Valid:", valid)
    for k, v in feats.items():
        print(f"  {k}: {v:.3f}")
