import geopandas as gpd
import rasterio
import numpy as np
import random
import csv

RASTERS = {
    "sentinel1_vv": "Shalatein Paper/Satelite raw data/Sentinel-1 VV band.tif",
    "sentinel1_vh": "Shalatein Paper/Satelite raw data/Sentinel-1 VH band.tif",
    "dem": "Shalatein Paper/Satelite raw data/ALOS DEM.tif",
    "slope": "Shalatein Paper/Satelite raw data/Slope.tif",
    "rainfall": "Shalatein Paper/Factors Raw maps/Rainfall.tif",
    "twi": "Shalatein Paper/Factors Raw maps/TWI.tif",
    "curvature": "Shalatein Paper/Factors Raw maps/Curvature.tif",
    "stream_density": "Shalatein Paper/Factors Raw maps/ Stream Density.tif",
    "lineament_density": "Shalatein Paper/Factors Raw maps/LD.tif",
}

# 1. Load real wells (presence points)
wells = gpd.read_file("Shalatein Paper/Infrastructures/Groundwater_Wells.shp")
presence_points = [(geom.x, geom.y) for geom in wells.geometry]
print(f"Loaded {len(presence_points)} real wells (presence points)")

# 2. Load surveyed area boundary
area = gpd.read_file("Shalatein Paper/Infrastructures/Land-based_Surveyed_area.shp")
boundary = area.geometry.iloc[0]
minx, miny, maxx, maxy = area.total_bounds

# 3. Generate pseudo-absence points: random, inside boundary, far from real wells
MIN_DIST_FROM_WELL = 1000  # meters
N_ABSENCE = len(presence_points)  # match count for balance

def far_enough(x, y):
    for px, py in presence_points:
        if ((x - px) ** 2 + (y - py) ** 2) ** 0.5 < MIN_DIST_FROM_WELL:
            return False
    return True

absence_points = []
random.seed(42)
attempts = 0
while len(absence_points) < N_ABSENCE and attempts < 20000:
    x = random.uniform(minx, maxx)
    y = random.uniform(miny, maxy)
    attempts += 1
    from shapely.geometry import Point
    pt = Point(x, y)
    if boundary.contains(pt) and far_enough(x, y):
        absence_points.append((x, y))

print(f"Generated {len(absence_points)} pseudo-absence points (after {attempts} attempts)")

# 4. Sample every raster at every point
def sample_rasters(points):
    results = []
    opened = {name: rasterio.open(path) for name, path in RASTERS.items()}
    for x, y in points:
        row = {}
        for name, src in opened.items():
            try:
                for val in src.sample([(x, y)]):
                    row[name] = float(val[0])
            except Exception:
                row[name] = None
        results.append(row)
    for src in opened.values():
        src.close()
    return results

presence_features = sample_rasters(presence_points)
absence_features = sample_rasters(absence_points)

# 5. Write combined CSV
feature_names = list(RASTERS.keys())
with open("groundwater_feature_table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y"] + feature_names + ["label"])
    for (x, y), feats in zip(presence_points, presence_features):
        writer.writerow([x, y] + [feats[n] for n in feature_names] + [1])
    for (x, y), feats in zip(absence_points, absence_features):
        writer.writerow([x, y] + [feats[n] for n in feature_names] + [0])

print("Saved groundwater_feature_table.csv")
