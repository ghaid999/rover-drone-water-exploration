import geopandas as gpd
import rasterio
import random
import csv
from shapely.geometry import Point

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

opened = {name: rasterio.open(path) for name, path in RASTERS.items()}

def sample_point(x, y):
    row = {}
    for name, src in opened.items():
        val = float(next(src.sample([(x, y)]))[0])
        nodata = src.nodata
        if nodata is not None and (val == nodata or val < -1e30):
            return None  # invalid point, at least one layer has no real data here
        row[name] = val
    return row

wells = gpd.read_file("Shalatein Paper/Infrastructures/Groundwater_Wells.shp")
presence_points = [(geom.x, geom.y) for geom in wells.geometry]
print(f"Loaded {len(presence_points)} real wells")

# Validate presence points too -- if any well itself sits on a no-data pixel, we need to know
presence_features = []
valid_presence_points = []
for x, y in presence_points:
    feats = sample_point(x, y)
    if feats is not None:
        presence_features.append(feats)
        valid_presence_points.append((x, y))
    else:
        print(f"WARNING: well at ({x:.1f}, {y:.1f}) has no-data in at least one raster -- excluded")

print(f"{len(valid_presence_points)} of {len(presence_points)} wells have fully valid data")

area = gpd.read_file("Shalatein Paper/Infrastructures/Land-based_Surveyed_area.shp")
boundary = area.geometry.iloc[0]
minx, miny, maxx, maxy = area.total_bounds

MIN_DIST_FROM_WELL = 1000
N_ABSENCE = len(valid_presence_points)

def far_enough(x, y):
    for px, py in valid_presence_points:
        if ((x - px) ** 2 + (y - py) ** 2) ** 0.5 < MIN_DIST_FROM_WELL:
            return False
    return True

absence_points = []
absence_features = []
random.seed(42)
attempts = 0
while len(absence_points) < N_ABSENCE and attempts < 50000:
    x = random.uniform(minx, maxx)
    y = random.uniform(miny, maxy)
    attempts += 1
    pt = Point(x, y)
    if boundary.contains(pt) and far_enough(x, y):
        feats = sample_point(x, y)
        if feats is not None:  # only accept if ALL rasters have real data here
            absence_points.append((x, y))
            absence_features.append(feats)

print(f"Generated {len(absence_points)} valid pseudo-absence points (after {attempts} attempts)")

for src in opened.values():
    src.close()

feature_names = list(RASTERS.keys())
with open("groundwater_feature_table_v2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y"] + feature_names + ["label"])
    for (x, y), feats in zip(valid_presence_points, presence_features):
        writer.writerow([x, y] + [feats[n] for n in feature_names] + [1])
    for (x, y), feats in zip(absence_points, absence_features):
        writer.writerow([x, y] + [feats[n] for n in feature_names] + [0])

print("Saved groundwater_feature_table_v2.csv")
