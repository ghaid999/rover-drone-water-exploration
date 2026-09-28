import geopandas as gpd
import rasterio
import numpy as np
import random
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wilcoxon

BASE = "Shalatein Paper"

FEATURE_RASTERS = {
    "sentinel1_vv":      f"{BASE}/Satelite raw data/Sentinel-1 VV band.tif",
    "sentinel1_vh":      f"{BASE}/Satelite raw data/Sentinel-1 VH band.tif",
    "dem":               f"{BASE}/Satelite raw data/ALOS DEM.tif",
    "slope":             f"{BASE}/Satelite raw data/Slope.tif",
    "rainfall":          f"{BASE}/Factors Raw maps/Rainfall.tif",
    "twi":               f"{BASE}/Factors Raw maps/TWI.tif",
    "curvature":         f"{BASE}/Factors Raw maps/Curvature.tif",
    "stream_density":    f"{BASE}/Factors Raw maps/ Stream Density.tif",
    "lineament_density": f"{BASE}/Factors Raw maps/LD.tif",
}
FEATURE_ORDER = list(FEATURE_RASTERS.keys())
RAINFALL_IDX = FEATURE_ORDER.index("rainfall")

_feat = {n: rasterio.open(p) for n, p in FEATURE_RASTERS.items()}


def sample_features(x, y):
    row = []
    for name in FEATURE_ORDER:
        src = _feat[name]
        v = float(next(src.sample([(x, y)]))[0])
        if (src.nodata is not None and v == src.nodata) or v < -1e30:
            return None
        row.append(v)
    return row


def centroids(path):
    return [(g.centroid.x, g.centroid.y) for g in gpd.read_file(path).geometry]


train_wells = centroids(f"{BASE}/Inventory Data/Training_groundwater_wells.shp")
test_wells  = centroids(f"{BASE}/Inventory Data/Testing_groundwater_wells.shp")
all_wells   = train_wells + test_wells

area = gpd.read_file(f"{BASE}/Infrastructures/Land-based_Surveyed_area.shp")
surveyed = area.geometry.iloc[0]
minx, miny, maxx, maxy = area.total_bounds

# Margin-shrunk region: the part of the real study area the simulated world maps onto
MARGIN = 0.15
sx = (maxx - minx) * MARGIN
sy = (maxy - miny) * MARGIN
M_MINX, M_MAXX = minx + sx, maxx - sx
M_MINY, M_MAXY = miny + sy, maxy - sy

print(f"Mapped region: {M_MAXX - M_MINX:,.0f} m x {M_MAXY - M_MINY:,.0f} m")

# How many held-out wells lie inside the region the rover can actually reach?
test_in = [w for w in test_wells if M_MINX <= w[0] <= M_MAXX and M_MINY <= w[1] <= M_MAXY]
print(f"Held-out wells inside mapped region: {len(test_in)} of {len(test_wells)}")
GROUND_TRUTH = test_in if len(test_in) >= 3 else test_wells
if GROUND_TRUTH is test_wells:
    print("  -> too few inside; using all 12 held-out wells as ground truth")
print()


def far_from_wells(x, y, d=1000):
    return all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 >= d for px, py in all_wells)


# ---- Train on the 28 training wells only ----
rng_t = random.Random(42)
train_neg, att = [], 0
while len(train_neg) < len(train_wells) * 2 and att < 200000:
    x, y = rng_t.uniform(minx, maxx), rng_t.uniform(miny, maxy)
    att += 1
    if surveyed.contains(Point(x, y)) and far_from_wells(x, y) and sample_features(x, y):
        train_neg.append((x, y))

X, y_lab = [], []
for px, py in train_wells:
    f = sample_features(px, py)
    if f: X.append(f); y_lab.append(1)
for px, py in train_neg:
    X.append(sample_features(px, py)); y_lab.append(0)

model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
model.fit(np.array(X), np.array(y_lab))
print(f"Model trained on {len(y_lab)} points ({sum(y_lab)} training wells). "
      f"Held-out wells never seen.\n")


def nearest_well_dist(x, y):
    return min(((x - wx) ** 2 + (y - wy) ** 2) ** 0.5 for wx, wy in GROUND_TRUTH)


# ---- Trials ----
N_TRIALS = 200
N_CANDIDATES = 5

d_model, d_random, d_rain, d_oracle = [], [], [], []

for trial in range(N_TRIALS):
    rng = random.Random(1000 + trial)
    cands, feats = [], []
    guard = 0
    while len(cands) < N_CANDIDATES and guard < 20000:
        x = rng.uniform(M_MINX, M_MAXX)
        y = rng.uniform(M_MINY, M_MAXY)
        guard += 1
        f = sample_features(x, y)
        if f is not None:
            cands.append((x, y)); feats.append(f)
    if len(cands) < N_CANDIDATES:
        continue

    F = np.array(feats)
    scores = model.predict_proba(F)[:, 1]
    dists = [nearest_well_dist(x, y) for x, y in cands]

    d_model.append(dists[int(np.argmax(scores))])
    d_random.append(dists[rng.randrange(N_CANDIDATES)])
    d_rain.append(dists[int(np.argmax(F[:, RAINFALL_IDX]))])
    d_oracle.append(min(dists))

d_model = np.array(d_model); d_random = np.array(d_random)
d_rain = np.array(d_rain);   d_oracle = np.array(d_oracle)
print(f"Completed {len(d_model)} trials of {N_CANDIDATES} candidates each.\n")

# ---- Results ----
print("=" * 78)
print("Distance from selected target to nearest held-out well (kilometres)")
print("=" * 78)
print(f"{'Selection method':<34}{'mean':>9}{'median':>9}{'std':>9}")
print("-" * 78)
for name, d in [("Model (this work)", d_model),
                ("Random candidate", d_random),
                ("Highest rainfall (1 feature)", d_rain),
                ("Oracle (best possible)", d_oracle)]:
    print(f"{name:<34}{d.mean()/1000:>9.2f}{np.median(d)/1000:>9.2f}{d.std()/1000:>9.2f}")
print("=" * 78)

# Paired test: model vs random, on the same candidate sets
stat, pval = wilcoxon(d_model, d_random)
better = (d_model < d_random).sum()
ties = (d_model == d_random).sum()
print(f"\nPaired Wilcoxon signed-rank, model vs random: W={stat:.0f}, p={pval:.4g}")
print(f"Model closer in {better}/{len(d_model)} trials "
      f"({100*better/len(d_model):.1f}%), tied in {ties}")

stat2, pval2 = wilcoxon(d_model, d_rain)
print(f"Paired Wilcoxon, model vs rainfall-only:      W={stat2:.0f}, p={pval2:.4g}")

# ---- Hit rates ----
print("\n" + "=" * 78)
print("Hit rate: proportion of trials with selected target within radius R of a well")
print("=" * 78)
print(f"{'R':<12}{'Model':>12}{'Random':>12}{'Rainfall':>12}{'Oracle':>12}")
print("-" * 78)
for R in [2000, 5000, 10000, 15000]:
    print(f"{R/1000:>4.0f} km    "
          f"{(d_model<R).mean()*100:>11.1f}%"
          f"{(d_random<R).mean()*100:>11.1f}%"
          f"{(d_rain<R).mean()*100:>11.1f}%"
          f"{(d_oracle<R).mean()*100:>11.1f}%")
print("=" * 78)
