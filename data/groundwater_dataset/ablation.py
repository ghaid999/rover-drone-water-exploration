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
MARGIN = 0.15
sx, sy = (maxx - minx) * MARGIN, (maxy - miny) * MARGIN
M_MINX, M_MAXX = minx + sx, maxx - sx
M_MINY, M_MAXY = miny + sy, maxy - sy
START = ((M_MINX + M_MAXX) / 2, (M_MINY + M_MAXY) / 2)   # rover start = region centre

GROUND_TRUTH = [w for w in test_wells
                if M_MINX <= w[0] <= M_MAXX and M_MINY <= w[1] <= M_MAXY]
print(f"Ground truth: {len(GROUND_TRUTH)} held-out wells inside the mapped region")
print(f"Rover start (region centre): ({START[0]:,.0f}, {START[1]:,.0f})\n")


def far_from_wells(x, y, d=1000):
    return all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 >= d for px, py in all_wells)


rng_t = random.Random(42)
train_neg, att = [], 0
while len(train_neg) < len(train_wells) * 2 and att < 200000:
    x, y = rng_t.uniform(minx, maxx), rng_t.uniform(miny, maxy)
    att += 1
    if surveyed.contains(Point(x, y)) and far_from_wells(x, y) and sample_features(x, y):
        train_neg.append((x, y))
X, yl = [], []
for px, py in train_wells:
    f = sample_features(px, py)
    if f: X.append(f); yl.append(1)
for px, py in train_neg:
    X.append(sample_features(px, py)); yl.append(0)
model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
model.fit(np.array(X), np.array(yl))
print(f"Model trained on {len(yl)} points ({sum(yl)} training wells)\n")


def nearest(x, y):
    return min(((x - wx) ** 2 + (y - wy) ** 2) ** 0.5 for wx, wy in GROUND_TRUTH)


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def draw_candidates(rng, n):
    cands, feats, guard = [], [], 0
    while len(cands) < n and guard < 50000:
        x = rng.uniform(M_MINX, M_MAXX); y = rng.uniform(M_MINY, M_MAXY)
        guard += 1
        f = sample_features(x, y)
        if f is not None:
            cands.append((x, y)); feats.append(f)
    return cands, feats


N_TRIALS = 200
N_DRONE = 5        # candidates the drone surveys (matches current implementation)
K_ROVER = 2        # sites the rover can physically reach and sense in one mission

res = {k: {"d": [], "t": []} for k in
       ["A_full", "A_full20", "B_rover2", "C_random", "D_single"]}

for t in range(N_TRIALS):
    rng = random.Random(2000 + t)

    # --- A: drone surveys N candidates, model ranks, rover drives to best ---
    c, f = draw_candidates(rng, N_DRONE)
    if len(c) < N_DRONE: continue
    F = np.array(f); s = model.predict_proba(F)[:, 1]
    best = int(np.argmax(s))
    res["A_full"]["d"].append(nearest(*c[best]))
    res["A_full"]["t"].append(dist(START, c[best]))

    # --- C: same survey, but target chosen at random (removes the model) ---
    pick = rng.randrange(N_DRONE)
    res["C_random"]["d"].append(nearest(*c[pick]))
    res["C_random"]["t"].append(dist(START, c[pick]))

    # --- A20: drone surveys 20 candidates (the recommended configuration) ---
    c20, f20 = draw_candidates(rng, 20)
    if len(c20) == 20:
        s20 = model.predict_proba(np.array(f20))[:, 1]
        b20 = int(np.argmax(s20))
        res["A_full20"]["d"].append(nearest(*c20[b20]))
        res["A_full20"]["t"].append(dist(START, c20[b20]))

    # --- B: no drone. Rover drives to K sites, senses each, returns to best ---
    cb, fb = draw_candidates(rng, K_ROVER)
    if len(cb) == K_ROVER:
        sb = model.predict_proba(np.array(fb))[:, 1]
        bb = int(np.argmax(sb))
        res["B_rover2"]["d"].append(nearest(*cb[bb]))
        tour = dist(START, cb[0]) + sum(dist(cb[i], cb[i+1]) for i in range(K_ROVER-1))
        if bb != K_ROVER - 1:                     # backtrack to the best site
            tour += dist(cb[K_ROVER-1], cb[bb])
        res["B_rover2"]["t"].append(tour)

    # --- D: no drone, no model. Rover drives to one random location ---
    cd, _ = draw_candidates(rng, 1)
    if cd:
        res["D_single"]["d"].append(nearest(*cd[0]))
        res["D_single"]["t"].append(dist(START, cd[0]))

for k in res:
    res[k]["d"] = np.array(res[k]["d"]); res[k]["t"] = np.array(res[k]["t"])

LABELS = {
    "A_full":    f"A  Full system (drone {N_DRONE} + model)",
    "A_full20":  "A* Full system (drone 20 + model)",
    "B_rover2":  f"B  Rover only, no drone (k={K_ROVER})",
    "C_random":  f"C  Drone {N_DRONE} + random target (no model)",
    "D_single":  "D  Random location (no drone, no model)",
}

print("=" * 86)
print("SYSTEM ABLATION — 200 trials")
print("=" * 86)
print(f"{'Condition':<40}{'dist to well (km)':>20}{'rover travel (km)':>20}")
print(f"{'':<40}{'mean':>10}{'median':>10}{'mean':>12}{'median':>8}")
print("-" * 86)
for k in ["A_full", "A_full20", "B_rover2", "C_random", "D_single"]:
    d, tr = res[k]["d"], res[k]["t"]
    print(f"{LABELS[k]:<40}{d.mean()/1000:>10.2f}{np.median(d)/1000:>10.2f}"
          f"{tr.mean()/1000:>12.2f}{np.median(tr)/1000:>8.2f}")
print("=" * 86)

print("\nPaired comparisons against condition A (full system, drone 5):")
base = res["A_full"]["d"]
for k in ["B_rover2", "C_random", "D_single"]:
    d = res[k]["d"]
    n = min(len(base), len(d))
    _, pv = wilcoxon(base[:n], d[:n])
    print(f"  A vs {LABELS[k]:<40} p = {pv:.4g}")

n = min(len(res["C_random"]["d"]), len(res["D_single"]["d"]))
_, pv_cd = wilcoxon(res["C_random"]["d"][:n], res["D_single"]["d"][:n])
print(f"\n  C vs D (expected to be equivalent):          p = {pv_cd:.4g}")
print("  (C and D are statistically the same by construction: choosing one of N")
print("   uniformly-random candidates at random is itself a uniform random point.)")
