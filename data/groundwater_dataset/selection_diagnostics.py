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

# Direction that should indicate MORE water potential, from the model's
# learned behaviour and hydrogeology: low elevation favours groundwater.
HIGHER_IS_BETTER = {
    "sentinel1_vv": True, "sentinel1_vh": True, "dem": False, "slope": False,
    "rainfall": True, "twi": True, "curvature": True,
    "stream_density": True, "lineament_density": True,
}


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

GROUND_TRUTH = [w for w in test_wells
                if M_MINX <= w[0] <= M_MAXX and M_MINY <= w[1] <= M_MAXY]
print(f"Ground truth: {len(GROUND_TRUTH)} held-out wells inside the mapped region\n")


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


def run_trials(n_candidates, n_trials=200, seed_base=1000):
    """Returns dict of method -> array of distances (metres)."""
    out = {"model": [], "random": [], "oracle": []}
    for f in FEATURE_ORDER:
        out[f] = []
    for t in range(n_trials):
        rng = random.Random(seed_base + t)
        cands, feats, guard = [], [], 0
        while len(cands) < n_candidates and guard < 50000:
            x = rng.uniform(M_MINX, M_MAXX); y = rng.uniform(M_MINY, M_MAXY)
            guard += 1
            fv = sample_features(x, y)
            if fv is not None:
                cands.append((x, y)); feats.append(fv)
        if len(cands) < n_candidates:
            continue
        F = np.array(feats)
        dists = np.array([nearest(x, y) for x, y in cands])
        out["model"].append(dists[int(np.argmax(model.predict_proba(F)[:, 1]))])
        out["random"].append(dists[rng.randrange(n_candidates)])
        out["oracle"].append(dists.min())
        for i, fname in enumerate(FEATURE_ORDER):
            col = F[:, i]
            pick = int(np.argmax(col)) if HIGHER_IS_BETTER[fname] else int(np.argmin(col))
            out[fname].append(dists[pick])
    return {k: np.array(v) for k, v in out.items()}


# ============ DIAGNOSTIC 1: every single feature vs the model ============
print("=" * 80)
print("DIAGNOSTIC 1 — single-feature heuristics vs the model (5 candidates, 200 trials)")
print("=" * 80)
r5 = run_trials(5)
dm = r5["model"]
print(f"{'Selection method':<30}{'mean km':>10}{'median km':>11}{'vs model p':>13}")
print("-" * 80)
print(f"{'MODEL (this work)':<30}{dm.mean()/1000:>10.2f}{np.median(dm)/1000:>11.2f}{'-':>13}")
rows = []
for fname in FEATURE_ORDER:
    d = r5[fname]
    try:
        _, pv = wilcoxon(dm, d)
    except ValueError:
        pv = float("nan")
    rows.append((d.mean(), fname, d, pv))
for mean_d, fname, d, pv in sorted(rows):
    flag = "  <-- matches model" if pv > 0.05 else ""
    print(f"{fname:<30}{d.mean()/1000:>10.2f}{np.median(d)/1000:>11.2f}{pv:>13.4g}{flag}")
print(f"{'random':<30}{r5['random'].mean()/1000:>10.2f}{np.median(r5['random'])/1000:>11.2f}"
      f"{wilcoxon(dm, r5['random'])[1]:>13.4g}")
print(f"{'ORACLE (best possible)':<30}{r5['oracle'].mean()/1000:>10.2f}"
      f"{np.median(r5['oracle'])/1000:>11.2f}{'-':>13}")
print("=" * 80)
n_match = sum(1 for _, _, _, pv in rows if pv > 0.05)
print(f"Single features statistically indistinguishable from the model: {n_match} of {len(FEATURE_ORDER)}")

# ============ DIAGNOSTIC 2: does a harder ranking problem help? ============
print("\n" + "=" * 80)
print("DIAGNOSTIC 2 — effect of candidate-set size")
print("=" * 80)
print(f"{'Candidates':<12}{'Model km':>11}{'Rainfall km':>13}{'Random km':>12}"
      f"{'Oracle km':>12}{'model vs rain p':>17}")
print("-" * 80)
for n in [5, 10, 20, 40]:
    r = run_trials(n)
    _, pv = wilcoxon(r["model"], r["rainfall"])
    print(f"{n:<12}{r['model'].mean()/1000:>11.2f}{r['rainfall'].mean()/1000:>13.2f}"
          f"{r['random'].mean()/1000:>12.2f}{r['oracle'].mean()/1000:>12.2f}{pv:>17.4g}")
print("=" * 80)
print("Reading: if the model/rainfall gap widens and p falls as candidates increase,")
print("the learned model's advantage appears only when the ranking problem is harder.")
