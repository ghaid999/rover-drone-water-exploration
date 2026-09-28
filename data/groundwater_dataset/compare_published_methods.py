import geopandas as gpd
import rasterio
import numpy as np
import random
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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

PUBLISHED_MAPS = {
    "WOE-IOE": f"{BASE}/Groundewater Potential maps/GwP WOE-IOE.tif",
    "EBF-IOE": f"{BASE}/Groundewater Potential maps/GwP EBF-IOE.tif",
}

_feat = {n: rasterio.open(p) for n, p in FEATURE_RASTERS.items()}
_maps = {n: rasterio.open(p) for n, p in PUBLISHED_MAPS.items()}


def _bad(src, v):
    return (src.nodata is not None and v == src.nodata) or v < -1e30


def sample_features(x, y):
    row = []
    for name in FEATURE_ORDER:
        src = _feat[name]
        v = float(next(src.sample([(x, y)]))[0])
        if _bad(src, v):
            return None
        row.append(v)
    return row


def sample_published(x, y, which):
    src = _maps[which]
    v = float(next(src.sample([(x, y)]))[0])
    return None if _bad(src, v) else v


def valid_everywhere(x, y):
    """Point must have real data in all 9 features AND both published maps,
    so every method is scored on exactly the same points."""
    if sample_features(x, y) is None:
        return False
    return all(sample_published(x, y, m) is not None for m in PUBLISHED_MAPS)


def centroids(path):
    return [(g.centroid.x, g.centroid.y) for g in gpd.read_file(path).geometry]


# ---------- Wells (the paper's own split) ----------
train_wells = centroids(f"{BASE}/Inventory Data/Training_groundwater_wells.shp")
test_wells  = centroids(f"{BASE}/Inventory Data/Testing_groundwater_wells.shp")
all_wells   = train_wells + test_wells

# ---------- Two sampling regions ----------
area = gpd.read_file(f"{BASE}/Infrastructures/Land-based_Surveyed_area.shp")
surveyed = area.geometry.iloc[0]
s_minx, s_miny, s_maxx, s_maxy = area.total_bounds

# Region B = intersection of every raster's extent (so all methods are defined there)
all_srcs = list(_feat.values()) + list(_maps.values())
r_minx = max(s.bounds.left   for s in all_srcs)
r_miny = max(s.bounds.bottom for s in all_srcs)
r_maxx = min(s.bounds.right  for s in all_srcs)
r_maxy = min(s.bounds.top    for s in all_srcs)

print("Region A (surveyed area):  "
      f"{s_maxx - s_minx:,.0f} m x {s_maxy - s_miny:,.0f} m")
print("Region B (full raster extent): "
      f"{r_maxx - r_minx:,.0f} m x {r_maxy - r_miny:,.0f} m")
print(f"Region B is ~{((r_maxx-r_minx)*(r_maxy-r_miny))/((s_maxx-s_minx)*(s_maxy-s_miny)):.1f}x "
      "the area of Region A\n")


def far_from_wells(x, y, min_dist=1000):
    return all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 >= min_dist for px, py in all_wells)


def sample_negatives(n, seed, region):
    """region='A' -> inside surveyed polygon; region='B' -> anywhere in raster extent."""
    rng = random.Random(seed)
    if region == "A":
        bx = (s_minx, s_miny, s_maxx, s_maxy)
    else:
        bx = (r_minx, r_miny, r_maxx, r_maxy)
    pts, attempts = [], 0
    while len(pts) < n and attempts < 400000:
        x = rng.uniform(bx[0], bx[2]); y = rng.uniform(bx[1], bx[3])
        attempts += 1
        if region == "A" and not surveyed.contains(Point(x, y)):
            continue
        if not far_from_wells(x, y):
            continue
        if not valid_everywhere(x, y):
            continue
        pts.append((x, y))
    return pts, attempts


def build_and_eval(region, label):
    print("=" * 72)
    print(f"TEST {region}: {label}")
    print("=" * 72)

    # Train the model with negatives from THIS region
    train_neg, att_t = sample_negatives(len(train_wells) * 2, seed=42, region=region)
    X, y = [], []
    for px, py in train_wells:
        f = sample_features(px, py)
        if f:
            X.append(f); y.append(1)
    for px, py in train_neg:
        X.append(sample_features(px, py)); y.append(0)

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(np.array(X), np.array(y))
    print(f"  Model trained on {len(y)} points ({sum(y)} wells, {len(y)-sum(y)} negatives from region {region})")

    # Evaluation set: held-out wells + fresh negatives from the SAME region
    eval_neg, att_e = sample_negatives(len(test_wells) * 2, seed=7, region=region)

    pts, labels = [], []
    dropped = 0
    for px, py in test_wells:
        if valid_everywhere(px, py):
            pts.append((px, py)); labels.append(1)
        else:
            dropped += 1
    for px, py in eval_neg:
        pts.append((px, py)); labels.append(0)
    labels = np.array(labels)
    if dropped:
        print(f"  WARNING: {dropped} held-out well(s) dropped for no-data")
    print(f"  Evaluation set: {len(labels)} points ({labels.sum()} wells, {(labels==0).sum()} negatives)")

    scores = {
        "Random Forest (this work)": np.array([model.predict_proba([sample_features(px, py)])[0][1] for px, py in pts]),
        "WOE-IOE (published)":       np.array([sample_published(px, py, "WOE-IOE") for px, py in pts]),
        "EBF-IOE (published)":       np.array([sample_published(px, py, "EBF-IOE") for px, py in pts]),
    }

    def boot(lbl, sc, n_boot=2000, seed=0):
        rng = np.random.default_rng(seed)
        ip = np.where(lbl == 1)[0]; ineg = np.where(lbl == 0)[0]
        out = []
        for _ in range(n_boot):
            s = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ineg, len(ineg), True)])
            if len(np.unique(lbl[s])) > 1:
                out.append(roc_auc_score(lbl[s], sc[s]))
        return np.percentile(out, 2.5), np.percentile(out, 97.5)

    print()
    print(f"  {'Method':<28}{'AUC':>8}   {'95% CI':>18}   {'mean@wells':>11} {'mean@neg':>9}")
    print("  " + "-" * 82)
    results = {}
    for name, sc in scores.items():
        auc = roc_auc_score(labels, sc)
        lo, hi = boot(labels, sc)
        results[name] = (auc, lo, hi)
        print(f"  {name:<28}{auc:>8.3f}   [{lo:.3f}, {hi:.3f}]"
              f"   {sc[labels==1].mean():>11.3f} {sc[labels==0].mean():>9.3f}")
    print()
    return results


res_a = build_and_eval("A", "negatives from INSIDE the surveyed area (local discrimination)")
res_b = build_and_eval("B", "negatives from the FULL raster extent (regional discrimination)")

print("=" * 72)
print("SUMMARY: AUC-ROC")
print("=" * 72)
print(f"{'Method':<28}{'Test A (local)':>18}{'Test B (regional)':>20}")
print("-" * 72)
for name in res_a:
    a = res_a[name]; b = res_b[name]
    print(f"{name:<28}{a[0]:>10.3f}        {b[0]:>10.3f}")
print("=" * 72)
print("Test A = can a method tell a well from nearby non-well terrain?")
print("Test B = can a method tell a well from the wider landscape?")
print("AUC 0.5 = chance;  1.0 = perfect ranking.")
