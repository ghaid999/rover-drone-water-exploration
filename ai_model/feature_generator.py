import json
import numpy as np

FEATURE_STATS = {
    "sentinel1_vv":      {"water_mean": -10.591, "water_std": 4.337, "nowater_mean": -10.041, "nowater_std": 5.145, "min": -22.0, "max": 3.5},
    "sentinel1_vh":      {"water_mean": -17.930, "water_std": 5.298, "nowater_mean": -18.522, "nowater_std": 6.998, "min": -46.0, "max": -5.0},
    "dem":               {"water_mean": 340.075, "water_std": 98.159, "nowater_mean": 435.500, "nowater_std": 114.546, "min": 80.0, "max": 700.0},
    "slope":             {"water_mean": 12.360, "water_std": 12.112, "nowater_mean": 11.274, "nowater_std": 8.920, "min": 0.0, "max": 65.0},
    "rainfall":          {"water_mean": 9.112, "water_std": 1.257, "nowater_mean": 8.542, "nowater_std": 0.464, "min": 7.0, "max": 13.5},
    "twi":               {"water_mean": 7.493, "water_std": 3.765, "nowater_mean": 6.453, "nowater_std": 2.263, "min": 0.5, "max": 30.0},
    "curvature":         {"water_mean": -0.608, "water_std": 1.528, "nowater_mean": -0.208, "nowater_std": 1.188, "min": -6.0, "max": 3.0},
    "stream_density":    {"water_mean": 1.894, "water_std": 0.695, "nowater_mean": 1.893, "nowater_std": 0.567, "min": 0.7, "max": 3.6},
    "lineament_density": {"water_mean": 0.538, "water_std": 0.173, "nowater_mean": 0.583, "nowater_std": 0.161, "min": 0.05, "max": 1.05},
}

FEATURE_ORDER = list(FEATURE_STATS.keys())
DECAY_SCALE_M = 6.0
MAX_RESAMPLE_ATTEMPTS = 50


def _load_water_locations(path):
    with open(path) as f:
        data = json.load(f)
    return data["water_locations"]


def _sample_in_range(rng, mean, std, lo, hi):
    """Draw from N(mean, std), rejecting and redrawing if outside [lo, hi],
    instead of clipping (which would create an artificial spike at the edge)."""
    for _ in range(MAX_RESAMPLE_ATTEMPTS):
        value = rng.normal(mean, std)
        if lo <= value <= hi:
            return value
    # Fallback (extremely unlikely to be reached): clip as a last resort
    return float(np.clip(rng.normal(mean, std), lo, hi))


def get_features_for_location(x, y, water_locations_path, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    water_locations = _load_water_locations(water_locations_path)

    closeness = 0.0
    for site in water_locations:
        dist = np.sqrt((x - site["x"]) ** 2 + (y - site["y"]) ** 2)
        site_closeness = np.exp(-dist / DECAY_SCALE_M) * site["water_score"]
        closeness = max(closeness, site_closeness)
    closeness = np.clip(closeness, 0.0, 1.0)

    features = {}
    for name in FEATURE_ORDER:
        stats = FEATURE_STATS[name]
        blended_mean = closeness * stats["water_mean"] + (1 - closeness) * stats["nowater_mean"]
        blended_std = closeness * stats["water_std"] + (1 - closeness) * stats["nowater_std"]
        features[name] = float(_sample_in_range(rng, blended_mean, blended_std, stats["min"], stats["max"]))

    return features, closeness


if __name__ == "__main__":
    water_json = "../data/hidden_water_locations.json"
    rng = np.random.default_rng(42)

    print("--- Point ON water_1 (12.0, 8.0) ---")
    feats, closeness = get_features_for_location(12.0, 8.0, water_json, rng)
    print("Closeness:", round(closeness, 3))
    for k, v in feats.items():
        print(f"  {k}: {v:.3f}")

    print()
    print("--- Point far from everything (-24, -24) ---")
    feats, closeness = get_features_for_location(-24.0, -24.0, water_json, rng)
    print("Closeness:", round(closeness, 3))
    for k, v in feats.items():
        print(f"  {k}: {v:.3f}")
