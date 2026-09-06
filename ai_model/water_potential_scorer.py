import pickle
import numpy as np
from feature_generator import get_features_for_location, FEATURE_ORDER

MODEL_PATH = "groundwater_potential_model_final.pkl"
WATER_LOCATIONS_PATH = "../data/hidden_water_locations.json"


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_water_potential_score(x, y, model, rng=None):
    """
    Full pipeline: coordinates -> synthetic features -> trained model -> score.
    Returns (score, closeness, features) where score is the model's predicted
    probability that this location has groundwater potential (0.0 to 1.0).
    """
    features, closeness = get_features_for_location(x, y, WATER_LOCATIONS_PATH, rng)
    feature_vector = np.array([[features[name] for name in FEATURE_ORDER]])
    probability = model.predict_proba(feature_vector)[0][1]  # index 1 = "water" class
    return float(probability), closeness, features


if __name__ == "__main__":
    model = load_model()
    rng = np.random.default_rng(42)

    test_points = [
        ("On water_1", 12.0, 8.0),
        ("On water_2", -18.0, -5.0),
        ("Far from everything", -24.0, -24.0),
        ("Middle of the map", 0.0, 0.0),
    ]

    print("Location            Score   Closeness")
    print("-" * 42)
    for label, x, y in test_points:
        score, closeness, _ = get_water_potential_score(x, y, model, rng)
        print(f"{label:20s} {score:.3f}   {closeness:.3f}")


def get_water_potential_score_averaged(x, y, model, n_samples=10, rng=None):
    """
    Same as get_water_potential_score, but averages n_samples draws to reduce
    the noise inherent in a single synthetic feature draw. Recommended for
    actual drone surveys, since a single reading can be misleading given how
    much the trained model's water/no-water classes overlap.
    """
    if rng is None:
        rng = np.random.default_rng()
    scores = []
    for _ in range(n_samples):
        score, _, _ = get_water_potential_score(x, y, model, rng)
        scores.append(score)
    return float(np.mean(scores)), float(np.std(scores))
