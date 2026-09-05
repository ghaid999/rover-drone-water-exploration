import csv
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pickle

with open("feature_table.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

def parse_row_col(tile_id):
    # e.g. ..._image_2_34.npy -> row=2, col=34
    match = re.search(r"_image_(\d+)_(\d+)\.npy", tile_id)
    return int(match.group(1)), int(match.group(2))

data = []
for r in rows:
    tile_id = r[0]
    row, col = parse_row_col(tile_id)
    ndwi = float(r[1])
    ndsi = float(r[2])
    label = int(r[4])
    data.append((row, col, ndwi, ndsi, label))

all_rows = sorted(set(d[0] for d in data))
print("Row range:", min(all_rows), "to", max(all_rows), "-", len(all_rows), "distinct rows")

# Use the last 20% of rows (by row index) as the test set -> spatially separate
split_row = all_rows[int(len(all_rows) * 0.8)]
print(f"Training on rows < {split_row}, testing on rows >= {split_row}")

train_data = [d for d in data if d[0] < split_row]
test_data = [d for d in data if d[0] >= split_row]

X_train = np.array([[d[2], d[3]] for d in train_data])
y_train = np.array([d[4] for d in train_data])
X_test = np.array([[d[2], d[3]] for d in test_data])
y_test = np.array([d[4] for d in test_data])

print(f"Train set: {len(y_train)} tiles ({sum(y_train)} water)")
print(f"Test set: {len(y_test)} tiles ({sum(y_test)} water)")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
print("Confusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred, target_names=["non-water", "water"]))

with open("water_potential_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to water_potential_model.pkl")
