import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pickle

with open("feature_table.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

X = np.array([[float(r[1]), float(r[2])] for r in rows])  # mean_ndwi, mean_ndsi
y = np.array([int(r[4]) for r in rows])                    # label

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
print("Confusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred, target_names=["non-water", "water"]))

with open("water_potential_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to water_potential_model.pkl")
