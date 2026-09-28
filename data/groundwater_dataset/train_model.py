import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pickle

with open("groundwater_feature_table.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

feature_names = header[2:-1]  # skip x, y; skip label
X = np.array([[float(v) for v in r[2:-1]] for r in rows])
y = np.array([int(r[-1]) for r in rows])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
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
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred, target_names=["no water", "water"]))

print("\nFeature importances:")
for name, imp in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")

with open("groundwater_potential_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved to groundwater_potential_model.pkl")
