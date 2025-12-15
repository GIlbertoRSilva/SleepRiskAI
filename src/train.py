# src/train.py

import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "data_processed.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "sleep_risk_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
FI_PATH = MODEL_DIR / "feature_importance.csv"


print("Loading processed dataset...")
df = pd.read_csv(DATA_PATH)

TARGET = "SleepRisk"  

X = df.drop(columns=[TARGET])
y = df[TARGET]

feature_names = list(X.columns)
with open(FEATURE_NAMES_PATH, "w") as f:
    json.dump(feature_names, f)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training RandomForest...")
model = model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

fi_df.to_csv(FI_PATH, index=False)
print(f"Feature importance saved to {FI_PATH}")
