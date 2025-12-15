from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "Sleep_health_and_lifestyle_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "data_processed.csv"
print("Loading raw dataset...")
df = pd.read_csv(RAW_DATA_PATH)
df = df.drop(columns=["Person ID", "Occupation"], errors="ignore")

df = df.fillna(0)

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

bmi_map = {
    "Normal": 0,
    "Normal Weight": 0,
    "Overweight": 1,
    "Obese": 2
}

df["BMI_Category"] = df["BMI Category"].map(bmi_map)
df = df.drop(columns=["BMI Category"])

bp = df["Blood Pressure"].str.split("/", expand=True)
df["BP_Systolic"] = bp[0].astype(int)
df["BP_Diastolic"] = bp[1].astype(int)
df = df.drop(columns=["Blood Pressure"])

df["Hypertension"] = (
    (df["BP_Systolic"] >= 140) | (df["BP_Diastolic"] >= 90)
).astype(int)

df["Sleep Disorder"] = df["Sleep Disorder"].replace(0, "None")
df["SleepRisk"] = (df["Sleep Disorder"] != "None").astype(int)
df = df.drop(columns=["Sleep Disorder"])

df = df.drop_duplicates()

df.to_csv(OUTPUT_PATH, index=False)
print(f"Processed dataset saved to {OUTPUT_PATH}")
