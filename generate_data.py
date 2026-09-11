import numpy as np
import pandas as pd

np.random.seed(42)

# Number of synthetic research records
n = 10000

data = {
    "patient_id": [f"NAI_{i:05d}" for i in range(1, n + 1)],

    "age": np.random.normal(58, 14, n).round(),

    "gender": np.random.choice(
        ["Male", "Female", "Other"],
        n,
        p=[0.48, 0.48, 0.04]
    ),

    "symptom_duration_months": np.random.exponential(24, n).round(1),

    "tremor_score": np.random.normal(4.5, 2, n).round(2),

    "cognitive_score": np.random.normal(72, 15, n).round(2),

    "sleep_quality_score": np.random.normal(6, 2, n).round(2),

    "heart_rate": np.random.normal(76, 12, n).round(1),

    "blood_pressure_systolic": np.random.normal(130, 20, n).round(),

    "blood_pressure_diastolic": np.random.normal(82, 12, n).round(),

    "reaction_time_ms": np.random.normal(420, 100, n).round(),

    "balance_score": np.random.normal(70, 18, n).round(2),

    "memory_score": np.random.normal(68, 17, n).round(2),

    "speech_score": np.random.normal(75, 15, n).round(2),

    "motor_score": np.random.normal(65, 20, n).round(2),

    "depression_score": np.random.normal(8, 5, n).round(2),

    "physical_activity_hours": np.random.normal(4, 2, n).round(2),

    "family_history": np.random.choice(
        ["Yes", "No"],
        n,
        p=[0.25, 0.75]
    ),

    "smoking_status": np.random.choice(
        ["Never", "Former", "Current"],
        n,
        p=[0.55, 0.30, 0.15]
    ),

    "alcohol_frequency": np.random.choice(
        ["Never", "Occasional", "Regular"],
        n,
        p=[0.35, 0.45, 0.20]
    ),

    "mri_abnormality_score": np.random.normal(3.5, 2, n).round(2),

    "eeg_abnormality_score": np.random.normal(3, 2, n).round(2),

    "inflammation_marker": np.random.normal(5, 2.5, n).round(2),

    "biomarker_level": np.random.normal(50, 15, n).round(2),
}

df = pd.DataFrame(data)


# --------------------------------------------------
# Create a realistic disease-stage target
# --------------------------------------------------

risk_score = (
    0.03 * df["age"]
    + 0.15 * df["tremor_score"]
    - 0.03 * df["cognitive_score"]
    + 0.02 * df["reaction_time_ms"]
    - 0.03 * df["balance_score"]
    - 0.02 * df["memory_score"]
    - 0.02 * df["motor_score"]
    + 0.5 * df["mri_abnormality_score"]
    + 0.4 * df["eeg_abnormality_score"]
    + 0.15 * df["symptom_duration_months"]
    + np.random.normal(0, 5, n)
)

df["disease_stage"] = pd.qcut(
    risk_score,
    q=4,
    labels=["Early", "Moderate", "Advanced", "Severe"]
)


# --------------------------------------------------
# Introduce real-world data problems
# --------------------------------------------------

# Missing values
missing_columns = [
    "cognitive_score",
    "sleep_quality_score",
    "mri_abnormality_score",
    "biomarker_level",
    "reaction_time_ms"
]

for column in missing_columns:
    missing_indices = np.random.choice(
        df.index,
        size=int(0.05 * n),
        replace=False
    )
    df.loc[missing_indices, column] = np.nan


# Duplicate records
duplicates = df.sample(100, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)


# Invalid ages
invalid_age_indices = np.random.choice(
    df.index,
    size=20,
    replace=False
)

df.loc[invalid_age_indices, "age"] = np.random.choice(
    [-10, 150, 200],
    size=20
)


# Extreme heart-rate values
outlier_indices = np.random.choice(
    df.index,
    size=20,
    replace=False
)

df.loc[outlier_indices, "heart_rate"] = np.random.choice(
    [250, 300, 400],
    size=20
)


# Extreme cognitive scores
outlier_indices = np.random.choice(
    df.index,
    size=20,
    replace=False
)

df.loc[outlier_indices, "cognitive_score"] = np.random.choice(
    [-20, 150, 200],
    size=20
)


# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# Save raw dataset
df.to_csv("data/neuro_patient_raw.csv", index=False)

print("Dataset created successfully!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())