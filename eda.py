import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/neuro_patient_raw.csv")

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# ==========================================
# 2. MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print(missing)

print("\nMissing Percentage:")
print((missing / len(df) * 100).round(2))

# ==========================================
# 3. DUPLICATES
# ==========================================

print("\n========== DUPLICATES ==========")

print("Duplicate rows:", df.duplicated().sum())

# ==========================================
# 4. STATISTICS
# ==========================================

print("\n========== NUMERICAL STATISTICS ==========")

print(df.describe())

# ==========================================
# 5. CATEGORICAL DISTRIBUTIONS
# ==========================================

print("\n========== CATEGORICAL DISTRIBUTIONS ==========")

categorical_columns = [
    "gender",
    "family_history",
    "smoking_status",
    "alcohol_frequency",
    "disease_stage"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))

# ==========================================
# 6. INVALID VALUES
# ==========================================

print("\n========== INVALID VALUES ==========")

print("Invalid ages:")
print(df[(df["age"] < 0) | (df["age"] > 100)][["patient_id", "age"]])

print("\nExtreme heart rates:")
print(df[df["heart_rate"] > 200][["patient_id", "heart_rate"]])

print("\nExtreme cognitive scores:")
print(
    df[
        (df["cognitive_score"] < 0) |
        (df["cognitive_score"] > 100)
    ][["patient_id", "cognitive_score"]]
)

# ==========================================
# 7. VISUAL EDA
# ==========================================

# Target distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="disease_stage")
plt.title("Disease Stage Distribution")
plt.show()

# Important numerical distributions
plot_columns = [
    "age",
    "tremor_score",
    "cognitive_score",
    "heart_rate",
    "reaction_time_ms",
    "motor_score"
]

for column in plot_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.show()

# ==========================================
# 8. OUTLIER VISUALIZATION
# ==========================================

for column in [
    "age",
    "heart_rate",
    "cognitive_score",
    "reaction_time_ms"
]:

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        data=df,
        x=column
    )

    plt.title(f"Outlier Analysis: {column}")
    plt.show()

# ==========================================
# 9. CORRELATION
# ==========================================

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(14, 10))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=False
)

plt.title("Feature Correlation Matrix")
plt.show()

# ==========================================
# 10. FEATURE VS TARGET
# ==========================================

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="disease_stage",
    y="tremor_score"
)

plt.title("Tremor Score by Disease Stage")
plt.show()

print("\n========== EDA COMPLETE ==========")