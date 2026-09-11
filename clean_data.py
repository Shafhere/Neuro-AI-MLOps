import pandas as pd

# ==========================================
# 1. LOAD RAW DATA
# ==========================================

df = pd.read_csv("data/neuro_patient_raw.csv")

print("Original shape:", df.shape)


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("After removing duplicates:", df.shape)


# ==========================================
# 3. HANDLE INVALID VALUES
# ==========================================

# Invalid age
df.loc[
    (df["age"] < 18) | (df["age"] > 100),
    "age"
] = None

# Invalid heart rate
df.loc[
    (df["heart_rate"] < 40) | (df["heart_rate"] > 200),
    "heart_rate"
] = None

# Invalid cognitive score
df.loc[
    (df["cognitive_score"] < 0) |
    (df["cognitive_score"] > 100),
    "cognitive_score"
] = None


# ==========================================
# 4. MISSING VALUE IMPUTATION
# ==========================================

numeric_columns = df.select_dtypes(
    include="number"
).columns

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


# ==========================================
# 5. CLEAN CATEGORICAL VALUES
# ==========================================

categorical_columns = [
    "gender",
    "family_history",
    "smoking_status",
    "alcohol_frequency"
]

for column in categorical_columns:

    df[column] = df[column].fillna(
        df[column].mode()[0]
    )


# ==========================================
# 6. VERIFY CLEAN DATA
# ==========================================

print("\n========== CLEANING RESULTS ==========")

print("Final shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ==========================================
# 7. SAVE CLEAN DATA
# ==========================================

df.to_csv(
    "data/neuro_patient_clean.csv",
    index=False
)

print("\nClean dataset saved successfully!")