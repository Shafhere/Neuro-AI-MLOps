import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD CLEAN DATA
# ==========================================

df = pd.read_csv("data/neuro_patient_clean.csv")

# Remove patient ID
X = df.drop(columns=["patient_id", "disease_stage"])
y = df["disease_stage"]


# ==========================================
# 2. IDENTIFY COLUMN TYPES
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns


# ==========================================
# 3. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ==========================================
# 4. MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)


# ==========================================
# 5. COMPLETE ML PIPELINE
# ==========================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 7. TRAIN
# ==========================================

pipeline.fit(X_train, y_train)


# ==========================================
# 8. PREDICT
# ==========================================

predictions = pipeline.predict(X_test)


# ==========================================
# 9. EVALUATE
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


# ==========================================
# 10. SAVE MODEL
# ==========================================

joblib.dump(
    pipeline,
    "model_v1.pkl"
)

print("\nModel saved as model_v1.pkl")