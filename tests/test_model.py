import joblib
import pandas as pd


def test_model_exists():
    model = joblib.load("model_v1.pkl")
    assert model is not None


def test_model_can_predict():
    model = joblib.load("model_v1.pkl")

    data = pd.read_csv("data/neuro_patient_clean.csv")

    X = data.drop(columns=["disease_stage"])

    predictions = model.predict(X.head(5))

    assert len(predictions) == 5