import os, joblib
MODEL_PATH = "models/baseline.joblib"
_model = None

def load_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    return _model

def predict(df_features):
    m = load_model()
    if m is None:
        return None
    return m.predict(df_features)
