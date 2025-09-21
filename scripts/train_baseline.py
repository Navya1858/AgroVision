import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

DATA = "data/raw/simulated.csv"
MODEL = "models/baseline.joblib"

def load_data(path=DATA):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    X = df[["soil_moisture_pct","air_temp_c","humidity_pct","rainfall_mm","wind_speed_ms","soil_type","crop_stage"]]
    y = df["need_liters_next_24h"]
    return X, y

def build_pipeline():
    num_feats = ["soil_moisture_pct","air_temp_c","humidity_pct","rainfall_mm","wind_speed_ms"]
    cat_feats = ["soil_type","crop_stage"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
    ])
    return Pipeline([("pre", pre), ("lr", LinearRegression())])

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = build_pipeline()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Test MAE: {mean_absolute_error(y_test, preds):.2f} | R2: {r2_score(y_test, preds):.3f}")
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    joblib.dump(model, MODEL)
    print(f"Saved model → {MODEL}")

if __name__ == "__main__":
    main()
