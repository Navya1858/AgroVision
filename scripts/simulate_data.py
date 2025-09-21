import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
np.random.seed(42)

OUT = "data/raw/simulated.csv"

def gen_series(start, periods, freq='H'):
    ts = pd.date_range(start=start, periods=periods, freq=freq)
    temp = 22 + 8*np.sin(np.linspace(0, 8*np.pi, periods)) + np.random.normal(0, 1.5, periods)
    humidity = 55 + 15*np.sin(np.linspace(0, 6*np.pi, periods)+1) + np.random.normal(0, 3, periods)
    rainfall = np.random.choice([0,0,0,0,2,5,10], size=periods, p=[0.55,0.15,0.1,0.05,0.07,0.05,0.03])
    wind = np.clip(np.random.normal(8, 2, periods), 0, None)

    soil = []
    sm = 50.0
    for i in range(periods):
        et0 = max(0.0, 0.08*temp[i] + 0.02*wind[i] - 1.2)
        sm = sm + rainfall[i]*0.6 - et0 + np.random.normal(0, 0.6)
        sm = float(np.clip(sm, 10, 85))
        soil.append(sm)

    df = pd.DataFrame({
        "timestamp": ts,
        "field_id": "field_1",
        "soil_moisture_pct": np.round(soil, 2),
        "air_temp_c": np.round(temp, 2),
        "humidity_pct": np.round(humidity, 2),
        "rainfall_mm": np.round(rainfall, 2),
        "wind_speed_ms": np.round(wind, 2),
        "soil_type": "loam",
        "crop_stage": "vegetative"
    })
    return df

def main():
    start = datetime.now() - timedelta(days=60)
    periods = 60*24
    df1 = gen_series(start, periods)
    df2 = gen_series(start, periods)
    df2["field_id"] = "field_2"
    df2["soil_type"] = "sandy"
    df2["crop_stage"] = "flowering"

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.sort_values(["field_id", "timestamp"]).reset_index(drop=True)
    df["need_liters_next_24h"] = (45 - df["soil_moisture_pct"]).clip(lower=0) * 2.0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows")

if __name__ == "__main__":
    main()
