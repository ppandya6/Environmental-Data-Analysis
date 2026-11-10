"""Generate illustrative observations, never represented as measured evidence."""
import numpy as np
import pandas as pd


def generate_sample(rows=500, seed=42):
    """Create synthetic daily weather and pollution observations with quality issues."""
    if rows < 50:
        raise ValueError("Sample data requires at least 50 rows")
    rng = np.random.default_rng(seed)
    temperature = rng.normal(18, 8, rows)
    humidity = np.clip(rng.normal(60, 15, rows), 10, 100)
    wind = rng.gamma(2, 2, rows)
    traffic = rng.uniform(20, 100, rows)
    pm25 = np.maximum(0, 8 + .35 * traffic - 1.8 * wind + .12 * humidity
                      + .03 * (temperature - 18)**2 + rng.normal(0, 3, rows))
    frame = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=rows),
                          "temperature_c": temperature, "humidity_pct": humidity,
                          "wind_speed_ms": wind, "traffic_index": traffic,
                          "pm25_ug_m3": pm25})
    for column in ["temperature_c", "humidity_pct", "wind_speed_ms"]:
        frame.loc[rng.choice(rows, max(1, rows // 30), replace=False), column] = np.nan
    return pd.concat([frame, frame.iloc[:3]], ignore_index=True)
