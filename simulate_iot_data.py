import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 5000  # 5000 lectures de capteurs

data = {
    "timestamp":        pd.date_range("2023-01-01", periods=n, freq="h"),
    "vehicle_id":       np.random.choice(["V001", "V002", "V003"], n),
    "engine_temp":      np.random.normal(90, 10, n),       # °C
    "rpm":              np.random.normal(3000, 500, n),    # tours/min
    "oil_pressure":     np.random.normal(4.0, 0.5, n),    # bar
    "fuel_consumption": np.random.normal(8.0, 1.5, n),   # L/100km
    "vibration":        np.random.normal(0.5, 0.15, n),  # g
    "mileage":          np.cumsum(np.random.randint(50, 150, n)),
}

# Cible binaire : maintenance requise si conditions critiques
df = pd.DataFrame(data)
df["maintenance_required"] = (
    (df["engine_temp"]  > 105) |
    (df["oil_pressure"] < 3.0)  |
    (df["vibration"]    > 0.8)
).astype(int)

os.makedirs("data", exist_ok=True)
df.to_csv("data/iot_data.csv", index=False)
print(f"Dataset créé : {df.shape[0]} lignes")
print(f"Alertes maintenance : {df['maintenance_required'].mean():.1%}")