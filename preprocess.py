import pandas as pd
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("data/iot_data.csv", parse_dates=["timestamp"])

# Features temporelles
df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"]  = df["timestamp"].dt.dayofweek

# Encodage vehicle_id → numérique
le = LabelEncoder()
df["vehicle_id_enc"] = le.fit_transform(df["vehicle_id"])

features = [
    "engine_temp", "rpm", "oil_pressure", "fuel_consumption",
    "vibration", "mileage", "hour", "day_of_week", "vehicle_id_enc"
]
target = "maintenance_required"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Sauvegarde
os.makedirs("models", exist_ok=True)
pd.DataFrame(X_train_sc, columns=features).to_csv("data/X_train.csv", index=False)
pd.DataFrame(X_test_sc,  columns=features).to_csv("data/X_test.csv",  index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv",   index=False)
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("Préprocessing terminé.")
print(f"Train : {X_train_sc.shape} | Test : {X_test_sc.shape}")