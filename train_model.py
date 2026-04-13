import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

X_train = pd.read_csv("data/X_train.csv").values
X_test  = pd.read_csv("data/X_test.csv").values
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test  = pd.read_csv("data/y_test.csv").values.ravel()

# ── Modèle 1 : Random Forest (baseline rapide) ──
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("=== Random Forest ===")
print(classification_report(y_test, y_pred))
pickle.dump(rf, open("models/random_forest.pkl", "wb"))

# ── Modèle 2 : LSTM Time Series ──
# Reshape pour LSTM : (samples, timesteps=1, features)
X_tr3 = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_te3 = X_test.reshape((X_test.shape[0],  1, X_test.shape[1]))

model = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[1])),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1,  activation="sigmoid")
])
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

model.fit(
    X_tr3, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

loss, acc = model.evaluate(X_te3, y_test, verbose=0)
print(f"\n=== LSTM ===\nAccuracy : {acc:.4f} | Loss : {loss:.4f}")

model.save("models/lstm_model.keras")
print("Modèles sauvegardés dans models/")