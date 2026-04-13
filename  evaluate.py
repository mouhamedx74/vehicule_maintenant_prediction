import pandas as pd
import pickle, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)

X_test  = pd.read_csv("data/X_test.csv").values
y_test  = pd.read_csv("data/y_test.csv").values.ravel()
rf      = pickle.load(open("models/random_forest.pkl", "rb"))
y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

os.makedirs("notebooks", exist_ok=True)

# ── Matrice de confusion ──
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["OK", "Maintenance"],
            yticklabels=["OK", "Maintenance"])
plt.title("Matrice de confusion — Random Forest")
plt.tight_layout()
plt.savefig("notebooks/confusion_matrix.png", dpi=150)
plt.close()

# ── Courbe ROC ──
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="steelblue", label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Faux positifs")
plt.ylabel("Vrais positifs")
plt.title("Courbe ROC — Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig("notebooks/roc_curve.png", dpi=150)
plt.close()

# ── Importance des features ──
rf_loaded = pickle.load(open("models/random_forest.pkl", "rb"))
features = [
    "engine_temp", "rpm", "oil_pressure", "fuel_consumption",
    "vibration", "mileage", "hour", "day_of_week", "vehicle_id_enc"
]
importances = pd.Series(rf_loaded.feature_importances_, index=features)
plt.figure(figsize=(7, 4))
importances.sort_values().plot(kind="barh", color="steelblue")
plt.title("Importance des features")
plt.tight_layout()
plt.savefig("notebooks/feature_importance.png", dpi=150)
plt.close()

print(classification_report(y_test, y_pred))
print(f"AUC-ROC : {roc_auc:.4f}")
print("Graphiques sauvegardés dans notebooks/")