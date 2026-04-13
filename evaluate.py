import pandas as pd
import pickle, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

X_test  = pd.read_csv("data/X_test.csv").values
y_test  = pd.read_csv("data/y_test.csv").values.ravel()
rf      = pickle.load(open("models/random_forest.pkl", "rb"))
y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

os.makedirs("notebooks", exist_ok=True)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["OK", "Maintenance"], yticklabels=["OK", "Maintenance"])
plt.title("Matrice de confusion")
plt.tight_layout()
plt.savefig("notebooks/confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="steelblue", label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.title("Courbe ROC")
plt.legend()
plt.tight_layout()
plt.savefig("notebooks/roc_curve.png", dpi=150)
plt.close()

print(classification_report(y_test, y_pred))
print(f"AUC-ROC : {roc_auc:.4f}")
print("Graphiques sauvegardés dans notebooks/")
