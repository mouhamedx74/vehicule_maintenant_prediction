# Vehicle Maintenance Prediction (IoT Data)

Prédiction de pannes véhicules à partir de données IoT simulées.
Capteurs : température moteur, RPM, pression huile, consommation, vibrations.

## Stack
- **Python** — pandas, numpy, scikit-learn
- **Time Series / AutoML** — LSTM (Keras/TensorFlow)
- **SQL** — SQLite pour le stockage des lectures IoT
- **Visualisation** — matplotlib, seaborn

## Structure
```
vehicle_maintenance_prediction/
├── simulate_iot_data.py   # Génération des données capteurs IoT
├── preprocess.py          # Nettoyage, feature engineering, scaling
├── train_model.py         # Random Forest + LSTM
├── evaluate.py            # Métriques, ROC, confusion matrix
├── database.py            # SQLite — CREATE, INSERT, SELECT
├── data/                  # Fichiers CSV générés
├── models/                # Modèles sauvegardés
├── notebooks/             # Graphiques PNG exportés
└── requirements.txt       # Dépendances Python
```

## Résultats
- Classification binaire : maintenance requise ou non
- Métriques : Accuracy, AUC-ROC, F1-score, Precision, Recall
- 3 véhicules simulés sur 5000 lectures horaires

## Lancer le projet
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python simulate_iot_data.py
python preprocess.py
python train_model.py
python evaluate.py
python database.py
```