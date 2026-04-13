import sqlite3
import pandas as pd

conn = sqlite3.connect("data/vehicle_maintenance.db")
cur  = conn.cursor()

# ── CREATE TABLE ──
cur.execute("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp            TEXT,
        vehicle_id           TEXT,
        engine_temp          REAL,
        rpm                  REAL,
        oil_pressure         REAL,
        fuel_consumption     REAL,
        vibration            REAL,
        mileage              INTEGER,
        maintenance_required INTEGER
    )
""")
conn.commit()

# ── INSERT : charge les données IoT ──
df = pd.read_csv("data/iot_data.csv")
df.to_sql("sensor_readings", conn, if_exists="replace", index=False)
print(f"{len(df)} lignes insérées.")

# ── SELECT 1 : alertes par véhicule ──
print("\n=== Alertes par véhicule ===")
q1 = pd.read_sql("""
    SELECT
        vehicle_id,
        COUNT(*) AS total_readings,
        SUM(maintenance_required) AS alerts,
        ROUND(AVG(engine_temp), 1) AS avg_temp_c,
        ROUND(AVG(oil_pressure), 2) AS avg_pressure_bar
    FROM sensor_readings
    GROUP BY vehicle_id
    ORDER BY alerts DESC
""", conn)
print(q1.to_string(index=False))

# ── SELECT 2 : dernières alertes critiques ──
print("\n=== 5 dernières alertes critiques ===")
q2 = pd.read_sql("""
    SELECT timestamp, vehicle_id,
           ROUND(engine_temp, 1)  AS temp_c,
           ROUND(oil_pressure, 2) AS pressure,
           ROUND(vibration, 3)    AS vibration_g
    FROM sensor_readings
    WHERE maintenance_required = 1
    ORDER BY timestamp DESC
    LIMIT 5
""", conn)
print(q2.to_string(index=False))

# ── SELECT 3 : statistiques globales ──
print("\n=== Statistiques globales ===")
q3 = pd.read_sql("""
    SELECT
        COUNT(*) AS total,
        SUM(maintenance_required) AS total_alertes,
        ROUND(100.0 * SUM(maintenance_required) / COUNT(*), 2) AS pct_alertes,
        ROUND(AVG(rpm), 0) AS avg_rpm,
        ROUND(MAX(engine_temp), 1) AS max_temp
    FROM sensor_readings
""", conn)
print(q3.to_string(index=False))

conn.close()
print("\nBase créée : data/vehicle_maintenance.db")
print("Ouvre-la dans VS Code avec l'extension SQLite Viewer !")