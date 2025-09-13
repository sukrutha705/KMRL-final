import pandas as pd
import numpy as np
from faker import Faker
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
from supabase import create_client, Client
import json

# --- Supabase Credentials ---
# These must match the credentials in your index.html file
SUPABASE_URL = "https://nquzhbqkdwwxqmkdwjuz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5xdXpoYnFrZHd3eHFta2R3anV6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzQzMDE5NywiZXhwIjoyMDczMDA2MTk3fQ.-KUdHmt3eNGvGhmjPhW9cytu-VArqtD1t1RAElrm4yI"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Constants for Data Generation & AI Pipeline ---
NUM_TRAINS = 25
NUM_JOB_CARDS = 200
TRAIN_PREFIX = "T"
STABLING_PREFIX = "IBL Bay"
BRANDING_TYPES = ["KMRL Standard", "Jio", "HDFC Bank", "Maruti Suzuki", "Dabur", "Samsung", "Myntra", "Amazon"]
CLEANING_STATUS = ["Completed", "Pending"]
MAXIMO_STATUS = ["Open", "Closed"]
MODEL_FILENAME = 'trained_train_model.joblib'
DATA_FILENAME = 'unified_open_source_dataset.csv'

# Define feature columns for consistency between training and prediction
FEATURE_COLUMNS = [
    'mileage_km', 'is_critical_open_job', 'num_open_jobs',
    'stabling_location_num', 'fitness_certificate'
]

JOB_CARD_NAMES = [
    # Inspection
    "Brake System Inspection",
    "Door Mechanism Check",
    "Air Conditioning Test",
    "Pantograph Inspection",
    "Wheel Profile Measurement",
    "Traction System Health Check",
    "Suspension Inspection",
    
    # Cleaning
    "Interior Deep Cleaning",
    "Exterior Wash",
    "Seat & Upholstery Sanitization",
    "Driver Cabin Sanitization",
    
    # Repairs & Replacements
    "Traction Motor Replacement",
    "Battery Pack Change",
    "HVAC Filter Replacement",
    "Coupler Repair",
    "Brake Pad Replacement",
    "Lighting System Repair",
    
    # Safety & Certification
    "Fire Extinguisher Check",
    "Emergency Lighting Test",
    "Annual Fitness Certification",
    "First Aid Kit Replenishment",
    
    # Electronics & Software
    "Onboard CCTV Firmware Update",
    "Passenger Information System Reset",
    "Driver Console Calibration",
    "Speed Sensor Recalibration"
]


def generate_open_source_data():
    """Generates a complete, open-source dataset for the project."""
    print("Generating open-source dataset...")
    fake = Faker()

    train_ids = [f'{TRAIN_PREFIX}{i+101}' for i in range(NUM_TRAINS)]

    job_cards = []
    for _ in range(NUM_JOB_CARDS):
        train_id = np.random.choice(train_ids)
        job_id = f'J{fake.unique.random_int(min=1000, max=99999999)}'
        status = np.random.choice(MAXIMO_STATUS, p=[0.3, 0.7])
        is_critical = np.random.choice([True, False], p=[0.1, 0.9])
        job_cards.append({
            'train_id': train_id,
            'job_id': job_id,
            "name": random.choice(JOB_CARD_NAMES),
            # 👈 Added for frontend dropdown
            'status': status,
            'isCritical': is_critical      # 👈 Renamed to camelCase for frontend
        })
    job_card_df = pd.DataFrame(job_cards)
    
    # Aggregate Maximo data and group job cards by train
    maximo_agg = job_card_df.groupby('train_id').apply(lambda x: pd.Series({
        'is_critical_open_job': (x['isCritical'] & (x['status'] == 'Open')).any(),
        'num_open_jobs': (x['status'] == 'Open').sum(),
        'num_closed_jobs': (x['status'] == 'Closed').sum(),
        'jobCards': x[['job_id', 'name', 'status', 'isCritical']].to_dict('records')
    }), include_groups=False).reset_index()

    iot_data = []
    for train_id in train_ids:
        mileage = np.random.randint(200, 10000)
        iot_data.append({'train_id': train_id, 'mileage_km': mileage})
    iot_df = pd.DataFrame(iot_data)

    operational_data = []
    for train_id in train_ids:
        # UPDATED PART: p list now has 8 values to match the 8 BRANDING_TYPES
        branding = np.random.choice(BRANDING_TYPES, p=[0.3, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.1])
        
        cleaning = np.random.choice(CLEANING_STATUS, p=[0.8, 0.2])
        stabling_loc = f'{STABLING_PREFIX} {np.random.randint(1, 26)}'
        fitness = np.random.choice([True, False], p=[0.9, 0.1])
        operational_data.append({
            'train_id': train_id,
            'branding': branding,
            'cleaningStatus': cleaning,
            'stablingLocation': stabling_loc,
            'fitnessCertificate': fitness,
            'inService': False
        })
    operational_df = pd.DataFrame(operational_data)

    unified_df = pd.merge(maximo_agg, iot_df, on='train_id', how='outer')
    unified_df = pd.merge(unified_df, operational_df, on='train_id', how='outer')
    unified_df = unified_df.fillna(0)
    
    unified_df['stabling_location_num'] = unified_df['stablingLocation'].str.extract(r'(\d+)').astype(float)
    
    unified_df['is_eligible'] = ~unified_df['is_critical_open_job'] & (unified_df['num_open_jobs'] < 3) & unified_df['fitnessCertificate']
    unified_df['human_override'] = unified_df.apply(
        lambda row: not row['is_eligible'] and (row['mileage_km'] < 1000 or row['branding'] == "Advertiser B"),
        axis=1
    )
    unified_df['final_decision'] = unified_df.apply(
        lambda row: row['is_eligible'] or row['human_override'],
        axis=1
    )
    
    unified_df.to_csv(DATA_FILENAME, index=False)
    print(f"Dataset saved to '{DATA_FILENAME}'.")
    return unified_df

def train_ai_model(df):
    """Trains a machine learning model on the generated data."""
    print("Training the Random Forest model...")
    
    global FEATURE_COLUMNS
    features_df = pd.get_dummies(df, columns=['branding', 'cleaningStatus'], dtype=int)
    
    new_features = [col for col in features_df.columns if col.startswith(('branding_', 'cleaningStatus_'))]
    for new_feat in new_features:
        if new_feat not in FEATURE_COLUMNS:
            FEATURE_COLUMNS.append(new_feat)
    
    features = features_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    target = features_df['final_decision']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    joblib.dump(model, MODEL_FILENAME)
    print(f"Trained model saved to '{MODEL_FILENAME}'.")
    
    return model

def generate_predictions(model, df):
    """Uses the trained model to generate predictions and upserts them directly to Supabase."""
    print("Generating AI predictions and writing to Supabase...")
    
    features_df = pd.get_dummies(df, columns=['branding', 'cleaningStatus'], dtype=int)
    features = features_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    
    predictions = model.predict(features)
    df['ai_prediction'] = pd.Series(predictions).map({True: 'Recommended', False: 'Not Recommended'})

    data_to_upsert = []
    for index, row in df.iterrows():
        data_to_upsert.append({
            'train_id': row['train_id'],
            'mileage': int(row['mileage_km']),
            'branding': row['branding'],
            'cleaningStatus': row['cleaningStatus'],
            'stablingLocation': row['stablingLocation'],
            'fitnessCertificate': bool(row['fitnessCertificate']),
            'jobCards': row['jobCards'],
            'inService': bool(row['inService']),
            'ai_prediction': row['ai_prediction']
        })

    response = supabase.table("trains").upsert(data_to_upsert).execute()

    if response.data:
        print(f"Successfully upserted {len(response.data)} records to Supabase.")
    else:
        print("Upsert failed or returned no data.")
        print(getattr(response, 'error', 'No error attribute available'))


if __name__ == "__main__":
    df = generate_open_source_data()
    model = train_ai_model(df)
    generate_predictions(model, df)
    print("AI pipeline executed successfully. Supabase database has been updated.")
