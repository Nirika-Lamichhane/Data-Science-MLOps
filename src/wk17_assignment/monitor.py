import os
import pandas as pd
import numpy as np
import mlflow

from sklearn.model_selection import train_test_split

# --- STRICT EVIDENTLY V0.7+ API IMPORTS ---
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
from evidently.metrics import ValueDrift

def main():
    # 1. Load Data
    data_file = "data/telco_churn.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Could not find dataset at {data_file}")
        
    df = pd.read_csv(data_file)
    
    # Basic cleanup 
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # 2. Split into Reference (70%) and Current (30%)
    reference_df, current_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Churn'])
    
    # Reset indices to prevent Pandas warnings
    reference_df = reference_df.copy().reset_index(drop=True)
    current_df = current_df.copy().reset_index(drop=True)
    
    # 3. Inject Synthetic Drift into the 'Current' dataframe
    print("Injecting synthetic drift into 'current' dataset...")
    
    # Feature Drift
    current_df['MonthlyCharges'] = current_df['MonthlyCharges'] + 50 + np.random.normal(0, 10, size=len(current_df))
    skew_indices = current_df.sample(frac=0.5, random_state=42).index
    current_df.loc[skew_indices, 'Contract'] = 'Month-to-month'
    
    # Target Drift
    flip_indices = current_df.sample(frac=0.15, random_state=99).index
    current_df.loc[flip_indices, 'Churn'] = 1 - current_df.loc[flip_indices, 'Churn']
    
    # 4. EVIDENTLY v0.7+ DATA DEFINITION
    # This officially replaces the old ColumnMapping
    categorical_features = current_df.select_dtypes(include=['object', 'bool']).columns.tolist()
    numeric_features = current_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features.remove('Churn')
    
    # Add Churn back into the numerical features so it gets monitored for drift!
    numeric_features.append('Churn')
    
    definition = DataDefinition(
        numerical_columns=numeric_features,
        categorical_columns=categorical_features
    )
    
    # 5. EVIDENTLY v0.7+ DATASET OBJECTS
    # We must wrap the raw Pandas dataframes in their new Dataset class
    reference_dataset = Dataset.from_pandas(reference_df, data_definition=definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=definition)
    
    # 6. Build the Evidently Report
    print("Generating Evidently AI Drift Report (This might take a minute)...")
    drift_report = Report(metrics=[
        DataDriftPreset(),
        ValueDrift(column="Churn"),         # Monitor the target
        ValueDrift(column="MonthlyCharges") # Monitor the drifted feature
    ])
    
    # The .run() method now RETURNS the evaluation! Save it to a variable:
    my_eval = drift_report.run(reference_dataset, current_dataset)
    
    # 7. Save Report locally
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/evidently_drift_report.html"
    
    # Call save_html on the NEW evaluation object, not the original report
    my_eval.save_html(report_path)

    print(f"Evidently report saved locally to {report_path}")

    # 8. Log the report to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telco_Customer_Churn_Experiment")
    
    # 8. Log the report to MLflow
    mlflow.set_experiment("Telco_Customer_Churn_Experiment") 
    with mlflow.start_run(run_name="Evidently_Drift_Monitoring"):
        mlflow.log_artifact(report_path)
        print("Report successfully logged to MLflow as an artifact.")

if __name__ == "__main__":
    main()