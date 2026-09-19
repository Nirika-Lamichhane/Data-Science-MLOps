import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    
    # Clean target variable
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # TotalCharges has empty spaces, convert to float (creates NaNs for blanks)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop customerID as it is an identifier, not a feature
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Identify numeric and categorical columns (added 'str' to fix Pandas4Warning)
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'bool', 'str']).columns.tolist()
    
    return X, y, numeric_features, categorical_features

def main():
    # 1. Load Data
    data_file = "data/telco_churn.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Could not find dataset at {data_file}. Please place it there.")
        
    X, y, numeric_features, categorical_features = load_and_preprocess_data(data_file)
    
    # Split data: 70% reference/train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Robust MLOps Preprocessor pipeline with Imputers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Define 3 different model architectures with distinct hyperparameters
    models = {
        "LogisticRegression": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    }
    
    mlflow.set_experiment("Telco_Customer_Churn_Experiment")
    
    best_f1 = 0
    best_run_id = None
    best_model_name = None

    os.makedirs("artifacts", exist_ok=True)

    for name, model_instance in models.items():
        with mlflow.start_run(run_name=name) as run:
            # Build full pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model_instance)
            ])
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Predict
            preds = pipeline.predict(X_test)
            probs = pipeline.predict_proba(X_test)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            roc_auc = roc_auc_score(y_test, probs)
            
            # Log Parameters
            mlflow.log_param("model_family", name)
            if name == "LogisticRegression":
                mlflow.log_param("C", 0.5)
            elif name == "RandomForest":
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 6)
            elif name == "GradientBoosting":
                mlflow.log_param("n_estimators", 150)
                mlflow.log_param("learning_rate", 0.05)
                mlflow.log_param("max_depth", 4)
                
            # Log Metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            
            # Generate and log artifacts
            cm = confusion_matrix(y_test, preds)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            cm_path = f"artifacts/cm_{name}.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)
            
            # Log model
            mlflow.sklearn.log_model(pipeline, "model", serialization_format="cloudpickle")

            print(f"Finished training {name} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
            
            # Track best model based on F1 Score
            if f1 > best_f1:
                best_f1 = f1
                best_run_id = run.info.run_id
                best_model_name = name

    print(f"\nBest Model: {best_model_name} with F1-Score: {best_f1:.4f}")
    
    # Register best model
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    model_details = mlflow.register_model(model_uri, "TelcoChurnModel")
    
    version = model_details.version
    print(f"Registered model 'TelcoChurnModel' version {version}")
    
    client.transition_model_version_stage(
        name="TelcoChurnModel",
        version=version,
        stage="Staging"
    )
    print(f"Transitioned version {version} to Staging.")

if __name__ == "__main__":
    main()