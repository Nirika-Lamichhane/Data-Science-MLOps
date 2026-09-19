# Week 17 Assignment - Track A: Data Science MLOps

**Project:** Telco Customer Churn Pipeline

**Tools Used:** Python, `uv`, MLflow, Evidently AI, Scikit-Learn, Pandas

This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline. It satisfies all Track A deliverable requirements, focusing on professional environment management, comprehensive experiment tracking, model registry transitions, and production data drift monitoring.

---

## 📋 Deliverables & Rubric Checklist

* **Environment & Reproducibility:** Configured using `uv` with a locked `uv.lock` file and a professional `src` package layout.
* **Experiment Tracking:** Implemented via MLflow. Tracked three distinct models with hyperparameters, metrics (F1, ROC-AUC, Accuracy, Precision, Recall), and visual artifacts (Confusion Matrices).
* **Model Registry:** The best-performing model was programmatically registered and transitioned to the "Staging" environment.
* **Drift Monitoring:** Implemented via Evidently AI. Data was split into reference (70%) and current (30%), with synthetic drift injected to validate the monitoring pipeline.
* **Documentation:** This README fulfills the requirement for architectural explanation and reflection on challenges solved.

---

## 🚀 How to Run the Pipeline

**1. Recreate the Environment**
Because this project uses `uv`, you do not need to guess dependency versions. Run this single command to exactly mirror the environment:

```bash
uv sync

```

**2. Run the Training Pipeline (MLflow)**
This script loads the data, scales/imputes it via a Scikit-Learn pipeline, trains 3 models, logs all metrics, and registers the winner:

```bash
uv run python -m src.wk17_assignment.train

```

**3. Run the Monitoring Pipeline (Evidently AI)**
This script simulates production traffic, injects data drift into the pricing and contract features, and generates an HTML report:

```bash
uv run python -m src.wk17_assignment.monitor

```

**4. View the MLflow Dashboard**
View the experiment tracking and model registry:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

```

---

## 🧠 Architectural Strategy

### Experiment Tracking (`train.py`)

To find the optimal architecture for predicting customer churn, I tested three models:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**

**Winning Model Justification:**
Because customer churn is heavily imbalanced, relying on raw accuracy is misleading. I optimized for the **F1-Score** (balancing Precision and Recall) and **ROC-AUC**.

* **Winner:** Logistic Regression outperformed the complex ensemble methods, achieving the highest F1-Score (**0.6079**) and ROC-AUC (**0.8446**).
* It was automatically registered in the MLflow Model Registry as `TelcoChurnModel` and transitioned to **Staging**.

### Data Drift Monitoring (`monitor.py`)

To simulate a real-world degradation of the model, I split the dataset (70% Reference / 30% Current) and injected aggressive synthetic drift into the Current dataset:

* **Feature Drift:** Skewed `MonthlyCharges` by adding a +50 mean offset, and forced 50% of the dataset into "Month-to-month" contracts.
* **Target Drift:** Randomly flipped 15% of the `Churn` labels to simulate a sudden, unexpected shift in customer retention.

The Evidently AI pipeline successfully caught the `MonthlyCharges` anomaly (Wasserstein distance: 1.587) and logged the visual HTML report directly into MLflow as a historical artifact.

---

## 🛠️ Challenges Faced & Solutions Implemented

Building a modern MLOps pipeline using cutting-edge library versions introduced several technical hurdles. Here is how they were resolved:

**1. Data Leakage & Pandas Copy-on-Write Errors**

* **Problem:** The Telco dataset contains hidden blank spaces for users with zero tenure, which convert to `NaN`. Attempting to patch these using Pandas `inplace=True` caused `ChainedAssignmentErrors` due to modern Pandas Copy-on-Write memory management, crashing the model training.
* **Solution:** Removed the Pandas hacks and delegated the missing value handling to a robust Scikit-Learn `ColumnTransformer` and `Pipeline` using `SimpleImputer(strategy='median')`. This ensures identical transformations during training and inference without memory warnings.

**2. MLflow Serialization Security Blocks (`skops`)**

* **Problem:** MLflow recently switched to `skops` for model serialization. When attempting to log the pipeline, `skops` threw an `UntrustedTypesFoundException` because it blocked standard NumPy data types by default for security reasons.
* **Solution:** Explicitly set the serialization format back to the standard Python format by passing `serialization_format="cloudpickle"` inside the `mlflow.sklearn.log_model()` function.

**3. Evidently AI v0.7+ Major API Overhaul**

* **Problem:** The `uv` package manager installed the absolute newest version of Evidently AI (0.7+). This version completely deleted legacy classes like `ColumnMapping`, `TargetDriftPreset`, and `ColumnSummaryMetric`, causing the script to repeatedly crash with `ModuleNotFoundError`s.
* **Solution:** Rewrote the monitoring script to comply with the new v0.7 API:
* Replaced `ColumnMapping` with `DataDefinition`.
* Wrapped the raw Pandas DataFrames into the required `Dataset.from_pandas()` objects.
* Replaced legacy presets with individual `ValueDrift` metrics.
* Captured the output of `.run()` into a snapshot object to call `.save_html()`.



**4. MLflow UI / FileStore Deprecation on Windows**

* **Problem:** When booting `mlflow ui`, the dashboard showed up completely blank. MLflow has deprecated saving runs to local `mlruns` folders and PowerShell was blocking the environmental variable overrides.
* **Solution:** Modified both Python scripts to explicitly build and write to a local SQLite database (`mlflow.set_tracking_uri("sqlite:///mlflow.db")`) before running experiments, fully modernizing the tracking backend and instantly fixing the UI dashboard.