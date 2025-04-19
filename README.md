# Interactive Cancer Risk Predictor (ML & DL)

An interactive web application for breast cancer risk prediction using Machine Learning and Deep Learning models. Built with Streamlit and tracked using MLflow.

## Features

- Upload CSV files for batch predictions
- Visualize model predictions with SHAP values
- Adjust confidence thresholds for classification
- Compare performance between ML and DL models

## Project Structure

```bash

├── streamlit_app.py        		# Streamlit web application
├── models/
│   ├── deep_learning_model.h5     # Deep Learning model
├── random_forest_model.pkl     	# Random Forest model
├── deep_learning_model.h5      	# Deep Learning model
├── stacking_model.pkl          	# Stacking model
├── bagging_model.pkl          		# Bagging model
├── svm_model.pkl               	# SVM model
├── mlruns/                        # MLflow tracking directory
├── Dockerfile                     # Docker configuration for containerization
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file
└── README.md                      # Project documentation
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sivkri/interactive-cancer-risk-predictor-ml-dl.git
   cd interactive-cancer-risk-predictor-ml-dl
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
     ```bash
   streamlit run app/streamlit_app.py
     ```
2. Open the provided local URL in your browser to interact with the application.

## Docker Setup

To run the application in a Docker container:

1. Build the Docker image:
   ```bash
      docker build -t DockerFile .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 DockerFile
   ```
3. Access the application at ```http://localhost:8501```

## 🧪 MLflow Guide 

MLflow is used to track experiments, log parameters and metrics, and save trained models. 

### 🔹 1. Start the MLflow UI 

```bash 
   mlflow ui ``` 

This will launch the MLflow Tracking UI at ``(http://localhost:5000) ````

### 🔹 2. Track a Model Run in Python 

```bash 
   import mlflow
   import mlflow.sklearn
   with mlflow.start_run():
      mlflow.log_param("model_type", "RandomForest")
      mlflow.log_param("n_estimators", 100)
      mlflow.log_metric("accuracy", 0.95)
      mlflow.sklearn.log_model(rf_model, "model")
```

This will: - Log hyperparameters (`model_type`, `n_estimators`) - Log a performance metric (`accuracy`) - Save the trained model as an artifact

### 🔹 3. View Your Runs in the MLflow UI Visit 
      To explore: - Runs and experiments - Parameters and metrics - Downloadable models 

### 🔹 4. Track Multiple Experiments To create a named experiment: 
```bash 
   mlflow.set_experiment("CancerRiskPrediction")
``` 

To log to this experiment: 

```bash
   with mlflow.start_run(run_name="RandomForest_Trial_1"): ...
```

### 🔹 5. Log Custom Artifacts (like plots, confusion matrices) 

```bash
   import matplotlib.pyplot as plt
      plt.plot([0, 1], [0.5, 0.9])
      plt.savefig("plot.png")
      mlflow.log_artifact("plot.png") ``` 

### MLflow Files
   Experiment logs are stored under the `mlruns/` directory - Each run contains logged metrics, parameters, models, and artifacts
