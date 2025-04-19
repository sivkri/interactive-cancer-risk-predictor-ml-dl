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
     ```streamlit run app/streamlit_app.py```
2. Open the provided local URL in your browser to interact with the application.
