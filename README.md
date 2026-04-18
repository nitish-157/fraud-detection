# Credit Card Fraud Detection - Machine Learning Project

A comprehensive machine learning solution for detecting fraudulent credit card transactions using Python and scikit-learn.

## 📁 Project Structure

```
fraud-detection/
├── data/                  # Raw and processed data
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── src/                   # Source code
│   ├── __init__.py
│   ├── preprocess.py      # Data preprocessing and feature engineering
│   ├── train.py           # Model training modules
│   └── evaluate.py        # Model evaluation and metrics
├── models/                # Trained model artifacts
├── app/                   # Streamlit web application
│   └── streamlit_app.py
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd fraud-detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Preparation

- Place your credit card transaction data in the `data/` directory
- Use the preprocessing module to clean and prepare data

```python
from src.preprocess import prepare_data

X_train, X_test, y_train, y_test = prepare_data("data/transactions.csv")
```

### 3. Train Model

```python
from src.train import train_model

model = train_model(X_train, y_train, model_type="random_forest")
model.save("models/fraud_detector.pkl")
```

### 4. Evaluate Performance

```python
from src.evaluate import evaluate_model

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

evaluator = evaluate_model(y_test, y_pred, y_pred_proba)
evaluator.print_summary()
evaluator.plot_roc_curve()
```

### 5. Run Web Application

```bash
streamlit run app/streamlit_app.py
```

## 📚 Module Documentation

### `preprocess.py`

Data preprocessing and feature engineering utilities:

- `load_data()` - Load CSV data
- `handle_missing_values()` - Handle missing data
- `remove_outliers()` - Remove statistical outliers
- `normalize_features()` - Scale features
- `prepare_data()` - Complete preprocessing pipeline

### `train.py`

Model training interface:

- `FraudDetectionModel` - Wrapper class for ML models
- `train_model()` - Train a new model
- Supported models: Random Forest, Gradient Boosting, Logistic Regression

### `evaluate.py`

Model evaluation and visualization:

- `ModelEvaluator` - Comprehensive evaluation class
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Visualizations: Confusion matrix, ROC curve, Precision-Recall curve

## 🔧 Available Models

1. **Random Forest** (default)
   - Excellent for non-linear relationships
   - Good interpretability
   - Fast prediction

2. **Gradient Boosting**
   - High accuracy
   - Better handling of class imbalance
   - Requires more tuning

3. **Logistic Regression**
   - Fast training
   - Interpretable coefficients
   - Baseline model

## 📊 Key Features

- ✅ Modular and reusable code structure
- ✅ Comprehensive documentation and docstrings
- ✅ Multiple evaluation metrics and visualizations
- ✅ Type hints for better code clarity
- ✅ Production-ready model saving/loading
- ✅ Interactive Streamlit web interface
- ✅ Stratified train-test split for imbalanced data

## 💉 Requirements

See `requirements.txt` for complete dependencies. Key packages:

- scikit-learn - Machine learning algorithms
- pandas - Data manipulation
- numpy - Numerical computing
- matplotlib - Plotting
- seaborn - Statistical visualization
- streamlit - Web app framework
- joblib - Model serialization

## 📈 Model Performance

Expected metrics on typical datasets:

- **Accuracy**: 98-99.5%
- **Precision**: 97-99%
- **Recall**: 92-96%
- **ROC-AUC**: 0.97-0.99

_Note: Actual performance depends on data quality and class balance_

## Model Explainability (SHAP)

To understand model decisions, SHAP (SHapley Additive exPlanations) was used.

- Identified most important features contributing to fraud detection
- Visualized feature impact using SHAP summary plots
- Improved interpretability of the ML model

This helps in understanding why a transaction is classified as fraud.

## 🔐 Best Practices

1. **Data Privacy**: Never commit sensitive data to version control
2. **Model Versioning**: Track model versions and hyperparameters
3. **Cross-Validation**: Use cross-validation for robust evaluation
4. **Class Imbalance**: Consider SMOTE or class weights for imbalanced data
5. **Feature Engineering**: Explore domain-specific features

## 🐛 Troubleshooting

### Memory Issues with Large Datasets

Use data sampling or process in batches

### Class Imbalance

- Use `class_weight="balanced"` in model initialization
- Consider SMOTE or undersampling techniques

### Poor Model Performance

- Analyze feature distributions
- Check for data leakage
- Tune hyperparameters using GridSearchCV
- Collect more training data

## 📝 Notebooks

Exploratory analysis notebooks go in the `notebooks/` directory:

- `01_eda.ipynb` - Exploratory Data Analysis
- `02_feature_engineering.ipynb` - Feature creation and selection
- `03_model_comparison.ipynb` - Compare different models

## 📞 Support

For issues or questions, please open a GitHub issue or contact the development team.

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: Kaggle Credit Card Fraud Detection
- Built with Python and scikit-learn

---

**Last Updated**: April 2026
