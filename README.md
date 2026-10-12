# Diabetes Prediction with Support Vector Machine

A machine learning model that predicts whether a patient has diabetes from 8 diagnostic measurements, using a **linear Support Vector Machine** on the PIMA Indians Diabetes dataset.

## Dataset

**PIMA Indians Diabetes Database** (National Institute of Diabetes and Digestive and Kidney Diseases; available on [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)). It has 768 female patients aged 21+ of Pima Indian heritage, of whom 268 have diabetes and 500 do not.

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (2-h oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin-fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Genetic likelihood score based on family history |
| Age | Age (years) |
| **Outcome** | 1 = diabetic, 0 = non-diabetic |

`diabetes.csv` is included in this repository.

## Method

1. Load the data and explore it: shape, summary statistics, class balance, and feature means by outcome.
2. **Standardize** all features with `StandardScaler`.
3. **Split** the data 80 / 20, stratified by outcome (`random_state=2`).
4. Train an **SVM with a linear kernel**.
5. Evaluate accuracy on the training and test sets.
6. Build a **predictive system** that takes one patient's 8 values, scales them, and predicts diabetic or non-diabetic.

## Results

| Data | Accuracy |
|---|---|
| Training set | 78.66 % |
| **Test set** | **77.27 %** |

Training and test accuracy are close, so the model is not overfitting. This is a simple, interpretable baseline.

## Usage

**Google Colab:** upload `Diabetes_Prediction.ipynb` and `diabetes.csv`, then run all cells.

**Locally:**
```bash
git clone https://github.com/arifex/Diabetes-Prediction-.git
cd Diabetes-Prediction-
pip install numpy pandas scikit-learn
python Diabetes_Prediction.py
```
In local runs, change the CSV path in the script from `/content/diabetes.csv` to `diabetes.csv`.

**Predict for a new patient:** edit `input_data` in the last cell, in the order
`(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)`, for example:
```python
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
```

## Repository structure

```
├── Diabetes_Prediction.ipynb   # notebook with outputs
├── Diabetes_Prediction.py      # same pipeline as a script
└── diabetes.csv                # PIMA Indians Diabetes dataset
```

## Possible improvements

- Treat physiologically impossible zero values (Glucose, BloodPressure, BMI, Insulin) as missing and impute them
- Compare with Logistic Regression, Random Forest and gradient boosting
- Use cross-validation and report precision, recall and ROC-AUC, not accuracy alone

## Tech stack

Python · NumPy · pandas · scikit-learn

## Author

**Md. Arifuzzaman** — Assistant Professor, Department of CSE, Leading University, Sylhet, Bangladesh
[GitHub](https://github.com/arifex) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=ngntKZ8AAAAJ)
