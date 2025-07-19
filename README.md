# Diabetes Risk Prediction Project

This project aims to predict the likelihood of a patient developing type 2 diabetes based on clinical variables using machine learning models.

It was developed as part of a personal practice initiative to strengthen my skills in data science and artificial intelligence, focusing on predictive modeling, exploratory data analysis, and structured data workflows.

---

## Dataset

This dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases. The goal is to diagnostically predict whether a patient has diabetes based on several medical measurements. The dataset includes only female patients who are at least 21 years old and of Pima Indian heritage.

- **Source**: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Format**: CSV
- **Features**:
  - `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
- **Target variable**:
  - `Outcome` (1 = diabetic, 0 = non-diabetic)

---

## Tech Stack

- **Language**: Python 3.x
- **Libraries**:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
- **Environment**:
  - Local development (VSCode + virtual environment)
  - Optional GPU via Google Colab

---

## Project Structure

diabetes-risk-prediction_project/
│
├── data/ # Raw dataset (CSV)
├── notebooks/ # Exploratory Data Analysis (EDA)
├── src/ # Python modules: data loading, preprocessing, model training
├── outputs/ # Visual outputs: plots, metrics
├── .venv/ # Virtual environment (excluded from version control)
├── .gitignore
├── README.md
├── requirements.txt
└── main.py # Pipeline to run data → model → evaluation


---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Luigib05/diabetes-risk-prediction.git
cd diabetes-risk-prediction

# Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\activate          # On Windows

# Install dependencies:

pip install -r requirements.txt

# Run the full pipeline:

python main.py

## Final Model: Random Forest Classifier

The final model was trained using a Random Forest classifier with the default configuration. Performance was evaluated on a holdout test set.

Evaluation Metrics:

Accuracy: 0.76

Precision: 0.68

Recall: 0.59

F1-score: 0.63

# Confusion Matrix:
[[85 15]
 [22 32]]

# Top Important Features:

Glucose

BMI

Age

Glucose × BMI (engineered feature)

# Next Steps (Optional Ideas)

Hyperparameter tuning with GridSearchCV

Cross-validation

Model persistence with joblib

Streamlit web app for interactive prediction

License

This project is for educational and non-commercial purposes.