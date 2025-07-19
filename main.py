# main.py

from src.load_data import load_diabetes_data
from src.preprocessing import preprocess_data
from src.model_training import (
    train_random_forest,
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importances
)

# Load the data
df = load_diabetes_data("data/diabetes.csv")
print(f"\nData loaded successfully. Shape: {df.shape}")

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(df)
print(f"Preprocessing completed.")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train Random Forest
rf_model = train_random_forest(X_train, y_train)

# Evaluate
print("\nRandom Forest Results:")
evaluate_model(rf_model, X_test, y_test)

# Plot confusion matrix
y_pred_rf = rf_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred_rf)

# Plot feature importances
feature_names = df.drop("Outcome", axis=1).columns
plot_feature_importances(rf_model, feature_names)
