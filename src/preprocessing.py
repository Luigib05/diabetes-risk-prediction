import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame):
    """
    Preprocesses the diabetes dataset:
    - Imputes missing values with median
    - Splits into features and target
    - Scales features using StandardScaler
    - Returns train-test split

    Args:
        df (pd.DataFrame): Raw dataset

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test
    """
    # Impute missing values
    cols_with_nan = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_nan:
        df[col] = df[col].fillna(df[col].median())
        
    # Create a new feature: Glucose * BMI
    df["glucose_bmi"] = df["Glucose"] * df["BMI"]

    # Split features and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
