import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main() -> None:
    # Load dataset from project directory
    data_path = os.path.join(os.path.dirname(__file__), "crop_yield1.csv")
    data = pd.read_csv(data_path)

    # Clean column names
    data.columns = data.columns.str.strip()

    # Normalise key categorical columns (match app.py behaviour)
    for col in ["Crop", "Season", "State"]:
        if col in data.columns:
            data[col] = (
                data[col].astype(str).str.strip().str.title()
            )

    # Target
    y = data["Yield"]

    # Drop target and helper columns from features so the model
    # no longer depends on Crop_Year or Production at all.
    drop_cols = ["Yield"]
    if "Crop_Year" in data.columns:
        drop_cols.append("Crop_Year")
    if "Production" in data.columns:
        drop_cols.append("Production")

    X = data.drop(columns=drop_cols)

    # Encode categorical features
    categorical_cols = ["Crop", "Season", "State"]
    encoders: dict[str, LabelEncoder] = {}

    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Basic evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Save model artefacts
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "yield_model.pkl"))
    joblib.dump(encoders, os.path.join(model_dir, "encoders.pkl"))
    joblib.dump(y, os.path.join(model_dir, "target_values.pkl"))

    print("Trained model (without Crop_Year) saved to 'model/'")


if __name__ == "__main__":
    main()

