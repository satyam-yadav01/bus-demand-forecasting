
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.models.evaluate import calculate_metrics


# ---------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. FEATURE CONFIGURATION
# ---------------------------------------------------------

NUMERIC_FEATURES = [
    "departure_hour",
    "day_of_week",
    "month",
    "is_weekend",
    "planned_capacity",
    "hist_route_time_avg",
    "route_time_history_count",
    "hist_route_avg",
    "route_history_count",
    "hist_global_time_avg",
]

CATEGORICAL_FEATURES = [
    "route_id",
    "season",
    "peak_period",
]

TARGET = "passenger_demand"

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------
# 3. LOAD DATA
# ---------------------------------------------------------

def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    validation_df = pd.read_csv(DATA_DIR / "validation.csv")

    return train_df, validation_df


# ---------------------------------------------------------
# 4. CREATE PREPROCESSING PIPELINE
# ---------------------------------------------------------

def create_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


# ---------------------------------------------------------
# 5. DEFINE MODELS
# ---------------------------------------------------------

def create_models():

    ridge_model = Ridge(alpha=10)

    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    return {
        "Ridge": ridge_model,
        "XGBoost": xgb_model,
    }


# ---------------------------------------------------------
# 6. SAVE VALIDATION PREDICTIONS
# ---------------------------------------------------------

def save_validation_predictions(
    validation_df,
    predictions,
    model_name,
):

    prediction_df = validation_df[
        [
            "trip_id",
            "route_id",
            "departure_datetime",
            "peak_period",
            TARGET,
        ]
    ].copy()

    prediction_df["predicted_demand"] = predictions

    file_name = (
        model_name.lower().replace(" ", "_")
        + "_validation_predictions.csv"
    )

    prediction_path = REPORT_DIR / file_name

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    print(f"Saved validation predictions: {prediction_path}")


# ---------------------------------------------------------
# 7. EXTRACT XGBOOST FEATURE IMPORTANCE
# ---------------------------------------------------------

def save_feature_importance(xgb_pipeline):

    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    xgb_model = xgb_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": xgb_model.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    )

    importance_path = REPORT_DIR / "xgboost_feature_importance.csv"

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    print("\nTop 15 XGBoost features:")
    print(
        importance_df.head(15).to_string(index=False)
    )

    print(f"\nSaved feature importance: {importance_path}")


# ---------------------------------------------------------
# 8. TRAIN AND EVALUATE MODELS
# ---------------------------------------------------------

def train_models():

    print("\nLoading training and validation data...")

    train_df, validation_df = load_data()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_validation = validation_df[FEATURES]
    y_validation = validation_df[TARGET]

    print(f"Training records: {len(train_df)}")
    print(f"Validation records: {len(validation_df)}")

    models = create_models()

    comparison_results = []
    fitted_models = {}

    for model_name, model in models.items():

        print("\n" + "=" * 60)
        print(f"Training {model_name}")
        print("=" * 60)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("model", model),
            ]
        )

        # Fit preprocessing and model using training data only.
        pipeline.fit(X_train, y_train)

        # Generate validation predictions.
        predictions = pipeline.predict(X_validation)

        # Demand cannot be negative.
        predictions = predictions.clip(min=0)

        # Calculate validation metrics.
        metrics = calculate_metrics(
            y_validation,
            predictions,
        )

        print(f"\n{model_name} validation results:")
        print(f"MAE:  {metrics['MAE']:.3f}")
        print(f"RMSE: {metrics['RMSE']:.3f}")
        print(f"WAPE: {metrics['WAPE']:.3f}%")

        comparison_results.append(
            {
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "WAPE": metrics["WAPE"],
            }
        )

        # Save validation predictions for later analysis.
        save_validation_predictions(
            validation_df,
            predictions,
            model_name,
        )

        # Save trained model pipeline.
        model_path = MODEL_DIR / (
            model_name.lower() + "_pipeline.joblib"
        )

        joblib.dump(pipeline, model_path)

        print(f"Saved model: {model_path}")

        fitted_models[model_name] = pipeline

    # -----------------------------------------------------
    # 9. FEATURE IMPORTANCE
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("XGBOOST FEATURE IMPORTANCE")
    print("=" * 60)

    save_feature_importance(fitted_models["XGBoost"])

    # -----------------------------------------------------
    # 10. MODEL COMPARISON
    # -----------------------------------------------------

    comparison_df = pd.DataFrame(comparison_results)

    comparison_df = comparison_df.sort_values(
        by="WAPE",
        ascending=True,
    )

    comparison_path = REPORT_DIR / "model_comparison_validation.csv"

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("FINAL VALIDATION MODEL COMPARISON")
    print("=" * 60)

    print(comparison_df.to_string(index=False))

    print(f"\nSaved comparison: {comparison_path}")

    print("\nTraining completed successfully.")


# ---------------------------------------------------------
# 11. MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    train_models()
