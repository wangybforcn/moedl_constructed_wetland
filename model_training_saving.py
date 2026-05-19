#用于读取csv数据集并训练和保存模型

# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_dataset(csv_path="extracted_parameters.csv"):
    """Load the CSV dataset and return only numeric parameter columns."""
    df = pd.read_csv(csv_path)
    if "pdf_name" in df.columns:
        df = df.drop(columns=["pdf_name"])
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df


def train_imputer(df):
    """Train an IterativeImputer on the numeric dataset."""
    feature_names = df.columns.tolist()
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    imputer = IterativeImputer(estimator=estimator, random_state=42, max_iter=10, initial_strategy="mean")
    imputer.fit(df)
    return imputer, feature_names


def save_model(imputer, feature_names, file_path="trained_imputer.joblib"):
    joblib.dump({"imputer": imputer, "feature_names": feature_names}, file_path)
    print(f"Model saved to {file_path}")


def load_model(file_path="trained_imputer.joblib"):
    data = joblib.load(file_path)
    return data["imputer"], data["feature_names"]


def predict_remaining_parameters(known_params, imputer, feature_names):
    """Predict all parameters when some values are provided."""
    row = pd.Series({name: np.nan for name in feature_names}, dtype=float)
    for key, value in known_params.items():
        if key not in feature_names:
            raise ValueError(f"Unknown parameter name: {key}")
        row[key] = value

    if row.isna().all():
        raise ValueError("At least one known parameter value must be provided.")

    filled = imputer.transform(row.to_frame().T)
    prediction = dict(zip(feature_names, filled[0]))
    remaining = {k: v for k, v in prediction.items() if pd.isna(known_params.get(k, np.nan))}
    return prediction, remaining


def evaluate_imputer(imputer, df):
    """Compute a quick evaluation by masking every column once.
    Only evaluates on rows where the original value is not NaN.
    """
    errors = {}
    for col in df.columns:
        # 只在原始值非NaN的行上评估
        valid_mask = df[col].notna()
        if valid_mask.sum() == 0:
            errors[col] = float('nan')
            continue
        df_masked = df.copy()
        df_masked[col] = np.nan
        filled = imputer.transform(df_masked)
        col_idx = df.columns.get_loc(col)
        y_true = df.loc[valid_mask, col].values
        y_pred = filled[valid_mask.values, col_idx]
        errors[col] = mean_squared_error(y_true, y_pred)
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description="Train and save an imputer model for parameter prediction.")
    parser.add_argument(
        "--data",
        default="extracted_parameters.csv",
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--output",
        default="trained_imputer.joblib",
        help="Path to save the trained model file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = load_dataset(args.data)
    imputer, feature_names = train_imputer(df)
    save_model(imputer, feature_names, args.output)

    # 示例：已知一部分参数，预测其余参数（使用合理范围内的值）
    sample_input = {
        "BOD": 5.2,
        "COD": 30.0,
        "氨氮": 3.5,
    }
    prediction, remaining = predict_remaining_parameters(sample_input, imputer, feature_names)
    print("已知参数:")
    print(sample_input)
    print("预测得到的完整参数集:")
    print(prediction)
    print("仅返回缺失参数:")
    print(remaining)

    eval_errors = evaluate_imputer(imputer, df)
    print("各参数预测均方误差:")
    for name, mse in eval_errors.items():
        print(f"{name}: {mse:.4f}")

