"""
model_training_saving.py
功能：读取预处理后的训练 CSV，训练并保存两类模型：
    - 非监督填补模型（IterativeImputer + 基学习器 RandomForest）
    - 监督回归模型（Pipeline：Scaler -> Imputer -> RandomForest）

脚本特性：
    - 提供命令行入口和可被其他脚本调用的函数接口 `run_imputer_training` / `run_supervised_training`。
    - 支持交叉验证、独立测试集、可选网格搜索调参（`--search`）。

常见使用场景：
    1. 用 `imputer` 模式训练缺失值自动填补模型。
    2. 用 `supervised` 模式对某个目标参数训练监督回归模型并评估性能。
"""

# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(csv_path="extracted_parameters.csv", min_non_na_ratio=0.2):
    """加载 CSV 并保留数值列。

    参数：
    - csv_path: CSV 文件路径，文件应为 `pdf_name` + 若干参数列（或仅参数列）
    - min_non_na_ratio: 某列至少要有多少比例的非空值才保留（防止缺失率过高的列参与训练）

    返回：清洗后的 DataFrame，仅保留数值列。
    """
    # 读取 CSV
    df = pd.read_csv(csv_path)
    # 如果包含 pdf_name（文件名列），删除它（模型训练只用数值特征）
    if "pdf_name" in df.columns:
        df = df.drop(columns=["pdf_name"])

    # 只保留数值类型的列
    numeric_df = df.select_dtypes(include=[np.number])

    # 过滤掉缺失率过高的特征（只保留非空比例 >= min_non_na_ratio 的列）
    valid_cols = numeric_df.columns[numeric_df.notna().mean() >= min_non_na_ratio]
    cleaned_df = numeric_df[valid_cols].copy()

    # 删除全为空的行（如果存在）
    cleaned_df = cleaned_df.dropna(how="all")
    return cleaned_df


def build_imputer(random_state=42, n_estimators=100, max_iter=10):
    # 使用随机森林作为 IterativeImputer 的基础估计器
    estimator = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    imputer = IterativeImputer(
        estimator=estimator, random_state=random_state, max_iter=max_iter, initial_strategy="mean"
    )
    return imputer


def train_imputer(df, n_estimators=100, max_iter=10):
    """Train an IterativeImputer on the numeric dataset."""
    feature_names = df.columns.tolist()
    imputer = build_imputer(n_estimators=n_estimators, max_iter=max_iter)
    imputer.fit(df)
    return imputer, feature_names


def evaluate_imputer(imputer, df):
    """通过依次屏蔽每一列来评估 imputer 的填补效果。

    对每个列：在原始非空的行上把该列置空，使用 imputer 填补，再计算 MSE。
    返回一个字典：{column: mse}
    """
    results = {}
    for col in df.columns:
        non_null_index = df[col].notnull()
        if non_null_index.sum() == 0:
            continue
        test_df = df.copy()
        # mask the column
        masked = test_df.loc[non_null_index, col].copy()
        test_df.loc[non_null_index, col] = np.nan
        imputed = imputer.transform(test_df)
        imputed_col = imputed[non_null_index, list(df.columns).index(col)]
        mse = mean_squared_error(masked, imputed_col)
        results[col] = mse
    return results
    


def make_supervised_pipeline(
    random_state=42,
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=1,
    max_features="auto",
    imputer_max_iter=10,
):
    # Pipeline 顺序：标准化 -> 缺失值填补 -> 随机森林回归
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "imputer",
                IterativeImputer(random_state=random_state, max_iter=imputer_max_iter, initial_strategy="mean"),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_supervised_model(
    X_train,
    y_train,
    n_splits=5,
    random_state=42,
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=1,
    max_features="auto",
    imputer_max_iter=10,
    search=False,
):
    if X_train.shape[1] == 0:
        raise ValueError("No predictor features available after removing the target column.")

    # 构建 pipeline（含 scaler, imputer, model）
    pipeline = make_supervised_pipeline(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        imputer_max_iter=imputer_max_iter,
    )

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 当 search=True 时执行网格搜索以寻找最优超参数
    if search:
        param_grid = {
            "model__n_estimators": [100, 150, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["auto", "sqrt", "log2"],
        }
        searcher = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        searcher.fit(X_train, y_train)
        best_model = searcher.best_estimator_
        best_params = searcher.best_params_
        cv_rmse = np.sqrt(-searcher.best_score_)
        # 返回 best_model 与 cv_rmse（包装为数组以兼容后续处理）
        return best_model, np.array([cv_rmse]), best_params

    # 非搜索模式：使用交叉验证评估并返回在全部训练数据上拟合好的模型
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    pipeline.fit(X_train, y_train)
    return pipeline, np.sqrt(-scores), None
    


def save_model(model, file_path="trained_model.joblib"):
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")


def load_model(file_path="trained_model.joblib"):
    return joblib.load(file_path)


def evaluate_supervised_model(model, X, y):
    """评估监督回归模型：返回 (mse, r2)。

    - model: 已训练的 pipeline 或回归模型
    - X, y: 测试集特征与目标
    """
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, r2


def run_imputer_training(
    data_path="extracted_parameters.csv",
    output_path="trained_imputer.joblib",
    min_non_na_ratio=0.2,
    n_estimators=100,
    max_iter=10,
):
    df = load_dataset(data_path, min_non_na_ratio=min_non_na_ratio)
    imputer, feature_names = train_imputer(df, n_estimators=n_estimators, max_iter=max_iter)
    save_model({"type": "imputer", "imputer": imputer, "feature_names": feature_names}, output_path)

    eval_errors = evaluate_imputer(imputer, df)
    print("各参数预测均方误差:")
    for name, mse in eval_errors.items():
        print(f"{name}: {mse:.4f}")

    return {
        "type": "imputer",
        "imputer": imputer,
        "feature_names": feature_names,
        "errors": eval_errors,
    }


def run_supervised_training(
    data_path,
    target,
    output_path="trained_supervised_model.joblib",
    min_non_na_ratio=0.2,
    test_size=0.2,
    n_splits=5,
    random_state=42,
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=1,
    max_features="auto",
    imputer_max_iter=10,
    search=False,
):
    df = load_dataset(data_path, min_non_na_ratio=min_non_na_ratio)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    df_supervised = df.dropna(subset=[target])
    X = df_supervised.drop(columns=[target])
    y = df_supervised[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    model, cv_rmse, best_params = train_supervised_model(
        X_train,
        y_train,
        n_splits=n_splits,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        imputer_max_iter=imputer_max_iter,
        search=search,
    )
    save_model({"type": "supervised", "model": model, "target": target}, output_path)

    train_mse, train_r2 = evaluate_supervised_model(model, X_train, y_train)
    test_mse, test_r2 = evaluate_supervised_model(model, X_test, y_test)

    print(f"监督学习目标: {target}")
    if search and best_params is not None:
        print(f"最佳参数: {best_params}")
    print(f"交叉验证 RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"训练集 MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
    print(f"测试集 MSE: {test_mse:.4f}, R2: {test_r2:.4f}")

    return {
        "type": "supervised",
        "model": model,
        "target": target,
        "cv_rmse": cv_rmse,
        "best_params": best_params,
        "train_mse": train_mse,
        "train_r2": train_r2,
        "test_mse": test_mse,
        "test_r2": test_r2,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train and save imputer or supervised model for parameter prediction.")
    parser.add_argument("--data", default="extracted_parameters.csv", help="Path to the input CSV dataset.")
    parser.add_argument("--output", default="trained_model.joblib", help="Path to save the trained model file.")
    parser.add_argument("--mode", choices=["imputer", "supervised"], default="imputer", help="Training mode: imputer or supervised.")
    parser.add_argument("--target", default=None, help="Supervised target column name. Required when mode=supervised.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out test size for supervised evaluation.")
    parser.add_argument("--min-non-na", type=float, default=0.2, help="Minimum non-missing ratio to keep feature columns.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of cross-validation splits for supervised training.")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees for RandomForest estimators.")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum depth for RandomForest.")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="Minimum samples per leaf for RandomForest.")
    parser.add_argument("--max-features", default="auto", help="Maximum features for RandomForest.")
    parser.add_argument("--search", action="store_true", help="Use grid search to tune RandomForest hyperparameters.")
    parser.add_argument("--max-iter", type=int, default=10, help="Maximum iterations for IterativeImputer.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "imputer":
        run_imputer_training(
            data_path=args.data,
            output_path=args.output,
            min_non_na_ratio=args.min_non_na,
            n_estimators=args.n_estimators,
            max_iter=args.max_iter,
        )
    else:
        if args.target is None:
            raise ValueError("mode=supervised 时必须指定 --target 参数")

        run_supervised_training(
            data_path=args.data,
            target=args.target,
            output_path=args.output,
            min_non_na_ratio=args.min_non_na,
            test_size=args.test_size,
            n_splits=args.n_splits,
            random_state=42,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            imputer_max_iter=args.max_iter,
            search=args.search,
        )
"""
非监督填补模式
    python model_training_saving.py --mode imputer --data extracted_parameters.csv --output trained_imputer.joblib --max-iter 15
监督模型模式
    python model_training_saving.py --mode supervised --data extracted_parameters.csv --target BOD --output trained_BOD_model.joblib --max-depth 10 --min-samples-leaf 2 --max-features sqrt
启用网格搜索
    python model_training_saving.py --mode supervised --data extracted_parameters.csv --target BOD --output trained_BOD_model.joblib --search
"""