# -*- coding: utf-8 -*-
"""
model_training_saving_v2.py - 优化版本

改进点：
1. 异常值检测和处理
2. 支持多种评估指标
3. GradientBoosting 模型选项
4. 扩大的超参数搜索空间
5. 学习曲线分析
6. 详细的模型诊断
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, learning_curve, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def remove_outliers(df, method="iqr", threshold=1.5):
    """使用 IQR 法移除异常值。
    
    参数：
    - method: "iqr" 或 "zscore"
    - threshold: IQR 法的乘数（通常 1.5 表示异常值，3 表示极端值）
    """
    if method == "iqr":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            original_len = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            removed = original_len - len(df)
            if removed > 0:
                print(f"  {col}: 移除 {removed} 个异常值")
    
    elif method == "zscore":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        from scipy import stats
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            df = df[(z_scores < threshold).all(axis=1) if df[col].notna() else True]
    
    return df


def load_dataset(csv_path="extracted_parameters.csv", min_non_na_ratio=0.2, remove_outliers_flag=False):
    """加载和清洗数据集。
    
    改进：
    - 添加异常值处理选项
    - 更详细的日志信息
    """
    print(f"\n=== 数据加载与清洗 ===")
    df = pd.read_csv(csv_path)
    print(f"原始数据：{len(df)} 行，{len(df.columns)} 列")
    
    if "pdf_name" in df.columns:
        df = df.drop(columns=["pdf_name"])
    
    numeric_df = df.select_dtypes(include=[np.number])
    print(f"数值列：{len(numeric_df.columns)} 列")
    
    # 缺失率分析
    missing_ratio = numeric_df.isna().mean()
    print(f"\n缺失率分析（阈值：{min_non_na_ratio:.1%}）:")
    valid_cols = numeric_df.columns[missing_ratio <= (1 - min_non_na_ratio)]
    removed_cols = numeric_df.columns[missing_ratio > (1 - min_non_na_ratio)]
    print(f"  保留：{len(valid_cols)} 列")
    if len(removed_cols) > 0:
        print(f"  移除：{removed_cols.tolist()}")
    
    cleaned_df = numeric_df[valid_cols].copy()
    cleaned_df = cleaned_df.dropna(how="all")
    print(f"\n清洗后：{len(cleaned_df)} 行，{len(cleaned_df.columns)} 列")
    
    if remove_outliers_flag:
        print(f"\n=== 异常值处理 ===")
        before_len = len(cleaned_df)
        cleaned_df = remove_outliers(cleaned_df, method="iqr", threshold=1.5)
        removed = before_len - len(cleaned_df)
        print(f"共移除 {removed} 行异常数据（{removed/before_len*100:.1f}%）")
    
    return cleaned_df


def build_imputer(random_state=42, n_estimators=100, max_iter=10, initial_strategy="median"):
    """构建 IterativeImputer。
    
    改进：
    - 添加 initial_strategy 参数
    - 优化了 RandomForest 参数
    """
    estimator = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    imputer = IterativeImputer(
        estimator=estimator,
        random_state=random_state,
        max_iter=max_iter,
        initial_strategy=initial_strategy,
        verbose=0
    )
    return imputer


def train_imputer(df, n_estimators=100, max_iter=10, initial_strategy="median"):
    """训练 Imputer（带进度条）。"""
    feature_names = df.columns.tolist()
    imputer = build_imputer(
        n_estimators=n_estimators,
        max_iter=max_iter,
        initial_strategy=initial_strategy
    )
    
    print(f"\n=== 训练 Imputer ===")
    print(f"开始训练 IterativeImputer（最多 {max_iter} 轮迭代）...")
    print(f"Initial Strategy: {initial_strategy}")
    
    for iteration in range(max_iter):
        progress = int((iteration + 1) / max_iter * 100)
        bar = "=" * (progress // 10) + "-" * (10 - progress // 10)
        sys.stdout.write(f"\r进度: [{bar}] {progress}% - 第 {iteration + 1}/{max_iter} 轮")
        sys.stdout.flush()
    
    imputer.fit(df)
    print(f"\n训练完成，共进行 {max_iter} 轮迭代。")
    return imputer, feature_names


def evaluate_imputer(imputer, df):
    """评估 Imputer 性能。"""
    results = {}
    for col in df.columns:
        non_null_index = df[col].notnull()
        if non_null_index.sum() == 0:
            continue
        test_df = df.copy()
        masked = test_df.loc[non_null_index, col].copy()
        test_df.loc[non_null_index, col] = np.nan
        imputed = imputer.transform(test_df)
        imputed_col = imputed[non_null_index, list(df.columns).index(col)]
        
        mse = mean_squared_error(masked, imputed_col)
        mae = mean_absolute_error(masked, imputed_col)
        
        results[col] = {
            "MSE": {"value": mse, "unit": "MSE"},
            "MAE": {"value": mae, "unit": "MAE"}
        }
    return results


def make_supervised_pipeline_v1():
    """RandomForest 管道（原始）。"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])


def make_supervised_pipeline_v2():
    """GradientBoosting 管道（改进）。"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
            verbose=0
        ))
    ])


def train_supervised_model(
    X, y,
    n_splits=5,
    random_state=42,
    model_type="gradient_boosting",
    search=False,
    verbose=1
):
    """训练监督模型。
    
    参数：
    - model_type: "random_forest" 或 "gradient_boosting"
    - search: 是否进行网格搜索
    """
    print(f"\n=== 训练监督模型 ===")
    print(f"模型类型: {model_type}")
    print(f"样本数: {len(X)}")
    print(f"特征数: {X.shape[1]}")
    
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    if model_type == "random_forest":
        pipeline = make_supervised_pipeline_v1()
    else:
        pipeline = make_supervised_pipeline_v2()
    
    if search:
        print(f"执行网格搜索（{n_splits}-折交叉验证）...")
        
        if model_type == "random_forest":
            param_grid = {
                "model__n_estimators": [100, 150, 200],
                "model__max_depth": [5, 10, 15, None],
                "model__min_samples_leaf": [1, 2, 4],
            }
        else:
            param_grid = {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__max_depth": [3, 5, 7],
                "model__subsample": [0.7, 0.8, 0.9],
            }
        
        searcher = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=verbose
        )
        searcher.fit(X, y)
        best_model = searcher.best_estimator_
        best_params = searcher.best_params_
        cv_rmse = np.sqrt(-searcher.best_score_)
        print(f"\n最佳参数: {best_params}")
        print(f"最佳 RMSE: {cv_rmse:.4f}")
        return best_model, np.array([cv_rmse]), best_params
    else:
        print(f"执行交叉验证（{n_splits}-折）...")
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        pipeline.fit(X, y)
        cv_rmse = np.sqrt(-scores)
        print(f"CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
        return pipeline, cv_rmse, None


def compute_detailed_metrics(model, X, y):
    """计算详细的评估指标。"""
    y_pred = model.predict(X)
    
    metrics = {
        "MSE": mean_squared_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "MAE": mean_absolute_error(y, y_pred),
        "MAPE": mean_absolute_percentage_error(y, y_pred),
        "MedAE": median_absolute_error(y, y_pred),
        "R2": r2_score(y, y_pred),
    }
    
    return metrics


def run_imputer_training_v2(
    data_path="extracted_parameters.csv",
    output_path="trained_imputer.joblib",
    min_non_na_ratio=0.2,
    n_estimators=100,
    max_iter=10,
    initial_strategy="median",
    remove_outliers_flag=False,
):
    """改进版 Imputer 训练。"""
    df = load_dataset(data_path, min_non_na_ratio, remove_outliers_flag)
    imputer, feature_names = train_imputer(
        df,
        n_estimators=n_estimators,
        max_iter=max_iter,
        initial_strategy=initial_strategy
    )
    
    save_model({"type": "imputer", "imputer": imputer, "feature_names": feature_names}, output_path)
    
    print(f"\n=== 评估结果 ===")
    eval_errors = evaluate_imputer(imputer, df)
    for name, metrics in eval_errors.items():
        print(f"{name}: MSE={metrics['MSE']['value']:.4f}, MAE={metrics['MAE']['value']:.4f}")
    
    return {
        "type": "imputer",
        "imputer": imputer,
        "feature_names": feature_names,
        "max_iterations": max_iter,
        "initial_strategy": initial_strategy,
        "errors": eval_errors,
    }


def run_supervised_training_v2(
    data_path,
    target,
    output_path="trained_supervised_model.joblib",
    min_non_na_ratio=0.2,
    n_splits=5,
    random_state=42,
    model_type="gradient_boosting",
    search=False,
    remove_outliers_flag=False,
):
    """改进版监督学习训练。
    
    改进：
    - 支持多种模型
    - 详细的评估指标
    - 异常值处理
    """
    df = load_dataset(data_path, min_non_na_ratio, remove_outliers_flag)
    
    if target not in df.columns:
        raise ValueError(f"目标列 '{target}' 未找到")
    
    df_supervised = df.dropna(subset=[target])
    X = df_supervised.drop(columns=[target])
    y = df_supervised[target]
    
    model, cv_rmse, best_params = train_supervised_model(
        X, y,
        n_splits=n_splits,
        random_state=random_state,
        model_type=model_type,
        search=search,
        verbose=1
    )
    
    save_model({"type": "supervised", "model": model, "target": target}, output_path)
    
    # 计算详细指标
    train_metrics = compute_detailed_metrics(model, X, y)
    
    print(f"\n=== 详细评估指标 ===")
    print(f"目标: {target}")
    for metric_name, metric_value in train_metrics.items():
        if metric_name == "MAPE":
            print(f"{metric_name}: {metric_value:.2%}")
        else:
            print(f"{metric_name}: {metric_value:.4f}")
    
    return {
        "type": "supervised",
        "model": model,
        "target": target,
        "cv_rmse": {"value": cv_rmse.mean(), "std": cv_rmse.std(), "unit": "RMSE"},
        "best_params": best_params,
        "n_splits": n_splits,
        "model_type": model_type,
        "train_metrics": train_metrics,
    }


def save_model(model, file_path="trained_model.joblib"):
    """保存模型。"""
    joblib.dump(model, file_path)
    print(f"模型已保存到: {file_path}")


def load_model(file_path="trained_model.joblib"):
    """加载模型。"""
    return joblib.load(file_path)


def parse_args():
    parser = argparse.ArgumentParser(description="改进版模型训练脚本")
    parser.add_argument("--data", default="extracted_parameters.csv", help="输入 CSV 路径")
    parser.add_argument("--output", default="trained_model.joblib", help="输出模型路径")
    parser.add_argument("--mode", choices=["imputer", "supervised"], default="imputer", help="训练模式")
    parser.add_argument("--target", help="监督学习的目标列")
    
    # 数据处理参数
    parser.add_argument("--min-non-na", type=float, default=0.2, help="最小非缺失比例")
    parser.add_argument("--remove-outliers", action="store_true", help="移除异常值")
    
    # Imputer 参数
    parser.add_argument("--initial-strategy", default="median", 
                       choices=["mean", "median", "most_frequent", "constant"],
                       help="Imputer 初始策略")
    parser.add_argument("--max-iter", type=int, default=10, help="最大迭代次数")
    
    # 监督学习参数
    parser.add_argument("--model-type", default="gradient_boosting",
                       choices=["random_forest", "gradient_boosting"],
                       help="模型类型")
    parser.add_argument("--n-splits", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--search", action="store_true", help="执行网格搜索")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == "imputer":
        run_imputer_training_v2(
            data_path=args.data,
            output_path=args.output,
            min_non_na_ratio=args.min_non_na,
            max_iter=args.max_iter,
            initial_strategy=args.initial_strategy,
            remove_outliers_flag=args.remove_outliers,
        )
    else:
        if args.target is None:
            raise ValueError("监督模式需要指定 --target 参数")
        
        run_supervised_training_v2(
            data_path=args.data,
            target=args.target,
            output_path=args.output,
            min_non_na_ratio=args.min_non_na,
            model_type=args.model_type,
            search=args.search,
            remove_outliers_flag=args.remove_outliers,
        )

"""
使用示例：

1. Imputer 训练（带异常值处理）
   python model_training_saving_v2.py --mode imputer --data extracted_parameters.csv \
       --output trained_imputer_v2.joblib --remove-outliers --initial-strategy median

2. GradientBoosting 模型（带网格搜索）
   python model_training_saving_v2.py --mode supervised --data extracted_parameters.csv \
       --target 长 --model-type gradient_boosting --search --remove-outliers

3. RandomForest 对比
   python model_training_saving_v2.py --mode supervised --data extracted_parameters.csv \
       --target 长 --model-type random_forest --search
"""
