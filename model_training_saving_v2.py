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
import datetime
import platform
import numpy as np
import pandas as pd
import joblib
import sklearn

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from data_audit import audit_csv, find_source_column, write_report


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
            df = df[df[col].isna() | df[col].between(lower, upper)]
            removed = original_len - len(df)
            if removed > 0:
                print(f"  {col}: 移除 {removed} 个异常值")
    
    elif method == "zscore":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        from scipy import stats
        for col in numeric_cols:
            present = df[col].notna()
            z_scores = np.abs(stats.zscore(df.loc[present, col], nan_policy="omit"))
            keep = pd.Series(True, index=df.index)
            keep.loc[present] = z_scores < threshold
            df = df[keep]
    
    return df


def load_dataset(
    csv_path="extracted_parameters.csv",
    min_non_na_ratio=0.2,
    remove_outliers_flag=False,
    require_reviewed=False,
    source_column=None,
    return_groups=False,
):
    """加载和清洗数据集。
    
    改进：
    - 添加异常值处理选项
    - 更详细的日志信息
    """
    print(f"\n=== 数据加载与清洗 ===")
    df = pd.read_csv(csv_path)
    print(f"原始数据：{len(df)} 行，{len(df.columns)} 列")

    if "quality_status" in df.columns:
        before = len(df)
        df = df[df["quality_status"].eq("accepted_candidate")]
        print(f"质量筛选：保留 {len(df)}/{before} 行 accepted_candidate")
    if require_reviewed and "review_status" in df.columns:
        before = len(df)
        df = df[df["review_status"].isin(("approved", "confirmed"))]
        print(f"人工复核筛选：保留 {len(df)}/{before} 行")
        if df.empty:
            raise ValueError("没有已人工确认的数据，review_status 应为 approved 或 confirmed")

    resolved_source = find_source_column(df, source_column)
    groups = df[resolved_source].copy() if resolved_source else None
    metadata_columns = [name for name in ("study_id", "source_id", "pdf_name") if name in df.columns]
    if metadata_columns:
        df = df.drop(columns=metadata_columns)
    
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
    
    if return_groups:
        aligned_groups = groups.loc[cleaned_df.index] if groups is not None else None
        return cleaned_df, aligned_groups, resolved_source
    return cleaned_df


def make_validation_splitter(n_splits, sample_count, groups=None, random_state=42):
    """优先使用来源分组验证；来源不足时降级为随机 KFold。"""
    if groups is not None:
        group_count = int(pd.Series(groups).nunique(dropna=True))
        if group_count >= 3:
            effective = min(n_splits, group_count)
            return GroupKFold(n_splits=effective), effective, "group_kfold"
    effective = min(n_splits, sample_count)
    return KFold(n_splits=effective, shuffle=True, random_state=random_state), effective, "kfold"


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
    """训练 Imputer。"""
    feature_names = df.columns.tolist()
    imputer = build_imputer(
        n_estimators=n_estimators,
        max_iter=max_iter,
        initial_strategy=initial_strategy
    )
    
    print(f"\n=== 训练 Imputer ===")
    print(f"开始训练 IterativeImputer（最多 {max_iter} 轮迭代）...")
    print(f"Initial Strategy: {initial_strategy}")
    
    imputer.fit(df)
    actual_iterations = getattr(imputer, "n_iter_", max_iter)
    print(f"训练完成，实际进行 {actual_iterations} 轮迭代。")
    return imputer, feature_names


def evaluate_imputer(imputer, df, mask_ratio=0.2, random_state=42):
    """随机遮挡一部分已知值，评估更接近真实缺失场景的填补误差。"""
    rng = np.random.default_rng(random_state)
    results = {}
    for col in df.columns:
        known_positions = np.flatnonzero(df[col].notna().to_numpy())
        if len(known_positions) < 3:
            continue
        mask_count = max(1, int(round(len(known_positions) * mask_ratio)))
        masked_positions = rng.choice(known_positions, size=mask_count, replace=False)
        test_df = df.copy()
        expected = test_df.iloc[masked_positions][col].to_numpy()
        test_df.iloc[masked_positions, test_df.columns.get_loc(col)] = np.nan
        evaluation_imputer = clone(imputer)
        evaluation_imputer.fit(test_df)
        imputed = evaluation_imputer.transform(test_df)
        imputed_col = imputed[masked_positions, list(df.columns).index(col)]
        
        mse = mean_squared_error(expected, imputed_col)
        mae = mean_absolute_error(expected, imputed_col)
        
        results[col] = {
            "MSE": {"value": mse, "unit": "MSE"},
            "MAE": {"value": mae, "unit": "MAE"}
        }
    return results


def make_supervised_pipeline_v1():
    """RandomForest 管道（原始）。"""
    return Pipeline([
        ("imputer", IterativeImputer(random_state=42, max_iter=10)),
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
        ("imputer", IterativeImputer(random_state=42, max_iter=10)),
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
    verbose=1,
    groups=None,
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
    
    if len(X) < 3:
        raise ValueError("监督训练至少需要 3 条具有目标值的样本")
    cv, n_splits, validation_strategy = make_validation_splitter(
        n_splits, len(X), groups=groups, random_state=random_state
    )
    fit_groups = groups if validation_strategy == "group_kfold" else None
    
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
        searcher.fit(X, y, groups=fit_groups)
        best_model = searcher.best_estimator_
        best_params = searcher.best_params_
        cv_rmse = np.sqrt(-searcher.best_score_)
        print(f"\n最佳参数: {best_params}")
        print(f"最佳 RMSE: {cv_rmse:.4f}")
        return best_model, np.array([cv_rmse]), best_params, validation_strategy, n_splits
    else:
        print(f"执行交叉验证（{n_splits}-折）...")
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            groups=fit_groups,
        )
        pipeline.fit(X, y)
        cv_rmse = np.sqrt(-scores)
        print(f"CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
        return pipeline, cv_rmse, None, validation_strategy, n_splits


def metrics_from_predictions(y, y_pred):
    return {
        "MSE": mean_squared_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "MAE": mean_absolute_error(y, y_pred),
        "MAPE": mean_absolute_percentage_error(y, y_pred),
        "MedAE": median_absolute_error(y, y_pred),
        "R2": r2_score(y, y_pred),
    }


def compute_detailed_metrics(model, X, y, n_splits=5, random_state=42, groups=None):
    """使用折外预测计算指标，避免在训练集自身上报告乐观结果。"""
    cv, _, strategy = make_validation_splitter(n_splits, len(X), groups, random_state)
    fit_groups = groups if strategy == "group_kfold" else None
    y_pred = cross_val_predict(model, X, y, cv=cv, groups=fit_groups, n_jobs=-1)
    return metrics_from_predictions(y, y_pred)


def compute_group_holdout_metrics(model, X, y, groups, random_state=42, test_size=0.2):
    """按完整来源留出独立测试集；来源不足时不生成留出指标。"""
    if groups is None or pd.Series(groups).nunique(dropna=True) < 3:
        return None
    if not 0 < test_size < 1:
        raise ValueError("holdout_size 必须在 0 和 1 之间")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(X, y, groups))
    holdout_model = clone(model)
    holdout_model.fit(X.iloc[train_index], y.iloc[train_index])
    prediction = holdout_model.predict(X.iloc[test_index])
    train_sources = sorted(str(value) for value in pd.Series(groups).iloc[train_index].dropna().unique())
    test_sources = sorted(str(value) for value in pd.Series(groups).iloc[test_index].dropna().unique())
    return {
        "metrics": metrics_from_predictions(y.iloc[test_index], prediction),
        "train_rows": int(len(train_index)),
        "test_rows": int(len(test_index)),
        "train_source_count": len(train_sources),
        "test_source_count": len(test_sources),
        "train_sources": train_sources,
        "test_sources": test_sources,
    }


def compute_baseline_cv(y, cv, groups=None):
    """用训练折中位数作为基线，返回各折 RMSE。"""
    X_dummy = np.zeros((len(y), 1))
    scores = cross_val_score(
        DummyRegressor(strategy="median"),
        X_dummy,
        y,
        cv=cv,
        groups=groups,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    return np.sqrt(-scores)


def run_imputer_training_v2(
    data_path="extracted_parameters.csv",
    output_path="trained_imputer.joblib",
    min_non_na_ratio=0.2,
    n_estimators=100,
    max_iter=10,
    initial_strategy="median",
    remove_outliers_flag=False,
    require_reviewed=False,
    report_path=None,
):
    """改进版 Imputer 训练。"""
    df = load_dataset(data_path, min_non_na_ratio, remove_outliers_flag, require_reviewed)
    constant_columns = [name for name in df.columns if df[name].nunique(dropna=True) <= 1]
    if constant_columns:
        df = df.drop(columns=constant_columns)
        print(f"移除常量特征：{constant_columns}")
    if df.empty or df.shape[1] < 2:
        raise ValueError("用于 Imputer 训练的数据不足：至少需要 1 行和 2 个数值特征")
    imputer, feature_names = train_imputer(
        df,
        n_estimators=n_estimators,
        max_iter=max_iter,
        initial_strategy=initial_strategy
    )
    
    audit_report = audit_csv(data_path)
    if report_path:
        write_report(audit_report, report_path)
    save_model({
        "schema_version": 1,
        "type": "imputer",
        "imputer": imputer,
        "feature_names": feature_names,
        "input_features": feature_names,
        "training_rows": len(df),
        "data_audit": audit_report,
        "training_environment": training_environment(),
    }, output_path)
    
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
    require_reviewed=False,
    source_column=None,
    holdout_size=0.2,
    report_path=None,
):
    """改进版监督学习训练。
    
    改进：
    - 支持多种模型
    - 详细的评估指标
    - 异常值处理
    """
    df, groups, resolved_source = load_dataset(
        data_path,
        min_non_na_ratio,
        remove_outliers_flag,
        require_reviewed,
        source_column=source_column,
        return_groups=True,
    )

    if target not in df.columns:
        raise ValueError(f"目标列 '{target}' 未找到")
    if df[target].nunique(dropna=True) <= 1:
        raise ValueError(f"目标列 '{target}' 没有足够变化，无法训练回归模型")

    df_supervised = df.dropna(subset=[target])
    X = df_supervised.drop(columns=[target])
    y = df_supervised[target]
    constant_features = [name for name in X.columns if X[name].nunique(dropna=True) <= 1]
    if constant_features:
        X = X.drop(columns=constant_features)
        print(f"移除常量输入特征：{constant_features}")
    if X.shape[1] == 0:
        raise ValueError("没有可用于监督训练的变化特征")
    supervised_groups = groups.loc[df_supervised.index] if groups is not None else None

    model, cv_rmse, best_params, validation_strategy, effective_n_splits = train_supervised_model(
        X, y,
        n_splits=n_splits,
        random_state=random_state,
        model_type=model_type,
        search=search,
        verbose=1,
        groups=supervised_groups,
    )

    validation_metrics = compute_detailed_metrics(
        model,
        X,
        y,
        n_splits=effective_n_splits,
        random_state=random_state,
        groups=supervised_groups,
    )
    holdout = compute_group_holdout_metrics(
        model, X, y, supervised_groups, random_state=random_state, test_size=holdout_size
    )
    baseline_cv, _, baseline_strategy = make_validation_splitter(
        effective_n_splits, len(X), supervised_groups, random_state
    )
    baseline_groups = supervised_groups if baseline_strategy == "group_kfold" else None
    baseline_rmse = compute_baseline_cv(y, baseline_cv, baseline_groups)
    baseline_comparison = {
        "strategy": "median",
        "rmse_mean": float(baseline_rmse.mean()),
        "rmse_std": float(baseline_rmse.std()),
        "model_improvement_ratio": float(1 - cv_rmse.mean() / baseline_rmse.mean())
        if baseline_rmse.mean() > 0 else None,
    }
    audit_report = audit_csv(data_path, source_column)
    experiment_report = {
        "type": "supervised_experiment",
        "target": target,
        "model_type": model_type,
        "random_state": random_state,
        "validation_strategy": validation_strategy,
        "n_splits": effective_n_splits,
        "source_column": resolved_source,
        "cv_rmse": {"value": float(cv_rmse.mean()), "std": float(cv_rmse.std())},
        "validation_metrics": {name: float(value) for name, value in validation_metrics.items()},
        "group_holdout": holdout,
        "baseline": baseline_comparison,
        "best_params": best_params,
        "data_audit": audit_report,
        "training_environment": training_environment(),
    }
    if report_path:
        write_report(experiment_report, report_path)

    save_model({
        "schema_version": 2,
        "type": "supervised",
        "model": model,
        "target": target,
        "feature_names": X.columns.tolist(),
        "input_features": X.columns.tolist(),
        "training_rows": len(X),
        "source_column": resolved_source,
        "validation_strategy": validation_strategy,
        "n_splits": effective_n_splits,
        "random_state": random_state,
        "data_sha256": audit_report["sha256"],
        "data_audit": audit_report,
        "validation_metrics": validation_metrics,
        "group_holdout": holdout,
        "baseline": baseline_comparison,
        "training_environment": training_environment(),
    }, output_path)

    print(f"\n=== 折外交叉验证指标 ===")
    print(f"目标: {target}")
    for metric_name, metric_value in validation_metrics.items():
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
        "n_splits": effective_n_splits,
        "model_type": model_type,
        "validation_metrics": validation_metrics,
        "validation_strategy": validation_strategy,
        "source_column": resolved_source,
        "group_holdout": holdout,
        "baseline": baseline_comparison,
        "data_audit": audit_report,
    }


def save_model(model, file_path="trained_model.joblib"):
    """保存模型。"""
    joblib.dump(model, file_path)
    print(f"模型已保存到: {file_path}")


def training_environment():
    return {
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


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
    parser.add_argument("--require-reviewed", action="store_true", help="仅使用已人工确认的数据")
    parser.add_argument("--source-column", default=None, help="论文/来源分组列，默认自动识别")
    parser.add_argument("--holdout-size", type=float, default=0.2, help="按来源留出的测试集比例")
    parser.add_argument("--report", default=None, help="训练/审计 JSON 报告路径")
    
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
            require_reviewed=args.require_reviewed,
            report_path=args.report,
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
            n_splits=args.n_splits,
            remove_outliers_flag=args.remove_outliers,
            require_reviewed=args.require_reviewed,
            source_column=args.source_column,
            holdout_size=args.holdout_size,
            report_path=args.report,
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
