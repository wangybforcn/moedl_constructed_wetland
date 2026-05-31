# -*- coding: utf-8 -*-
"""Project API layer — 汇总各模块的接口供 GUI 调用。

目标：
- 将各模块的常用功能以稳定函数暴露出来，避免 GUI 直接依赖实现细节。
"""

import importlib
from typing import Optional

import data_to_excel
import joblib
import parameter_io

import model_training_saving_v2 as mts_v2


def extract_pdfs(input_folder: str, output_excel: str, output_csv: str, merge_xlsx: Optional[str] = None):
    """从 PDF 文件夹提取参数并保存为 Excel/CSV。

    返回训练数据的 DataFrame 路径（CSV 路径）。
    """
    data_to_excel.process_pdf(input_folder, output_excel, output_csv, merge_xlsx)
    return output_csv


def train_imputer(csv_path: str, output_path: str, **kwargs):
    """训练 imputer 并保存模型。"""
    return mts_v2.run_imputer_training_v2(data_path=csv_path, output_path=output_path, **kwargs)


def train_supervised(csv_path: str, target: str, output_path: str, **kwargs):
    """训练监督模型并保存。"""
    return mts_v2.run_supervised_training_v2(data_path=csv_path, target=target, output_path=output_path, **kwargs)


def load_model(path: str):
    """加载 joblib 模型并返回对象。"""
    return joblib.load(path)


def get_model_feature_names(model_path: str):
    """尝试从保存的模型中读取 feature_names 字段，若无则返回 None。"""
    return parameter_io.get_feature_names_from_model(model_path)


def compute_parameters(params: dict, model_path: str) -> dict:
    """仅计算并返回由模型填补后的完整参数集。"""
    design_module = importlib.import_module("3d")
    return design_module.predict_missing_parameters(params, model_path)


def generate_design(params: dict, model_path: str, output_prefix: str, mode: str = "all", openscad_path: str = "openscad"):
    """统一调用 3d.generate_from_params 并返回结果字典。"""
    design_module = importlib.import_module("3d")
    return design_module.generate_from_params(params, model_path=model_path, output_prefix=output_prefix, mode=mode, openscad_path=openscad_path)
