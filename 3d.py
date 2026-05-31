# -*- coding: utf-8 -*-
"""3D 设计生成模块：负责缺失参数预测、OpenSCAD 脚本生成与渲染。"""

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from generator_2d import build_image_prompt, generate_2d_image
from generator_3d import generate_openscad_script, render_openscad


def load_imputer_model(model_path: str) -> Tuple[Any, Optional[List[str]]]:
    model_obj = joblib.load(model_path)
    if isinstance(model_obj, dict):
        imputer = model_obj.get("imputer")
        feature_names = model_obj.get("feature_names")
        return imputer, feature_names
    return model_obj, None


def predict_missing_parameters(params: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    """使用已训练的 Imputer 模型填补缺失参数。"""
    imputer, feature_names = load_imputer_model(model_path)
    if imputer is None or feature_names is None:
        raise ValueError("模型文件中未找到可用的 imputer 或 feature_names")

    values = []
    for feature in feature_names:
        value = params.get(feature)
        if value is None:
            values.append(np.nan)
        else:
            values.append(float(value))

    input_array = np.array([values], dtype=np.float64)
    imputed = imputer.transform(input_array)

    completed = params.copy()
    for idx, feature in enumerate(feature_names):
        completed[feature] = float(imputed[0, idx])

    return completed


def generate_from_params(
    params: Dict[str, Any],
    model_path: Optional[str],
    output_prefix: str,
    mode: str = "all",
    openscad_path: str = "openscad",
) -> Dict[str, Any]:
    """根据参数生成 2D/3D 输出，并返回路径与提示信息。"""
    design_params = params.copy()
    if model_path:
        design_params = predict_missing_parameters(design_params, model_path)

    result: Dict[str, Any] = {"parameters": design_params, "mode": mode}

    if mode in ("2d", "all"):
        image_path = f"{output_prefix}_2d.png"
        generate_2d_image(design_params, image_path)
        result["2d_image"] = image_path
        result["image_prompt"] = build_image_prompt(design_params)

    if mode in ("3d", "all"):
        scad_path = f"{output_prefix}.scad"
        stl_path = f"{output_prefix}.stl"
        png_path = f"{output_prefix}_3d.png"
        generate_openscad_script(design_params, scad_path)
        render_openscad(scad_path, output_stl=stl_path, output_png=png_path, openscad_path=openscad_path)
        result["3d_scad"] = scad_path
        result["3d_stl"] = stl_path
        result["3d_png"] = png_path

    return result
