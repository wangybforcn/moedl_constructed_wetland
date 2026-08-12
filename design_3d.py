# -*- coding: utf-8 -*-
"""设计生成服务：补全数值参数、校验设计参数并生成二维/三维输出。"""

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from generator_2d import build_image_prompt, generate_2d_image
from generator_3d import generate_openscad_script, render_openscad
from parameter_schema import DESIGN_FIELDS, apply_design_defaults, validate_parameters


def load_imputer_model(model_path: str) -> Tuple[Any, Optional[List[str]]]:
    model_obj = joblib.load(model_path)
    if isinstance(model_obj, dict):
        return model_obj.get("imputer"), model_obj.get("feature_names")
    return model_obj, None


def predict_missing_parameters(params: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    """使用已训练的数值 Imputer 补全模型覆盖的字段。"""
    imputer, feature_names = load_imputer_model(model_path)
    if imputer is None or feature_names is None:
        raise ValueError("模型文件中未找到可用的 imputer 或 feature_names")

    values = [np.nan if params.get(name) is None else float(params[name]) for name in feature_names]
    input_frame = pd.DataFrame([values], columns=feature_names, dtype=np.float64)
    imputed = imputer.transform(input_frame)
    completed = params.copy()
    for index, name in enumerate(feature_names):
        completed[name] = float(imputed[0, index])
    return completed


def prepare_design_parameters(params: Dict[str, Any], model_path: Optional[str] = None) -> Dict[str, Any]:
    """可选地使用模型补全数值字段，再应用默认值和物理约束。"""
    design_params = predict_missing_parameters(params, model_path) if model_path else params.copy()
    design_params = apply_design_defaults(design_params)
    design_params, errors = validate_parameters(design_params, DESIGN_FIELDS)
    if errors:
        raise ValueError("参数校验失败：\n- " + "\n- ".join(errors))
    return design_params


def generate_from_params(
    params: Dict[str, Any],
    model_path: Optional[str],
    output_prefix: str,
    mode: str = "all",
    openscad_path: str = "openscad",
    render_timeout: int = 180,
) -> Dict[str, Any]:
    """生成 2D/3D 结果。类别和结构默认值来自 schema，不由数值模型猜测。"""
    design_params = prepare_design_parameters(params, model_path)
    if mode not in ("2d", "3d", "all"):
        raise ValueError("mode 必须为 2d、3d 或 all")

    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    result: Dict[str, Any] = {"parameters": design_params, "mode": mode}
    if mode in ("2d", "all"):
        image_path = "{}_2d.png".format(output_prefix)
        generate_2d_image(design_params, image_path)
        result.update({"2d_image": image_path, "image_prompt": build_image_prompt(design_params)})
    if mode in ("3d", "all"):
        scad_path = "{}.scad".format(output_prefix)
        stl_path = "{}.stl".format(output_prefix)
        png_path = "{}_3d.png".format(output_prefix)
        generate_openscad_script(design_params, scad_path)
        render_info = render_openscad(
            scad_path,
            output_stl=stl_path,
            output_png=png_path,
            openscad_path=openscad_path,
            timeout=render_timeout,
        )
        result.update({"3d_scad": scad_path, "3d_stl": stl_path, "3d_png": png_path, "render": render_info})
    return result
