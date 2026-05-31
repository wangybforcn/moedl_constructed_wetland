# -*- coding: utf-8 -*-
"""参数输入/输出模块：独立处理参数归一化、保存、加载和模型特征读取。"""

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib


def normalize_parameter_value(value: Any) -> Optional[Any]:
    """将字符串值归一化为数字或文本，空值保留为 None。"""
    if value is None:
        return None
    value_str = str(value).strip()
    if value_str == "":
        return None
    try:
        return float(value_str)
    except ValueError:
        return value_str


def normalize_parameters(raw_inputs: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    """将原始输入字典中的值转换为 float/string/None。"""
    return {key: normalize_parameter_value(value) for key, value in raw_inputs.items()}


PARAMETER_UNITS = {
    "长": "m",
    "宽": "m",
    "高": "m",
    "flow_type": "",
    "compartments": "个",
    "inlet_diameter": "m",
    "water_depth": "m",
    "wall_thickness": "m",
    "partition_thickness": "m",
    "进水量": "m³/d",
    "水力停留时间": "h",
    "水力负荷": "kg/(m²·d)",
}


def get_feature_display_name(feature_name: str) -> str:
    """返回带单位的特征显示名称。"""
    unit = PARAMETER_UNITS.get(feature_name)
    if unit:
        return f"{feature_name} ({unit})"
    return feature_name


def get_feature_units(feature_name: str) -> Optional[str]:
    """返回特征单位，如果没有则返回 None。"""
    return PARAMETER_UNITS.get(feature_name)


def ensure_all_features(params: Dict[str, Any], feature_names: Iterable[str]) -> Dict[str, Optional[Any]]:
    """确保输出参数包含所有特征名称，缺失值设置为 None。"""
    return {name: params.get(name, None) for name in feature_names}


def save_parameters(params: Dict[str, Any], file_path: str) -> str:
    """将参数保存为 JSON 文件。"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    return file_path


def load_parameters(file_path: str) -> Dict[str, Any]:
    """从 JSON 文件加载参数。"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_names_from_model(model_path: str) -> Optional[List[str]]:
    """从保存的模型中读取 feature_names。"""
    model = joblib.load(model_path)
    if isinstance(model, dict):
        if "feature_names" in model:
            return model["feature_names"]
        if "imputer" in model and "feature_names" in model:
            return model.get("feature_names")
    return None


def validate_parameters(params: Dict[str, Any], required_fields: Optional[Iterable[str]] = None) -> Tuple[bool, List[str]]:
    """验证必填字段是否存在并且不为 None。"""
    if required_fields is None:
        return True, []
    missing = [name for name in required_fields if params.get(name) is None]
    return len(missing) == 0, missing
