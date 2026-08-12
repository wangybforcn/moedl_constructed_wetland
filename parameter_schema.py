# -*- coding: utf-8 -*-
"""人工湿地项目的统一参数定义与物理约束。"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    dtype: str
    unit: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    default: Any = None
    choices: Tuple[str, ...] = ()
    required_for_design: bool = False
    description: str = ""


PARAMETER_SPECS: Dict[str, ParameterSpec] = {
    "长": ParameterSpec("长", "float", "m", 0.1, 10000.0, required_for_design=True, description="湿地池长度"),
    "宽": ParameterSpec("宽", "float", "m", 0.1, 10000.0, required_for_design=True, description="湿地池宽度"),
    "高": ParameterSpec("高", "float", "m", 0.1, 20.0, required_for_design=True, description="湿地池总高度"),
    "flow_type": ParameterSpec("flow_type", "category", choices=("horizontal", "vertical"), default="horizontal", required_for_design=True, description="水流方向"),
    "compartments": ParameterSpec("compartments", "int", "个", 1, 100, 1, required_for_design=True, description="隔室数量"),
    "inlet_diameter": ParameterSpec("inlet_diameter", "float", "m", 0.005, 5.0, 0.1, required_for_design=True, description="进出水管直径"),
    "water_depth": ParameterSpec("water_depth", "float", "m", 0.02, 19.0, 0.6, required_for_design=True, description="设计水深"),
    "wall_thickness": ParameterSpec("wall_thickness", "float", "m", 0.01, 2.0, 0.15, required_for_design=True, description="池壁厚度"),
    "partition_thickness": ParameterSpec("partition_thickness", "float", "m", 0.005, 1.0, 0.1, required_for_design=True, description="隔墙厚度"),
    "进水量": ParameterSpec("进水量", "float", "m³/d", 0.0, None, description="设计进水流量"),
    "水力停留时间": ParameterSpec("水力停留时间", "float", "h", 0.0, None, description="水力停留时间"),
    "水力负荷": ParameterSpec("水力负荷", "float", "m³/(m²·d)", 0.0, None, description="水力表面负荷"),
    "BOD": ParameterSpec("BOD", "float", "mg/L", 0.0, None),
    "COD": ParameterSpec("COD", "float", "mg/L", 0.0, None),
    "氨氮": ParameterSpec("氨氮", "float", "mg/L", 0.0, None),
    "总氮": ParameterSpec("总氮", "float", "mg/L", 0.0, None),
    "总磷": ParameterSpec("总磷", "float", "mg/L", 0.0, None),
    "pH": ParameterSpec("pH", "float", "", 0.0, 14.0),
    "DO": ParameterSpec("DO", "float", "mg/L", 0.0, None),
    "温度": ParameterSpec("温度", "float", "°C", -50.0, 80.0),
}

DESIGN_FIELDS: Tuple[str, ...] = tuple(
    name for name, spec in PARAMETER_SPECS.items() if spec.required_for_design
)


def coerce_value(name: str, value: Any) -> Any:
    """按照 schema 转换单个值；空字符串转换为 None。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    spec = PARAMETER_SPECS.get(name)
    if spec is None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value).strip()
    if spec.dtype == "category":
        text = str(value).strip().lower()
        aliases = {"水平": "horizontal", "水平潜流": "horizontal", "垂直": "vertical", "垂直流": "vertical"}
        return aliases.get(text, text)
    number = float(value)
    if spec.dtype == "int":
        if not number.is_integer():
            raise ValueError("{} 必须是整数".format(name))
        return int(number)
    return number


def apply_design_defaults(params: Dict[str, Any]) -> Dict[str, Any]:
    """只为有安全默认值的结构参数补默认值，不猜测池体尺寸。"""
    result = dict(params)
    for name in DESIGN_FIELDS:
        spec = PARAMETER_SPECS[name]
        if result.get(name) is None and spec.default is not None:
            result[name] = spec.default
    return result


def validate_parameters(
    params: Dict[str, Any], required_fields: Optional[Iterable[str]] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """返回转换后的参数和所有校验错误。"""
    converted: Dict[str, Any] = {}
    errors: List[str] = []
    for name, value in params.items():
        try:
            converted[name] = coerce_value(name, value)
        except (TypeError, ValueError) as exc:
            converted[name] = None
            errors.append("{}: {}".format(name, exc))

    for name, value in converted.items():
        if value is None or name not in PARAMETER_SPECS:
            continue
        spec = PARAMETER_SPECS[name]
        if spec.choices and value not in spec.choices:
            errors.append("{} 必须是 {} 之一".format(name, ", ".join(spec.choices)))
        if isinstance(value, (int, float)):
            if spec.minimum is not None and value < spec.minimum:
                errors.append("{} 不能小于 {} {}".format(name, spec.minimum, spec.unit))
            if spec.maximum is not None and value > spec.maximum:
                errors.append("{} 不能大于 {} {}".format(name, spec.maximum, spec.unit))

    for name in required_fields or ():
        if converted.get(name) is None:
            errors.append("缺少必填参数: {}".format(name))

    height = converted.get("高")
    water_depth = converted.get("water_depth")
    if height is not None and water_depth is not None and water_depth >= height:
        errors.append("water_depth 必须小于池体高度")
    length = converted.get("长")
    width = converted.get("宽")
    wall = converted.get("wall_thickness")
    if length is not None and wall is not None and 2 * wall >= length:
        errors.append("两侧壁厚之和必须小于池体长度")
    if width is not None and wall is not None and 2 * wall >= width:
        errors.append("两侧壁厚之和必须小于池体宽度")
    return converted, errors

