# -*- coding: utf-8 -*-
"""向后兼容入口；新代码请导入 design_3d。"""

from design_3d import generate_from_params, load_imputer_model, predict_missing_parameters, prepare_design_parameters

__all__ = ["generate_from_params", "load_imputer_model", "predict_missing_parameters", "prepare_design_parameters"]
