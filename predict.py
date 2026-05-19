# -*- coding: utf-8 -*-
"""
人工湿地参数预测交互程序
加载训练好的 IterativeImputer 模型，根据用户输入的已知参数预测缺失参数。
"""

import sys
from model_training_saving import load_model, predict_remaining_parameters


def main():
    model_path = "trained_imputer.joblib"

    print("=" * 50)
    print("  人工湿地参数预测系统")
    print("=" * 50)

    # 加载模型
    try:
        imputer, feature_names = load_model(model_path)
    except FileNotFoundError:
        print(f"错误：未找到模型文件 {model_path}")
        print("请先运行 model_training_saving.py 训练并保存模型。")
        sys.exit(1)

    print(f"模型加载成功！共 {len(feature_names)} 个可预测参数。\n")

    # 显示可用参数及单位
    param_units = {
        "BOD": "mg/L", "COD": "mg/L", "氨氮": "mg/L", "总氮": "mg/L",
        "总磷": "mg/L", "pH": "", "水温(气温)": "°C",
        "水力负荷": "m³/(m²·d)", "水力停留时间": "d", "进水量": "m³/d",
        "温度": "°C", "DO": "mg/L", "溶解氧": "mg/L",
        "宽": "m", "高": "m", "长": "m",
        "种植密度": "株/m²", "覆盖度": "%", "CNP": "",
    }

    print("可用参数列表：")
    for i, name in enumerate(feature_names, 1):
        unit = param_units.get(name, "")
        label = f"{name}({unit})" if unit else name
        print(f"  {i:>2}. {label}")

    print("\n使用说明：")
    print("  - 输入格式：参数名=数值，多个参数用逗号分隔")
    print("  - 示例：BOD=5.2,COD=30.0,氨氮=3.5")
    print("  - 输入 q 退出程序")
    print("  - 直接回车查看示例预测\n")

    while True:
        user_input = input("请输入已知参数：").strip()

        if user_input.lower() == "q":
            print("退出程序。")
            break

        if not user_input:
            # 示例预测
            user_input = "BOD=5.2,COD=30.0,氨氮=3.5"
            print(f"使用示例输入：{user_input}")

        # 解析输入
        known_params = {}
        parse_errors = []

        for part in user_input.split(","):
            part = part.strip()
            if "=" not in part:
                parse_errors.append(f"格式错误：'{part}'，应为 参数名=数值")
                continue
            key, val = part.split("=", 1)
            key = key.strip()
            val = val.strip()
            try:
                known_params[key] = float(val)
            except ValueError:
                parse_errors.append(f"数值无效：'{val}'（参数 {key}）")

        if parse_errors:
            print("\n输入有误：")
            for err in parse_errors:
                print(f"  - {err}")
            print()
            continue

        # 检查参数名是否有效
        invalid_keys = [k for k in known_params if k not in feature_names]
        if invalid_keys:
            print(f"\n以下参数名不在模型中：{', '.join(invalid_keys)}")
            print(f"有效参数：{', '.join(feature_names)}\n")
            continue

        if not known_params:
            print("至少需要输入一个已知参数。\n")
            continue

        # 预测
        try:
            prediction, remaining = predict_remaining_parameters(
                known_params, imputer, feature_names
            )
        except ValueError as e:
            print(f"预测错误：{e}\n")
            continue

        # 输出结果
        print("\n" + "-" * 50)
        print("【已知参数】")
        for k, v in known_params.items():
            unit = param_units.get(k, "")
            print(f"  {k}: {v} {unit}")

        print("\n【预测结果 - 缺失参数】")
        for k, v in remaining.items():
            unit = param_units.get(k, "")
            print(f"  {k}: {v:.4f} {unit}")

        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
