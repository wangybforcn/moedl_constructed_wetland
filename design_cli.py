# -*- coding: utf-8 -*-
"""从 JSON 参数文件生成二维或三维人工湿地设计。"""

import argparse
import json
import sys

from design_3d import generate_from_params
from parameter_io import load_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="生成参数化人工湿地设计")
    parser.add_argument("--params", required=True, help="参数 JSON 文件")
    parser.add_argument("--model", default=None, help="可选的 Imputer joblib 模型")
    parser.add_argument("--output-prefix", default="wetland_design", help="输出文件前缀")
    parser.add_argument("--mode", choices=("2d", "3d", "all"), default="2d")
    parser.add_argument("--openscad-path", default="openscad")
    parser.add_argument("--render-timeout", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    result = generate_from_params(
        load_parameters(args.params),
        model_path=args.model,
        output_prefix=args.output_prefix,
        mode=args.mode,
        openscad_path=args.openscad_path,
        render_timeout=args.render_timeout,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    try:
        print(output)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((output + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
