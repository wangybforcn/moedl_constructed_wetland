# -*- coding: utf-8 -*-
"""2D 平面图与图像生成提示模块。"""

from PIL import Image, ImageDraw


def generate_2d_image(parameters, output_path):
    """生成参数驱动的 2D 平面图并保存到指定路径。"""
    length = parameters["长"]
    width = parameters["宽"]
    flow_type = parameters["flow_type"]
    compartments = int(parameters["compartments"])

    scale = max(40, min(120, 400 / max(length, width)))
    img_w = int(max(640, width * scale + 340))
    img_h = int(max(560, length * scale + 220))

    image = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(image)

    left = 80
    top = 80
    right = left + int(width * scale)
    bottom = top + int(length * scale)

    draw.rectangle([left, top, right, bottom], outline="black", width=4)

    for i in range(1, compartments):
        y = top + int(i * length * scale / compartments)
        draw.line([left, y, right, y], fill="blue", width=3)
        draw.text((right + 10, y - 10), f"区{i}", fill="black")

    if flow_type == "horizontal":
        arrow_start = (left - 10, top + int(length * scale * 0.5))
        arrow_end = (left + 30, top + int(length * scale * 0.5))
        draw.line([arrow_start, arrow_end], fill="green", width=4)
        draw.polygon(
            [
                (left + 30, top + int(length * scale * 0.5)),
                (left + 20, top + int(length * scale * 0.5) - 8),
                (left + 20, top + int(length * scale * 0.5) + 8),
            ],
            fill="green",
        )
        draw.text((left - 60, top + int(length * scale * 0.5) - 20), "进水", fill="green")
        draw.text((right + 15, top + int(length * scale * 0.5) - 20), "出水", fill="red")
    else:
        arrow_start = (left + int(width * scale * 0.5), bottom + 10)
        arrow_end = (left + int(width * scale * 0.5), bottom - 30)
        draw.line([arrow_start, arrow_end], fill="green", width=4)
        draw.polygon(
            [
                (left + int(width * scale * 0.5), bottom - 30),
                (left + int(width * scale * 0.5) - 8, bottom - 20),
                (left + int(width * scale * 0.5) + 8, bottom - 20),
            ],
            fill="green",
        )
        draw.text((left + int(width * scale * 0.5) + 10, bottom - 40), "进水", fill="green")
        draw.text((left + int(width * scale * 0.5) + 10, top - 30), "出水", fill="red")

    annotation_x = right + 20
    annotation_y = top
    def format_optional(name, suffix=""):
        value = parameters.get(name)
        return "{}=未提供".format(name) if value is None else "{}={:.2f}{}".format(name, value, suffix)

    info_lines = [
        f"长={length:.1f}m",
        f"宽={width:.1f}m",
        f"高={parameters['高']:.1f}m",
        f"流型={flow_type}",
        f"隔间={compartments}",
        format_optional("进水量", "m³/d"),
        format_optional("水力停留时间", "h"),
        format_optional("水力负荷", "m³/(m²·d)"),
        f"设计水深={parameters['water_depth']:.2f}m",
        f"进水管径={parameters['inlet_diameter']:.2f}m",
    ]
    for idx, text in enumerate(info_lines):
        draw.text((annotation_x, annotation_y + idx * 24), text, fill="black")

    legend_top = bottom + 30
    draw.rectangle([left, legend_top, left + 120, legend_top + 70], outline="black", width=2)
    draw.text((left + 10, legend_top + 5), "图例", fill="black")
    draw.rectangle([left + 10, legend_top + 25, left + 30, legend_top + 45], fill="black")
    draw.text((left + 40, legend_top + 25), "湿地池体", fill="black")

    image.save(output_path)
    return output_path


def build_image_prompt(parameters):
    """生成可用于图像生成模型的文本提示。"""
    prompt_lines = [
        "请绘制一个人工湿地工程平面图，",
        f"尺寸约为 {parameters['长']:.1f}m x {parameters['宽']:.1f}m x {parameters['高']:.1f}m，",
        f"流型为 {parameters['flow_type']}，共有 {parameters['compartments']} 个隔间，",
        "请以工程制图风格展示进水、出水、隔断和主要流向。",
        "图中应标注隔间编号、水流方向和进水口位置。",
    ]
    if parameters.get("水力停留时间") is not None:
        prompt_lines.insert(3, "水力停留时间约 {:.1f} 小时，".format(parameters["水力停留时间"]))
    if parameters.get("水力负荷") is not None:
        prompt_lines.insert(4, "水力负荷约 {:.2f} m³/(m²·d)，".format(parameters["水力负荷"]))
    return "".join(prompt_lines)
