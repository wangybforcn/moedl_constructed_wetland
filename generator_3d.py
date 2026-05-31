# -*- coding: utf-8 -*-
"""3D OpenSCAD 与渲染模块。"""

import os
import shutil
import subprocess


def generate_openscad_script(parameters, output_scad):
    """生成 OpenSCAD 脚本，保存到 output_scad。"""
    length = parameters["长"]
    width = parameters["宽"]
    height = parameters["高"]
    compartments = parameters["compartments"]
    flow_type = parameters["flow_type"]
    inlet_diameter = parameters["inlet_diameter"]
    wall_thickness = parameters["wall_thickness"]
    partition_thickness = parameters["partition_thickness"]
    water_depth = parameters["water_depth"]

    lines = [
        f"length = {length:.2f};",
        f"width = {width:.2f};",
        f"height = {height:.2f};",
        f"wall_thickness = {wall_thickness:.4f};",
        f"partition_thickness = {partition_thickness:.4f};",
        f"water_depth = {water_depth:.4f};",
        f"compartments = {compartments};",
        f"inlet_diameter = {inlet_diameter:.4f};",
        f"flow_type = \"{flow_type}\";",
        "module wetland() {",
        "    difference() {",
        "        // 外壳",
        "        cube([length, width, height], center=false);",
        "        // 内部水体空间",
        "        translate([wall_thickness, wall_thickness, 0])",
        "            cube([length - 2*wall_thickness, width - 2*wall_thickness, water_depth], center=false);",
        "    }",
        "    // 隔室墙体",
        "    for (i = [1:compartments-1]) {",
        "        translate([i * (length - 2*wall_thickness) / compartments + wall_thickness - partition_thickness / 2, wall_thickness, 0])",
        "            cube([partition_thickness, width - 2*wall_thickness, water_depth], center=false);",
        "    }",
        "    // 进水口和出水口",
    ]

    if flow_type == "horizontal":
        lines += [
            "    translate([-inlet_diameter, width * 0.5 - inlet_diameter / 2, water_depth * 0.5])",
            "        rotate([0, 90, 0]) cylinder(h=inlet_diameter * 1.5, r=inlet_diameter / 2, center=true);",
            "    translate([length, width * 0.5 - inlet_diameter / 2, water_depth * 0.5])",
            "        rotate([0, 90, 0]) cylinder(h=inlet_diameter * 1.5, r=inlet_diameter / 2, center=true);",
        ]
    else:
        lines += [
            "    translate([length * 0.5 - inlet_diameter / 2, -inlet_diameter, water_depth * 0.5])",
            "        rotate([90, 0, 0]) cylinder(h=inlet_diameter * 1.5, r=inlet_diameter / 2, center=true);",
            "    translate([length * 0.5 - inlet_diameter / 2, width, water_depth * 0.5])",
            "        rotate([90, 0, 0]) cylinder(h=inlet_diameter * 1.5, r=inlet_diameter / 2, center=true);",
        ]

    lines += [
        "}",
        "wetland();",
    ]

    with open(output_scad, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_scad


def render_openscad(scad_path, output_stl=None, output_png=None, openscad_path="openscad"):
    """调用 OpenSCAD 渲染 SCAD 文件为 STL 和/或 PNG。"""
    if not shutil.which(openscad_path):
        raise FileNotFoundError("未找到 OpenSCAD 可执行文件。请安装 OpenSCAD 并确保它在 PATH 中，或使用 --openscad-path 指定路径。")

    if output_stl:
        subprocess.run([openscad_path, "-o", output_stl, scad_path], check=True)
    if output_png:
        subprocess.run(
            [
                openscad_path,
                "-o",
                output_png,
                "-D",
                "projection = \"orthographic\";",
                scad_path,
            ],
            check=True,
        )

    return output_stl, output_png
