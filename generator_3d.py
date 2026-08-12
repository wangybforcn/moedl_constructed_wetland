# -*- coding: utf-8 -*-
"""3D OpenSCAD 脚本生成、可执行程序探测与渲染。"""

import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_OPENSCAD_LOCATIONS = (
    r"C:\Program Files\OpenSCAD\openscad.exe",
    r"C:\Program Files (x86)\OpenSCAD\openscad.exe",
)


class OpenSCADRenderError(RuntimeError):
    """OpenSCAD 命令失败、超时或未产生有效文件。"""


def find_openscad(openscad_path="openscad"):
    """解析显式路径、PATH 和 Windows 常见安装目录。"""
    candidate = Path(str(openscad_path)).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    located = shutil.which(str(openscad_path))
    if located:
        return located
    if str(openscad_path).lower() in ("openscad", "openscad.exe"):
        for location in DEFAULT_OPENSCAD_LOCATIONS:
            if Path(location).is_file():
                return location
    return None


def _render_one(executable, scad_path, output_path, extra_args, timeout):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    command = [executable, "-o", str(output)] + list(extra_args) + [str(scad_path)]
    log_path = output.with_suffix(output.suffix + ".log")
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(
            "COMMAND: {}\n\nERROR: OpenSCAD 渲染超过 {} 秒\n".format(
                subprocess.list2cmdline(command), timeout
            ),
            encoding="utf-8",
        )
        raise OpenSCADRenderError(
            "OpenSCAD 渲染超过 {} 秒: {}，日志: {}".format(timeout, output.name, log_path)
        ) from exc
    log_text = "COMMAND: {}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}\n".format(
        subprocess.list2cmdline(command), completed.stdout, completed.stderr
    )
    log_path.write_text(log_text, encoding="utf-8")
    if completed.returncode != 0:
        raise OpenSCADRenderError(
            "OpenSCAD 渲染失败（退出码 {}），日志: {}".format(completed.returncode, log_path)
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise OpenSCADRenderError("OpenSCAD 未生成有效文件: {}，日志: {}".format(output, log_path))
    return str(output), str(log_path)


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
        "floor_thickness = wall_thickness;",
        "module wetland() {",
        "    difference() {",
        "        union() {",
        "            // 带底板且顶部敞开的池体外壳",
        "            difference() {",
        "                cube([length, width, height], center=false);",
        "                translate([wall_thickness, wall_thickness, floor_thickness])",
        "                    cube([length - 2*wall_thickness, width - 2*wall_thickness, height], center=false);",
        "            }",
        "            // 隔室墙体，交替设置过水口",
        "            for (i = [1:compartments-1]) {",
        "                difference() {",
        "                    translate([i * (length - 2*wall_thickness) / compartments + wall_thickness - partition_thickness / 2, wall_thickness, floor_thickness])",
        "                        cube([partition_thickness, width - 2*wall_thickness, water_depth], center=false);",
        "                    opening_y = (i % 2 == 0) ? wall_thickness : width - 2*wall_thickness;",
        "                    translate([i * (length - 2*wall_thickness) / compartments + wall_thickness - partition_thickness, opening_y, floor_thickness])",
        "                        cube([partition_thickness * 2, wall_thickness * 2, water_depth * 0.35], center=false);",
        "                }",
        "            }",
        "        }",
        "        // 在池壁上切出进水口和出水口",
    ]

    if flow_type == "horizontal":
        lines += [
            "        translate([-wall_thickness, width * 0.5, floor_thickness + water_depth * 0.7])",
            "            rotate([0, 90, 0]) cylinder(h=wall_thickness * 3, r=inlet_diameter / 2, center=false, $fn=48);",
            "        translate([length - 2*wall_thickness, width * 0.5, floor_thickness + water_depth * 0.7])",
            "            rotate([0, 90, 0]) cylinder(h=wall_thickness * 3, r=inlet_diameter / 2, center=false, $fn=48);",
        ]
    else:
        lines += [
            "        translate([length * 0.5, -wall_thickness, floor_thickness + water_depth * 0.7])",
            "            rotate([-90, 0, 0]) cylinder(h=wall_thickness * 3, r=inlet_diameter / 2, center=false, $fn=48);",
            "        translate([length * 0.5, width - 2*wall_thickness, floor_thickness + water_depth * 0.7])",
            "            rotate([-90, 0, 0]) cylinder(h=wall_thickness * 3, r=inlet_diameter / 2, center=false, $fn=48);",
        ]

    lines += [
        "    }",
    ]

    lines += [
        "}",
        "wetland();",
    ]

    with open(output_scad, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_scad


def render_openscad(scad_path, output_stl=None, output_png=None, openscad_path="openscad", timeout=180):
    """渲染 SCAD，并校验 STL/PNG 产物，日志写入相邻 ``*.log`` 文件。"""
    executable = find_openscad(openscad_path)
    if not executable:
        raise FileNotFoundError("未找到 OpenSCAD 可执行文件。请安装 OpenSCAD 并确保它在 PATH 中，或使用 --openscad-path 指定路径。")
    scad = Path(scad_path)
    if not scad.is_file():
        raise FileNotFoundError("SCAD 文件不存在: {}".format(scad))
    if not output_stl and not output_png:
        raise ValueError("必须至少指定 output_stl 或 output_png")
    logs = []
    if output_stl:
        _, log_path = _render_one(executable, scad, output_stl, (), timeout)
        logs.append(log_path)
    if output_png:
        _, log_path = _render_one(
            executable,
            scad,
            output_png,
            ("--imgsize=1200,800", "--viewall", "--autocenter", "--projection=o"),
            timeout,
        )
        logs.append(log_path)
    return {"stl": output_stl, "png": output_png, "logs": logs, "openscad": executable}
