"""
data_to_excel.py
功能：从论文 PDF 中提取数值参数并保存为 Excel/CSV，支持中英文参数别名与合并已有训练表。

主要步骤：
1. 定义参数别名字典 `param_aliases` 和分类 `category_map`，用于将不同语言/写法统一为规范列名。
2. 使用 pdfminer 提取每页文本，基于正则抽取“数值 + 单位”模式。
3. 在数值附近查找参数关键词并归一化为规范名（`keyword_to_canonical`）。
4. 把提取出的记录写入 DataFrame，生成汇总表与训练数据（每个 PDF 一行，参数为列）。
5. 可选择合并已有的训练 Excel 文件。

使用说明（命令行）：
python data_to_excel.py -i ./pdf -o extracted_parameters.xlsx -c extracted_parameters.csv -m existing.xlsx
"""

# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path as pt

import pandas as pd
from pdfminer.high_level import extract_pages

# -----------------------------
# 参数与词表
# -----------------------------

# 中文/英文参数别名映射：用于统一参数名称并支持不同语言的论文内容
param_aliases = {
    "BOD": ["BOD", "BOD5"],
    "COD": ["COD"],
    "氨氮": ["氨氮", "NH3-N", "NH3N", "Ammonia nitrogen", "ammonia nitrogen", "ammonia-n", "ammonia"],
    "总氮": ["总氮", "TN", "Total nitrogen", "total nitrogen"],
    "总磷": ["总磷", "TP", "Total phosphorus", "total phosphorus"],
    "pH": ["pH"],
    "水温(气温)": ["水温", "气温", "water temperature", "water temp", "temperature", "temp"],
    "CNP": ["CNP"],
    "水力负荷": ["水力负荷", "hydraulic load", "hydraulic loading rate", "HLR"],
    "水力停留时间": ["水力停留时间", "hydraulic retention time", "HRT", "retention time"],
    "进水量": ["进水量", "inflow", "influent flow", "influent volume", "flow rate"],
    "填料类型(结构)": ["填料类型", "填料类型(结构)", "media type", "filter media", "media material"],
    "长": ["长", "length"],
    "宽": ["宽", "width"],
    "高": ["高", "height", "depth"],
    "垂直": ["垂直", "vertical"],
    "水平潜流": ["水平潜流", "horizontal subsurface flow", "horizontal flow"],
    "分层结构": ["分层结构", "layered structure", "stratified structure"],
    "湿地类型": ["湿地类型", "wetland type", "wetland system", "constructed wetland"],
    "芦苇": ["芦苇", "reed"],
    "美人蕉": ["美人蕉", "canna", "canna indica"],
    "香蒲": ["香蒲", "cattail", "typha"],
    "菖蒲": ["菖蒲", "acorus", "iris pseudacorus"],
    "再力花": ["再力花", "iris"],
    "种植密度": ["种植密度", "planting density", "plant density"],
    "覆盖度": ["覆盖度", "coverage", "canopy coverage"],
    "温度": ["温度", "temperature", "temp"],
    "DO": ["DO", "dissolved oxygen"],
    "溶解氧": ["溶解氧", "dissolved oxygen"],
    "地区": ["地区", "region", "area"],
    "气候": ["气候", "climate"],
    "季节": ["季节", "season"],
    "光照": ["光照", "illumination", "sunlight", "light intensity"],
}

category_map = {
    "水质参数": ["BOD", "COD", "氨氮", "总氮", "总磷", "pH", "水温(气温)", "CNP", "DO", "溶解氧"],
    "运行参数": ["水力负荷", "水力停留时间", "进水量", "填料类型(结构)"],
    "湿地(类型)结构": ["长", "宽", "高", "垂直", "水平潜流", "分层结构", "湿地类型"],
    "植物参数": ["芦苇", "美人蕉", "香蒲", "菖蒲", "再力花", "种植密度", "覆盖度"],
    "环境信息": ["温度", "地区", "气候", "季节", "光照"],
}

keyword_to_canonical = {}
for canonical, aliases in param_aliases.items():
    for alias in aliases:
        keyword_to_canonical[alias.lower()] = canonical

NUMERIC_PARAMETERS = {
    "BOD", "COD", "氨氮", "总氮", "总磷", "pH", "水温(气温)", "水力负荷", "水力停留时间",
    "进水量", "长", "宽", "高", "种植密度", "覆盖度", "温度", "DO", "溶解氧",
}
SUPPORTED_UNITS = (
    "m³/(m²·d)", "m3/(m2·d)", "kg/m³", "kg/m3", "m³/d", "m3/d", "m³/h", "m3/h",
    "mg/L", "µg/L", "ug/L", "g/m³", "g/m3", "L/day", "L/d", "days", "day", "ppm", "ppb",
    "hr", "h", "d", "m", "°C", "℃", "%",
)
unit_pattern = "(?:{})".format("|".join(re.escape(unit) for unit in SUPPORTED_UNITS))
value_unit_pattern = re.compile(
    rf"(?<![\w.])(?P<value>-?\d+(?:\.\d+)?(?:\s*(?:~|－|～|\s-\s)\s*-?\d+(?:\.\d+)?)?)\s*(?P<unit>{unit_pattern})?(?![\w/])",
    re.IGNORECASE,
)

WATER_ROLE_PATTERNS = {
    "influent": ("进水", "进水端", "influent", "inlet"),
    "effluent": ("出水", "出水端", "effluent", "outlet"),
    "removal": ("去除率", "removal efficiency", "removal rate"),
}

CONCENTRATION_PARAMETERS = {"BOD", "COD", "氨氮", "总氮", "总磷", "DO", "溶解氧"}
EXPECTED_UNIT_GROUPS = {
    "BOD": "concentration", "COD": "concentration", "氨氮": "concentration",
    "总氮": "concentration", "总磷": "concentration", "DO": "concentration", "溶解氧": "concentration",
    "pH": "unitless", "水温(气温)": "temperature", "温度": "temperature",
    "水力停留时间": "time", "进水量": "flow", "水力负荷": "hydraulic_load",
    "长": "length", "宽": "length", "高": "length",
    "种植密度": "density", "覆盖度": "percent",
}
UNIT_GROUPS = {
    "mg/l": "concentration", "ug/l": "concentration", "µg/l": "concentration",
    "g/m³": "concentration", "g/m3": "concentration", "kg/m³": "concentration", "kg/m3": "concentration",
    "ppm": "concentration", "ppb": "concentration", "h": "time", "hr": "time",
    "d": "time", "day": "time", "days": "time", "m³/d": "flow", "m3/d": "flow",
    "l/d": "flow", "l/day": "flow", "m³/h": "flow", "m3/h": "flow",
    "m³/(m²·d)": "hydraulic_load", "m3/(m2·d)": "hydraulic_load",
    "m": "length", "°c": "temperature", "℃": "temperature", "%": "percent",
}

REFERENCE_PATTERNS = (
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(r"\b(?:fig(?:ure)?|table|eq(?:uation)?|reference|ref\.?|page|pp?\.)\s*\d+", re.I),
    re.compile(r"(?:图|表|式|参考文献|第)\s*\d+"),
)


def extract_data_from_pdf(pdf_path):
    """从 PDF 文件中按页提取文本。

    返回值：列表，每项为 {'page': 页码, 'text': 页面文本}
    """
    pages = []
    for i, page_layout in enumerate(extract_pages(pdf_path), start=1):
        text = ""
        # 遍历页面布局元素并累加文本
        for element in page_layout:
            if hasattr(element, "get_text"):
                text += element.get_text()
        pages.append({"page": i, "text": text})
    return pages


def normalize_keyword(keyword):
    return keyword_to_canonical.get(keyword.lower())


def search_keyword_positions(context, allowed_parameters=None):
    context_lower = context.lower()
    matches = []

    for token, canonical in keyword_to_canonical.items():
        if allowed_parameters is not None and canonical not in allowed_parameters:
            continue
        if token.isascii():
            pattern = r"\b" + re.escape(token) + r"\b"
            for m in re.finditer(pattern, context_lower):
                matches.append((canonical, token, m.start()))
        else:
            start = 0
            while True:
                idx = context_lower.find(token, start)
                if idx == -1:
                    break
                matches.append((canonical, token, idx))
                start = idx + len(token)

    return matches


def classify_parameter(context):
    """根据上下文匹配已知关键词并返回所属类别（如水质参数/运行参数）。

    如果没有匹配到已知关键词，返回 '其他参数'。
    """
    matches = search_keyword_positions(context)
    for canonical, _, _ in matches:
        for category, items in category_map.items():
            if canonical in items:
                return category
    return "其他参数"


def category_for_parameter(parameter):
    for category, items in category_map.items():
        if parameter in items:
            return category
    return "其他参数"


def search_role_positions(context):
    matches = []
    lowered = context.lower()
    for role, aliases in WATER_ROLE_PATTERNS.items():
        for alias in aliases:
            for match in re.finditer(re.escape(alias.lower()), lowered):
                matches.append((role, match.start()))
    return matches


def classify_water_role(context, value_pos=None):
    """根据与当前数值的距离判断进水、出水或去除率。"""
    offset = 0
    local_context = context
    if value_pos is not None:
        left_delimiters = [context.rfind(mark, 0, value_pos) for mark in (".", ";", "。", "；", "\n")]
        left = max(left_delimiters) + 1
        right_candidates = [context.find(mark, value_pos) for mark in (".", ";", "。", "；", "\n")]
        right_candidates = [position for position in right_candidates if position >= 0]
        right = min(right_candidates) if right_candidates else len(context)
        local_context = context[left:right]
        offset = left
    matches = search_role_positions(local_context)
    if not matches:
        return "unspecified"
    if value_pos is None:
        return matches[0][0]
    local_value_pos = value_pos - offset
    role, position = min(
        matches,
        key=lambda item: abs(item[1] - local_value_pos) + (12 if item[1] > local_value_pos else 0),
    )
    return role if abs(position - local_value_pos) <= 35 else "unspecified"


def find_nearest_keyword_match(context, value_pos=None):
    matches = search_keyword_positions(context, NUMERIC_PARAMETERS)
    if not matches:
        return None

    if value_pos is None:
        return matches[0]

    canonical, token, position = min(
        matches,
        key=lambda item: abs(item[2] - value_pos) + (12 if item[2] > value_pos else 0),
    )
    return (canonical, token, position) if abs(position - value_pos) <= 50 else None


def find_nearest_keyword(context, value_pos=None):
    """兼容旧接口，仅返回规范参数名。"""
    match = find_nearest_keyword_match(context, value_pos)
    return match[0] if match else ""


def assess_candidate(parameter, value, unit, distance, context):
    """给抽取候选评分，并返回 (置信度, 质量状态, 原因)。"""
    reasons = []
    score = 1.0
    expected_group = EXPECTED_UNIT_GROUPS.get(parameter)
    actual_group = UNIT_GROUPS.get(unit.lower()) if unit else None

    if distance > 30:
        score -= 0.25
        reasons.append("参数关键词距离较远")
    if expected_group == "unitless":
        if unit:
            score -= 0.45
            reasons.append("无量纲参数带有不相容单位")
    elif expected_group and not unit:
        score -= 0.55
        reasons.append("缺少单位")
    elif expected_group and actual_group != expected_group:
        score -= 0.65
        reasons.append("参数与单位不相容")
    elif unit and actual_group is None:
        score -= 0.2
        reasons.append("单位尚未纳入规则")

    local_value = str(value)
    if not unit and any(pattern.search(local_value) for pattern in REFERENCE_PATTERNS):
        score = 0.0
        reasons.append("疑似年份或文献编号")
    if parameter == "pH" and not (0 <= value <= 14):
        score = 0.0
        reasons.append("超出 pH 合理范围")
    if expected_group != "temperature" and value < 0:
        score -= 0.5
        reasons.append("非温度参数出现负值")

    score = max(0.0, min(1.0, score))
    status = "accepted_candidate" if score >= 0.7 else "review_required" if score >= 0.4 else "rejected"
    return score, status, "; ".join(reasons)


def preprocess_value(value_str):
    """将单值或范围转换为浮点数，范围使用中点。"""
    value_str = str(value_str).strip()
    range_match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*(?:~|－|～|\s-\s)\s*(-?\d+(?:\.\d+)?)", value_str)
    if range_match:
        try:
            nums = [float(range_match.group(1)), float(range_match.group(2))]
            return sum(nums) / len(nums)
        except ValueError:
            return None
    try:
        return float(value_str)
    except ValueError:
        return None


UNIT_CONVERSIONS = {
    "ug/l": (0.001, "mg/L"),
    "µg/l": (0.001, "mg/L"),
    "g/m³": (1.0, "mg/L"),
    "g/m3": (1.0, "mg/L"),
    "kg/m³": (1000.0, "mg/L"),
    "kg/m3": (1000.0, "mg/L"),
    "m3/d": (1.0, "m³/d"),
    "m³/h": (24.0, "m³/d"),
    "m3/h": (24.0, "m³/d"),
    "l/d": (0.001, "m³/d"),
    "l/day": (0.001, "m³/d"),
    "hr": (1.0, "h"),
    "d": (24.0, "h"),
    "day": (24.0, "h"),
    "days": (24.0, "h"),
    "ppm": (1.0, "mg/L"),
    "ppb": (0.001, "mg/L"),
    "℃": (1.0, "°C"),
}


def normalize_unit(value, unit):
    """转换常见等价单位并返回 (标准值, 标准单位)。"""
    key = unit.lower() if unit else ""
    factor, canonical = UNIT_CONVERSIONS.get(key, (1.0, unit))
    return value * factor, canonical


def load_existing_training_data(xlsx_path):
    xlsx_path = pt(xlsx_path)
    if not xlsx_path.exists():
        print(f"警告：合并Excel文件未找到：{xlsx_path}")
        return None

    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    sheet_names = [
        "训练数据",
        "training_data",
        "Training Data",
        "Training_Data",
        "Sheet1",
    ]
    for sheet_name in sheet_names:
        if sheet_name in sheets:
            df = sheets[sheet_name]
            break
    else:
        df = next(iter(sheets.values()))

    if "pdf_name" not in df.columns:
        first_col = df.columns[0]
        if first_col.lower() in ("index", "name", "file", "file_name", "pdf"):
            df = df.rename(columns={first_col: "pdf_name"})

    return df


def merge_training_data(existing, new):
    if existing is None or existing.empty:
        return new

    combined = pd.concat([existing, new], ignore_index=True, sort=False)
    if "pdf_name" in combined.columns:
        combined = combined.drop_duplicates(subset=["pdf_name"], keep="last")
    return combined


def build_output_tables(records):
    """由抽取明细构造汇总、训练宽表、待复核表和质量统计。"""
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("未提取到任何数据")
    df = df.drop_duplicates().dropna(subset=["parameter", "value"])
    accepted = df["quality_status"].eq("accepted_candidate")
    ambiguous_water_quality = df["category"].eq("水质参数") & df["water_role"].eq("unspecified")
    training_records = df[accepted & ~ambiguous_water_quality].copy()
    water_quality = training_records["category"].eq("水质参数")
    training_records["training_parameter"] = training_records["parameter"]
    training_records.loc[water_quality, "training_parameter"] = (
        training_records.loc[water_quality, "parameter"] + "__" + training_records.loc[water_quality, "water_role"]
    )
    summary = (
        df.groupby(["pdf_name", "quality_status", "category", "parameter", "unit"], dropna=False).size()
        .reset_index(name="出现次数")
    )
    training_data = training_records.pivot_table(
        index="pdf_name", columns="training_parameter", values="value", aggfunc="mean"
    ).reset_index()
    review_mask = df["quality_status"].ne("accepted_candidate") | ambiguous_water_quality
    review_data = df[review_mask].copy()
    quality_report = (
        df.groupby(["quality_status", "quality_reason"], dropna=False).size()
        .reset_index(name="记录数")
        .sort_values(["quality_status", "记录数"], ascending=[True, False])
    )
    return df, summary, training_data, review_data, quality_report


def extract_parameters_form_text(pdf, pages):
    """在页面文本中查找所有数值+单位并关联到最近的参数关键词，返回记录列表。

    记录格式：{pdf_name, page, category, parameter, value, unit, context}
    """
    records = []
    for page in pages:
        page_no = page["page"]
        text = page["text"]

        # 在全文中查找数值+单位模式
        for match in value_unit_pattern.finditer(text):
            value = match.group("value")
            unit = match.group("unit") or ""

            # 取数值附近的上下文窗口以便定位参数名
            start = max(match.start() - 40, 0)
            end = min(match.end() + 40, len(text))
            context = text[start:end].replace("\n", " ")
            value_pos_in_context = match.start() - start

            # 找到最接近的参数名并进行归一化
            keyword_match = find_nearest_keyword_match(context, value_pos_in_context)
            if not keyword_match:
                # 未能匹配到已知参数，跳过
                continue
            param_name, matched_alias, keyword_pos = keyword_match

            category = category_for_parameter(param_name)
            processed_value = preprocess_value(value)
            if processed_value is None:
                # 数值解析失败，跳过
                continue

            confidence, quality_status, quality_reason = assess_candidate(
                param_name, processed_value, unit, abs(keyword_pos - value_pos_in_context), context
            )
            processed_value, normalized_unit = normalize_unit(processed_value, unit)
            water_role = classify_water_role(context, keyword_pos) if category == "水质参数" else "not_applicable"
            records.append({
                "record_id": "{}:p{}:{}".format(pdf, page_no, len(records) + 1),
                "pdf_name": pdf,
                "page": page_no,
                "water_role": water_role,
                "review_status": "pending",
                "quality_status": quality_status,
                "confidence": confidence,
                "quality_reason": quality_reason,
                "category": category,
                "parameter": param_name,
                "matched_alias": matched_alias,
                "value": processed_value,
                "unit": normalized_unit,
                "raw_value": value,
                "raw_unit": unit,
                "context": context,
            })
    return records


def process_pdf(input_folder, output_excel, output_csv, merge_xlsx=None):
    input_folder = pt(input_folder)
    if not input_folder.exists():
        print(f"输入文件夹不存在：{input_folder}")
        raise FileNotFoundError("输入文件夹不存在：{}".format(input_folder))

    pdf_files = sorted(input_folder.glob("*.pdf")) + sorted(input_folder.glob("*.PDF"))
    if not pdf_files:
        print(f"未找到PDF文件：{input_folder}")
        raise FileNotFoundError("未找到PDF文件：{}".format(input_folder))

    all_records = []
    for pdf_file in pdf_files:
        print(f"正在处理：{pdf_file.name}")
        pages = extract_data_from_pdf(pdf_file)
        records = extract_parameters_form_text(pdf_file.name, pages)
        all_records.extend(records)

    try:
        df, summary, training_data, review_data, quality_report = build_output_tables(all_records)
    except ValueError:
        print("未提取到任何数据。")
        raise

    if merge_xlsx:
        existing = load_existing_training_data(merge_xlsx)
        if existing is not None:
            training_data = merge_training_data(existing, training_data)
            print(f"已合并现有训练数据：{merge_xlsx}")

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="提取参数明细", index=False)
        summary.to_excel(writer, sheet_name="数据汇总", index=False)
        training_data.to_excel(writer, sheet_name="训练数据", index=False)
        review_data.to_excel(writer, sheet_name="待人工复核", index=False)
        quality_report.to_excel(writer, sheet_name="质量统计", index=False)

    training_data.to_csv(output_csv, index=False)
    print(f"数据已成功保存到：{output_excel} 和 {output_csv}")
    return {
        "detail_rows": len(df),
        "accepted_rows": len(df) - len(review_data),
        "review_rows": len(review_data),
        "training_rows": len(training_data),
        "excel_path": str(output_excel),
        "csv_path": str(output_csv),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从论文PDF提取参数并生成训练数据，可选合并已有处理好的Excel。")
    parser.add_argument("-i", "--input-folder", default="./pdf", help="PDF输入文件夹")
    parser.add_argument("-o", "--output-excel", default="extracted_parameters.xlsx", help="输出Excel文件")
    parser.add_argument("-c", "--output-csv", default="extracted_parameters.csv", help="输出CSV文件")
    parser.add_argument("-m", "--merge-xlsx", default=None, help="如果需要合并已有处理好的Excel，请传入路径")
    args = parser.parse_args()

    process_pdf(args.input_folder, args.output_excel, args.output_csv, args.merge_xlsx)
