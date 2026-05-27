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

unit_pattern = r"(?:mg/L|µg/L|ug/L|g/m³|g/m3|kg/m³|kg/m3|m³/d|m3/d|m³/(m²·d)|m3/(m2·d)|m³/h|m3/h|L/d|L/day|h|hr|d|m|°C|℃|%|ppm|ppb)"
value_unit_pattern = re.compile(
    rf"(?P<value>\d+\.?\d*(?:\s*(?:~|-|－|～)\s*\d+\.?\d*)?)\s*(?P<unit>{unit_pattern})?",
    re.IGNORECASE,
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


def search_keyword_positions(context):
    context_lower = context.lower()
    matches = []

    for token, canonical in keyword_to_canonical.items():
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


def find_nearest_keyword(context, value_pos=None):
    matches = search_keyword_positions(context)
    if not matches:
        return ""

    if value_pos is None:
        return matches[0][0]

    left = [(canonical, idx) for canonical, _, idx in matches if idx <= value_pos]
    right = [(canonical, idx) for canonical, _, idx in matches if idx > value_pos]

    if left:
        left.sort(key=lambda x: value_pos - x[1])
        return left[0][0]

    if right:
        right.sort(key=lambda x: x[1] - value_pos)
        if right[0][1] - value_pos <= 30:
            return right[0][0]

    return matches[0][0]


def preprocess_value(value_str):
    """把提取到的数值字符串转换为浮点数。

    如果是范围（例如 1~3），返回平均值；否则返回浮点数。失败返回 None。
    """
    if not isinstance(value_str, str):
        value_str = str(value_str)

    if any(separator in value_str for separator in ['~', '-', '－', '～']):
        parts = re.split(r'[~－～-]', value_str)
        try:
            nums = [float(p.strip()) for p in parts if p.strip()]
            return sum(nums) / len(nums)
        except ValueError:
            return None
    try:
        return float(value_str)
    except ValueError:
        return None


def preprocess_value(value_str):
    if any(separator in value_str for separator in ['~', '-', '－', '～']):
        parts = re.split(r'[~－～-]', value_str)
        try:
            nums = [float(p.strip()) for p in parts if p.strip()]
            return sum(nums) / len(nums)
        except ValueError:
            return None
    try:
        return float(value_str)
    except ValueError:
        return None


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
            param_name = find_nearest_keyword(context, value_pos_in_context)
            if not param_name:
                # 未能匹配到已知参数，跳过
                continue

            category = classify_parameter(context)
            processed_value = preprocess_value(value)
            if processed_value is None:
                # 数值解析失败，跳过
                continue

            records.append({
                "pdf_name": pdf,
                "page": page_no,
                "category": category,
                "parameter": param_name,
                "value": processed_value,
                "unit": unit,
                "context": context,
            })
    return records
    for page in pages:
        page_no = page["page"]
        text = page["text"]

        for match in value_unit_pattern.finditer(text):
            value = match.group("value")
            unit = match.group("unit") or ""
            start = max(match.start() - 40, 0)
            end = min(match.end() + 40, len(text))
            context = text[start:end].replace("\n", " ")
            value_pos_in_context = match.start() - start

            param_name = find_nearest_keyword(context, value_pos_in_context)
            if not param_name:
                continue

            category = classify_parameter(context)
            processed_value = preprocess_value(value)
            if processed_value is None:
                continue

            records.append({
                "pdf_name": pdf,
                "page": page_no,
                "category": category,
                "parameter": param_name,
                "value": processed_value,
                "unit": unit,
                "context": context,
            })
    return records


def process_pdf(input_folder, output_excel, output_csv, merge_xlsx=None):
    input_folder = pt(input_folder)
    if not input_folder.exists():
        print(f"输入文件夹不存在：{input_folder}")
        return

    pdf_files = sorted(input_folder.glob("*.pdf")) + sorted(input_folder.glob("*.PDF"))
    if not pdf_files:
        print(f"未找到PDF文件：{input_folder}")
        return

    all_records = []
    for pdf_file in pdf_files:
        print(f"正在处理：{pdf_file.name}")
        pages = extract_data_from_pdf(pdf_file)
        records = extract_parameters_form_text(pdf_file.name, pages)
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    if df.empty:
        print("未提取到任何数据。")
        return

    df = df.drop_duplicates()
    df = df.dropna(subset=["parameter", "value"])

    summary = (
        df.groupby(["pdf_name", "category", "parameter", "unit"]).size()
        .reset_index(name="出现次数")
    )

    training_data = (
        df.pivot_table(
            index="pdf_name",
            columns="parameter",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    if merge_xlsx:
        existing = load_existing_training_data(merge_xlsx)
        if existing is not None:
            training_data = merge_training_data(existing, training_data)
            print(f"已合并现有训练数据：{merge_xlsx}")

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="提取参数明细", index=False)
        summary.to_excel(writer, sheet_name="数据汇总", index=False)
        training_data.to_excel(writer, sheet_name="训练数据", index=False)

    training_data.to_csv(output_csv, index=False)
    print(f"数据已成功保存到：{output_excel} 和 {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从论文PDF提取参数并生成训练数据，可选合并已有处理好的Excel。")
    parser.add_argument("-i", "--input-folder", default="./pdf", help="PDF输入文件夹")
    parser.add_argument("-o", "--output-excel", default="extracted_parameters.xlsx", help="输出Excel文件")
    parser.add_argument("-c", "--output-csv", default="extracted_parameters.csv", help="输出CSV文件")
    parser.add_argument("-m", "--merge-xlsx", default=None, help="如果需要合并已有处理好的Excel，请传入路径")
    args = parser.parse_args()

    process_pdf(args.input_folder, args.output_excel, args.output_csv, args.merge_xlsx)
