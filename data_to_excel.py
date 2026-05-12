#将pdf提取并保存到excel中， 仍需修正，提取内容过于冗杂

# -*- coding: utf-8 -*-

import re
from pdfminer.high_level import extract_pages
import pandas as pd
from pathlib import Path as pt


#定义核心参数列表，单位，数值范围
params_ ={
    "水质参数": [
        "CNP", "COD", "BOD", "氨氮", "总氮", "总磷", "pH", "水温(气温)"
    ],
    "运行参数": [
        "水力负荷", "水力停留时间", "进水量", "填料类型(结构)"
    ],
    "湿地(类型)结构": [
        "长", "宽", "高", "垂直", "水平潜流", "分层结构", "湿地类型"
    ],
    "工艺模式": [

    ],
    "植物参数": [

    ],
    "环境信息": [

    ]
}

unit_pattern = r"(mg/L|µg/L|g/m³|kg/m³|m³/d|m³/(m²·d)|m³/h|L/d|h|d|m|°C|%)"
value_unit_pattern = re.compile(
    rf"(?P<value>\d+\.?\d*\s*(?:~|-|－|～)\s*\d+\.?\d*|\d+\.?\d*)\s*(?P<unit>{unit_pattern})"
)

#提取中的数据
def extract_data_from_pdf(pdf_path):
    pages = []
    for i, page_layout in enumerate(extract_pages(pdf_path), start=1):
        text = ""
        for element in page_layout:
            if hasattr(element, "get_text"):
                text += element.get_text()
        pages.append({"page": i, "text": text})

    return pages

#根据上下文判断参数类别
def classify_parameter(context):
    for category, keywords in params_.items():
        for kw in keywords:
            if kw.lower() in context.lower():
                return category
    return "其他参数"

#找出上下文中最接近的参数名
def find_nearest_keyword(context):
    found = []
    for keyword in params_.values():
        for kw in keyword:
            if kw.lower() in context.lower():
                found.append(kw)
    return "、".join(sorted(set(found))) if found else ""

#预处理数值：如果范围，取平均值
def preprocess_value(value_str):
    if '~' in value_str or '-' in value_str or '－' in value_str or '～' in value_str:
        parts = re.split(r'[~－～-]', value_str)
        try:
            nums = [float(p.strip()) for p in parts if p.strip()]
            return sum(nums) / len(nums)
        except ValueError:
            return None
    else:
        try:
            return float(value_str)
        except ValueError:
            return None

#从文本中提取参数
def extract_parameters_form_text(pdf, pages):
    records = []

    for page in pages:
        page_no = page["page"]
        text = page["text"]
    
        for match in value_unit_pattern.finditer(text):
            value = match.group("value")
            unit = match.group("unit")

            start = max(match.start() - 60, 0)
            end = min(match.end() + 80, len(text))
            context = text[start:end].replace("\n", " ")

            category = classify_parameter(context)
            param_name = find_nearest_keyword(context)

            if category == "其他参数" and not param_name:
                continue

            processed_value = preprocess_value(value)

            records.append({
                "pdf_name": pdf,
                "page": page_no,
                "category": category,
                "parameter": param_name,
                "value": processed_value,
                "unit": unit,
                "context": context
            })

    return records

#将信息保存到excel和csv
def process_pdf(input_folder, output_excel, output_csv):
    input_folder = pt(input_folder)
    all_records = []

    pdf_files = list(input_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"未找到PDF文件：{input_folder}")
        return
    
    for pdf_file in pdf_files:
        print(f"正在处理：{pdf_file.name}")
        pages = extract_data_from_pdf(pdf_file)
        records = extract_parameters_form_text(pdf_file.name, pages)
        all_records.extend(records)

    df = pd.DataFrame(all_records)

    if df.empty:
        print("未提取到任何数据。")
        return
    
    #去重
    df = df.drop_duplicates()

    #处理缺失值：删除有缺失的行
    df = df.dropna()

    summary = (
        df.groupby(["pdf_name", "category", "parameter", "unit"]).size()
        .reset_index(name="出现次数")
    )

    #为模型训练准备数据：将参数作为特征，按pdf分组
    #每个pdf一行，参数作为列
    training_data = df.pivot_table(
        index="pdf_name",
        columns="parameter",
        values="value",
        aggfunc="mean"  # 如果有多个值，取平均
    ).reset_index()

    #填充缺失值
    training_data = training_data.fillna(0)

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="提取参数明细", index=False)
        summary.to_excel(writer, sheet_name="数据汇总", index=False)
        training_data.to_excel(writer, sheet_name="训练数据", index=False)

    #保存CSV
    training_data.to_csv(output_csv, index=False)

    print(f"数据已成功保存到：{output_excel} 和 {output_csv}")


#main
if __name__ == "__main__":
    input_folder = "./pdf"
    output_excel = "extracted_parameters.xlsx"
    output_csv = "extracted_parameters.csv"
    process_pdf(input_folder, output_excel, output_csv)