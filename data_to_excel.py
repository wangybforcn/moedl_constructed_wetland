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
        "芦苇", "美人蕉", "香蒲", "菖蒲", "再力花", "种植密度", "覆盖度"
    ],
    "环境信息": [
        "温度", "季节", "地区", "气候", "光照", "DO", "溶解氧", "pH", "水温(气温)"
    ]
}

# 数值参数：可以被量化的参数，参与模型训练
numeric_params = [
    "BOD", "COD", "氨氮", "总氮", "总磷", "pH", "水温(气温)",
    "水力负荷", "水力停留时间", "进水量",
    "温度", "DO", "溶解氧",
    "宽", "高", "长",
    "种植密度", "覆盖度", "CNP"
]

# 参数合理范围（用于过滤异常提取值）
param_ranges = {
    "BOD": (0, 1000),
    "COD": (0, 2000),
    "氨氮": (0, 500),
    "总氮": (0, 1000),
    "总磷": (0, 100),
    "pH": (0, 14),
    "水温(气温)": (0, 50),
    "温度": (0, 50),
    "水力负荷": (0, 10),
    "水力停留时间": (0, 30),
    "进水量": (0, 100000),
    "DO": (0, 20),
    "溶解氧": (0, 20),
    "宽": (0, 1000),
    "高": (0, 10),
    "长": (0, 10000),
    "种植密度": (0, 100),
    "覆盖度": (0, 100),
    "CNP": (0, 100),
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

def is_valid_param_value(param_name, value):
    """检查参数值是否在合理范围内。"""
    if param_name not in param_ranges or value is None:
        return True
    min_v, max_v = param_ranges[param_name]
    return min_v <= value <= max_v

def find_nearest_keyword(context, value_pos=None):
    """在上下文中查找与数值最关联的参数名。
    策略：
    1. 只在 numeric_params 中查找（过滤分类/文本参数）
    2. 英文参数使用 \b 整词匹配，避免匹配到其他单词中
    3. 中文单字参数要求前面不是字母/数字/汉字，排除"时长""最高"等
    4. 优先查找数值前方（左侧）的参数名，前方无匹配再看后方（<15字符）
    """
    found = []
    context_lower = context.lower()

    for kw in numeric_params:
        kw_lower = kw.lower()

        if kw.isascii():
            # 英文参数（BOD, COD, DO, pH等）：使用整词匹配
            pattern = r'\b' + re.escape(kw_lower) + r'\b'
            for m in re.finditer(pattern, context_lower):
                found.append((kw, m.start()))
        elif len(kw) == 1 and '\u4e00' <= kw <= '\u9fa5':
            # 中文单字参数（长、宽、高）：要求前面不是字母/数字/汉字
            # 避免"时长""最高""生长""加高"等误匹配
            pattern = r'(?<![a-zA-Z0-9\u4e00-\u9fa5])' + re.escape(kw_lower)
            for m in re.finditer(pattern, context_lower):
                found.append((kw, m.start()))
        else:
            # 中文多字参数：普通子串匹配即可（不容易误匹配）
            search_from = 0
            while True:
                idx = context_lower.find(kw_lower, search_from)
                if idx == -1:
                    break
                found.append((kw, idx))
                search_from = idx + 1

    if not found:
        return ""

    if value_pos is not None:
        # 分离前方和后方匹配
        left_matches = [(kw, idx) for kw, idx in found if idx < value_pos]
        right_matches = [(kw, idx) for kw, idx in found if idx > value_pos]

        if left_matches:
            # 优先取数值前方最近的参数名
            left_matches.sort(key=lambda x: value_pos - x[1])
            return left_matches[0][0]
        elif right_matches:
            # 前方无匹配时，取后方最近的，但必须在很近范围内
            right_matches.sort(key=lambda x: x[1] - value_pos)
            dist = right_matches[0][1] - value_pos
            if dist < 15:  # 后方参数名必须在15字符以内
                return right_matches[0][0]
            return ""
        return ""
    else:
        return found[0][0]

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

            # 收紧上下文窗口，只取数值附近25+25字符
            start = max(match.start() - 25, 0)
            end = min(match.end() + 25, len(text))
            context = text[start:end].replace("\n", " ")

            # 计算数值在context中的位置
            value_pos_in_context = match.start() - start

            category = classify_parameter(context)
            param_name = find_nearest_keyword(context, value_pos_in_context)

            # 过滤：没有匹配到数值参数 或 参数名为空
            if not param_name:
                continue

            processed_value = preprocess_value(value)

            # 过滤：数值超出合理范围
            if not is_valid_param_value(param_name, processed_value):
                continue

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

    #保留缺失值（NaN），供IterativeImputer学习插补
    # training_data = training_data.fillna(0)  # 旧逻辑：用0填充会破坏IterativeImputer的学习前提

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="提取参数明细", index=False)
        summary.to_excel(writer, sheet_name="数据汇总", index=False)
        training_data.to_excel(writer, sheet_name="训练数据", index=False)

    #保存CSV
    training_data.to_csv(output_csv, index=False)

    print(f"数据已成功保存到：{output_excel} 和 {output_csv}")


#main
if __name__ == "__main__":
    input_folder = "./model_pdf"
    output_excel = "extracted_parameters.xlsx"
    output_csv = "extracted_parameters.csv"
    process_pdf(input_folder, output_excel, output_csv)