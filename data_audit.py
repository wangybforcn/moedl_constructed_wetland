# -*- coding: utf-8 -*-
"""训练数据质量审计与 JSON 报告导出。"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


SOURCE_COLUMNS = ("study_id", "source_id", "pdf_name")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_source_column(frame: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    if preferred:
        if preferred not in frame.columns:
            raise ValueError("来源分组列不存在: {}".format(preferred))
        return preferred
    return next((name for name in SOURCE_COLUMNS if name in frame.columns), None)


def audit_dataframe(frame: pd.DataFrame, source_column: Optional[str] = None) -> Dict[str, Any]:
    numeric = frame.select_dtypes(include=[np.number])
    duplicate_rows = int(frame.duplicated().sum())
    constant_columns = [name for name in numeric.columns if numeric[name].nunique(dropna=True) <= 1]
    all_missing_columns = [name for name in frame.columns if frame[name].isna().all()]
    source_column = find_source_column(frame, source_column)
    source_count = int(frame[source_column].nunique(dropna=True)) if source_column else 0
    warnings = []
    if len(frame) < 30:
        warnings.append("样本少于 30 行，模型指标波动可能很大")
    if source_column is None:
        warnings.append("缺少 study_id/source_id/pdf_name，无法执行来源分组验证")
    elif source_count < 3:
        warnings.append("独立来源少于 3 个，无法进行可靠的分组交叉验证")
    if duplicate_rows:
        warnings.append("存在 {} 行完全重复记录".format(duplicate_rows))
    if constant_columns:
        warnings.append("存在常量数值列: {}".format(", ".join(constant_columns)))
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "numeric_columns": int(len(numeric.columns)),
        "source_column": source_column,
        "source_count": source_count,
        "duplicate_rows": duplicate_rows,
        "all_missing_columns": all_missing_columns,
        "constant_numeric_columns": constant_columns,
        "missing_ratio": {name: float(value) for name, value in numeric.isna().mean().items()},
        "warnings": warnings,
    }


def audit_csv(csv_path: str, source_column: Optional[str] = None) -> Dict[str, Any]:
    frame = pd.read_csv(csv_path)
    report = audit_dataframe(frame, source_column)
    report.update({"data_path": str(Path(csv_path).resolve()), "sha256": file_sha256(csv_path)})
    return report


def write_report(report: Dict[str, Any], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def main():
    parser = argparse.ArgumentParser(description="审计人工湿地训练 CSV")
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-column", default=None)
    parser.add_argument("--output", default="data_audit.json")
    args = parser.parse_args()
    report = audit_csv(args.data, args.source_column)
    write_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
