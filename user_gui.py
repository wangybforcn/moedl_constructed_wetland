# -*- coding: utf-8 -*-
"""
简单 GUI：整合 PDF 提取、模型训练与 2D/3D 生成。
功能：
- 打开 PDF 文件夹并提取参数（调用 data_to_excel.process_pdf）
- 训练 Imputer（调用 api.train_imputer）
- 加载训练好的模型
- 输入已知参数并生成 2D/3D 输出（调用 3d.generate_from_params）
"""

import os
import json
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import data_to_excel
import importlib
import parameter_io

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Note: import of local 3d module may be named '3d' which is not a valid identifier;
# we will import by module name using importlib
try:
    design3d = importlib.import_module('3d')
except Exception:
    design3d = None

class App:
    def __init__(self, root):
        self.root = root
        root.title("Wetland Designer GUI")

        self.model_path = None
        self.data_folder = None

        frm = tk.Frame(root)
        frm.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        tk.Button(frm, text="打开 PDF 文件夹", command=self.select_pdf_folder).grid(row=0, column=0, sticky="ew")
        tk.Button(frm, text="提取并生成 CSV", command=self.extract_pdf).grid(row=0, column=1, sticky="ew")
        tk.Button(frm, text="训练 Imputer", command=self.train_imputer).grid(row=1, column=0, sticky="ew")
        tk.Button(frm, text="加载模型", command=self.load_model).grid(row=1, column=1, sticky="ew")
        tk.Button(frm, text="输入参数并生成", command=self.open_generate_window).grid(row=2, column=0, columnspan=2, sticky="ew")

        self.status = tk.StringVar()
        self.status.set("就绪")
        tk.Label(root, textvariable=self.status).pack(fill=tk.X, padx=8, pady=6)

    def select_pdf_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.data_folder = folder
            self.status.set(f"已选择 PDF 文件夹: {folder}")

    def extract_pdf(self):
        if not self.data_folder:
            messagebox.showwarning("未选择", "请先选择 PDF 文件夹")
            return
        out_xlsx = os.path.join(self.data_folder, "extracted_parameters.xlsx")
        out_csv = os.path.join(self.data_folder, "extracted_parameters.csv")
        self.status.set("正在提取 PDF...")

        def worker():
            try:
                data_to_excel.process_pdf(self.data_folder, out_xlsx, out_csv)
                self.status.set(f"提取完成，保存: {out_xlsx}, {out_csv}")
            except Exception as e:
                messagebox.showerror("错误", str(e))
                self.status.set("提取失败")

        threading.Thread(target=worker, daemon=True).start()

    def train_imputer(self):
        if not self.data_folder:
            messagebox.showwarning("未选择", "请先选择包含 extracted_parameters.csv 的文件夹")
            return
        csv_path = os.path.join(self.data_folder, "extracted_parameters.csv")
        if not os.path.exists(csv_path):
            messagebox.showwarning("未找到", f"未找到 {csv_path}，请先提取 PDF")
            return
        out_model = os.path.join(self.data_folder, "trained_imputer.joblib")
        self.status.set("开始训练 imputer（后台）...")

        def worker():
            try:
                import api
                model_info = api.train_imputer(csv_path, out_model, min_non_na_ratio=0.2, n_estimators=100, max_iter=10)
                self.model_path = out_model
                self.status.set(f"训练完成: {out_model}")
            except Exception as e:
                messagebox.showerror("训练失败", str(e))
                self.status.set("训练失败")

        threading.Thread(target=worker, daemon=True).start()

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Joblib", "*.joblib"), ("All", "*")])
        if path:
            self.model_path = path
            self.status.set(f"已加载模型: {path}")

    def open_generate_window(self):
        if not self.model_path:
            messagebox.showwarning("未加载模型", "请先训练或加载模型")
            return
        if design3d is None:
            messagebox.showerror("模块错误", "无法导入 3d 模块")
            return

        try:
            feature_names = parameter_io.get_feature_names_from_model(self.model_path)
            if not feature_names:
                messagebox.showerror(
                    "模型错误",
                    "无法从模型中读取特征列表，请确保模型以字典方式保存包含 'feature_names' 字段",
                )
                return
        except Exception as e:
            messagebox.showerror("模型加载失败", str(e))
            return
        
        win = tk.Toplevel(self.root)
        win.title("生成参数输入")
        tk.Label(win, text="输入参数值（缺失则保留为空）：").pack(anchor="w", padx=8, pady=4)
        
        # 创建滚动框架
        canvas = tk.Canvas(win)
        scrollbar = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        entries = {}
        for feat in feature_names:
            frm = tk.Frame(scrollable_frame)
            frm.pack(anchor="w", padx=8, pady=2, fill="x")
            display_name = parameter_io.get_feature_display_name(feat)
            tk.Label(frm, text=display_name, width=24, anchor="w").pack(side="left")
            ent = tk.Entry(frm, width=20)
            ent.pack(side="left")
            entries[feat] = ent
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        btn_frame = tk.Frame(win)
        btn_frame.pack(fill="x", padx=8, pady=8)
        
        def do_generate_all():
            self._do_generate(entries, feature_names, "all")
        
        def do_generate_2d():
            self._do_generate(entries, feature_names, "2d")
        
        def do_generate_3d():
            self._do_generate(entries, feature_names, "3d")

        def do_compute_parameters():
            self._compute_parameters(entries, feature_names)
        
        tk.Button(btn_frame, text="生成全部 (2D+3D)", command=do_generate_all).pack(side="left", padx=4)
        tk.Button(btn_frame, text="生成 2D", command=do_generate_2d).pack(side="left", padx=4)
        tk.Button(btn_frame, text="生成 3D", command=do_generate_3d).pack(side="left", padx=4)
        tk.Button(btn_frame, text="仅计算参数", command=do_compute_parameters).pack(side="left", padx=4)
    
    def _do_generate(self, entries, feature_names, mode):
        """从输入框获取参数并调用生成函数。"""
        raw_inputs = {k: e.get() for k, e in entries.items()}
        params = parameter_io.normalize_parameters(raw_inputs)
        params = parameter_io.ensure_all_features(params, feature_names)

        try:
            output_prefix = os.path.join(self.data_folder or ".", f"output_{mode}")
            import api
            res = api.generate_design(
                params,
                model_path=self.model_path,
                output_prefix=output_prefix,
                mode=mode,
            )
            result_text = json.dumps(res, ensure_ascii=False, indent=2)
            messagebox.showinfo(f"生成完成 ({mode})", result_text)
            self.status.set(f"生成完成 ({mode})")
        except Exception as e:
            messagebox.showerror("生成失败", str(e))
            self.status.set("生成失败")

    def _compute_parameters(self, entries, feature_names):
        """仅调用模型计算缺失参数并显示结果。"""
        raw_inputs = {k: e.get() for k, e in entries.items()}
        params = parameter_io.normalize_parameters(raw_inputs)
        params = parameter_io.ensure_all_features(params, feature_names)

        try:
            import api
            computed = api.compute_parameters(params, self.model_path)
            result_text = json.dumps(computed, ensure_ascii=False, indent=2)
            messagebox.showinfo("计算完成", result_text)
            self.status.set("参数计算完成")
        except Exception as e:
            messagebox.showerror("计算失败", str(e))
            self.status.set("计算失败")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
