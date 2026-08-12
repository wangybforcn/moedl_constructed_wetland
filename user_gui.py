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
from tkinter import filedialog, messagebox, ttk

import data_to_excel
import parameter_io
import design_3d
from parameter_schema import DESIGN_FIELDS, PARAMETER_SPECS
from generator_3d import find_openscad
from version import __version__

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

class App:
    def __init__(self, root):
        self.root = root
        root.title("人工湿地辅助设计工具 v{}".format(__version__))
        root.minsize(580, 260)

        self.model_path = None
        self.data_folder = None
        self.output_folder = str(Path.cwd() / "output")
        self.openscad_path = find_openscad() or "openscad"
        self.busy = False
        self.closing = False
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        frm = tk.Frame(root)
        frm.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        tk.Button(frm, text="打开 PDF 文件夹", command=self.select_pdf_folder).grid(row=0, column=0, sticky="ew")
        tk.Button(frm, text="提取并生成 CSV", command=self.extract_pdf).grid(row=0, column=1, sticky="ew")
        tk.Button(frm, text="训练 Imputer", command=self.train_imputer).grid(row=1, column=0, sticky="ew")
        tk.Button(frm, text="加载模型", command=self.load_model).grid(row=1, column=1, sticky="ew")
        tk.Button(frm, text="输入参数并生成", command=self.open_generate_window).grid(row=2, column=0, sticky="ew")
        tk.Button(frm, text="选择输出目录", command=self.select_output_folder).grid(row=2, column=1, sticky="ew")
        tk.Button(frm, text="选择 OpenSCAD", command=self.select_openscad).grid(row=3, column=0, sticky="ew")
        tk.Button(frm, text="打开输出目录", command=self.open_output_folder).grid(row=3, column=1, sticky="ew")
        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        self.status = tk.StringVar()
        self.status.set("就绪")
        tk.Label(root, textvariable=self.status).pack(fill=tk.X, padx=8, pady=6)
        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=8, pady=(0, 8))

    def select_pdf_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.data_folder = folder
            self.status.set(f"已选择 PDF 文件夹: {folder}")

    def select_output_folder(self):
        folder = filedialog.askdirectory(initialdir=self.output_folder)
        if folder:
            self.output_folder = folder
            self.status.set("输出目录: {}".format(folder))

    def select_openscad(self):
        path = filedialog.askopenfilename(
            title="选择 OpenSCAD 可执行文件",
            filetypes=[("OpenSCAD", "openscad.exe"), ("可执行文件", "*.exe"), ("所有文件", "*")],
        )
        if path:
            self.openscad_path = path
            self.status.set("OpenSCAD: {}".format(path))

    def open_output_folder(self):
        folder = Path(self.output_folder)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))
        except (AttributeError, OSError) as exc:
            messagebox.showerror("无法打开目录", str(exc))

    def _set_busy(self, busy, status=None):
        self.busy = busy
        if status:
            self.status.set(status)
        if busy:
            self.progress.start(10)
        else:
            self.progress.stop()

    def _run_background(self, status, task, on_success):
        if self.busy:
            messagebox.showwarning("任务进行中", "请等待当前任务完成")
            return
        self._set_busy(True, status)

        def worker():
            try:
                result = task()
                if not self.closing:
                    self.root.after(0, lambda: self._background_success(result, on_success))
            except Exception as exc:
                if not self.closing:
                    self.root.after(0, lambda error=str(exc): self._background_failure(error))

        threading.Thread(target=worker, daemon=True).start()

    def _background_success(self, result, callback):
        self._set_busy(False)
        callback(result)

    def _background_failure(self, error):
        self._set_busy(False, "任务失败")
        messagebox.showerror("任务失败", error)

    def on_close(self):
        self.closing = True
        self.progress.stop()
        self.root.destroy()

    def extract_pdf(self):
        if not self.data_folder:
            messagebox.showwarning("未选择", "请先选择 PDF 文件夹")
            return
        out_xlsx = os.path.join(self.data_folder, "extracted_parameters.xlsx")
        out_csv = os.path.join(self.data_folder, "extracted_parameters.csv")
        self._run_background(
            "正在提取 PDF...",
            lambda: data_to_excel.process_pdf(self.data_folder, out_xlsx, out_csv),
            lambda _: self.status.set(f"提取完成，保存: {out_xlsx}, {out_csv}"),
        )

    def train_imputer(self):
        if not self.data_folder:
            messagebox.showwarning("未选择", "请先选择包含 extracted_parameters.csv 的文件夹")
            return
        csv_path = os.path.join(self.data_folder, "extracted_parameters.csv")
        if not os.path.exists(csv_path):
            messagebox.showwarning("未找到", f"未找到 {csv_path}，请先提取 PDF")
            return
        out_model = os.path.join(self.data_folder, "trained_imputer.joblib")
        def task():
            import api
            return api.train_imputer(csv_path, out_model, min_non_na_ratio=0.2, n_estimators=100, max_iter=10)

        def success(_):
            self.model_path = out_model
            self.status.set(f"训练完成: {out_model}")

        self._run_background("开始训练 Imputer...", task, success)

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Joblib", "*.joblib"), ("All", "*")])
        if path:
            self.model_path = path
            self.status.set(f"已加载模型: {path}")

    def open_generate_window(self):
        model_features = []
        if self.model_path:
            try:
                model_features = parameter_io.get_feature_names_from_model(self.model_path) or []
            except Exception as exc:
                messagebox.showerror("模型加载失败", str(exc))
                return
        optional_fields = ("进水量", "水力停留时间", "水力负荷")
        feature_names = list(dict.fromkeys(list(DESIGN_FIELDS) + list(model_features) + list(optional_fields)))
        
        win = tk.Toplevel(self.root)
        win.title("生成参数输入")
        model_note = "已加载模型，可留空数值字段" if self.model_path else "未加载模型，请填写长、宽、高；结构字段已有默认值"
        tk.Label(win, text=model_note).pack(anchor="w", padx=8, pady=4)
        
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
            default = PARAMETER_SPECS.get(feat).default if feat in PARAMETER_SPECS else None
            if default is not None:
                ent.insert(0, str(default))
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

        output_prefix = os.path.join(self.output_folder, f"wetland_{mode}")

        def task():
            import api
            return api.generate_design(
                params,
                model_path=self.model_path,
                output_prefix=output_prefix,
                mode=mode,
                openscad_path=self.openscad_path,
            )

        def success(res):
            result_text = json.dumps(res, ensure_ascii=False, indent=2)
            messagebox.showinfo(f"生成完成 ({mode})", result_text)
            self.status.set(f"生成完成 ({mode})")

        self._run_background("正在生成 {}...".format(mode.upper()), task, success)

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
