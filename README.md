# 人工湿地数据与辅助设计工具

本项目从人工湿地论文 PDF 中提取带来源上下文的参数，生成 Excel/CSV 数据集，训练数值缺失值填补或监督回归模型，并根据经过校验的参数生成二维示意图和 OpenSCAD 三维模型。

当前版本是研究与设计辅助原型。自动抽取数据必须人工复核，模型输出和几何模型不能替代环境工程计算、结构设计或施工图审查。

## 工作流程

1. `data_to_excel.py` 从 PDF 提取参数，保留论文名、页码、原始值、原始单位和上下文。
2. 常见单位统一后生成参数明细、质量统计、待人工复核表和宽表训练数据。
3. `model_training_saving_v2.py` 训练 IterativeImputer 或回归模型。
4. `parameter_schema.py` 统一字段类型、单位、默认值和物理约束。
5. `3d.py` 补全数值参数并生成 2D PNG、SCAD、STL 和 3D PNG。
6. `user_gui.py` 提供桌面操作入口。

## 环境安装

推荐 Python 3.11，最低 Python 3.9。

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
```

生成 STL 或 3D PNG 还需安装 OpenSCAD，并把 `openscad` 加入 PATH。只生成 2D 图片不需要 OpenSCAD。

程序也会自动检查 Windows 常见 OpenSCAD 安装目录。GUI 可手动选择 `openscad.exe`。每个 STL/PNG 渲染任务会生成相邻的 `.log` 文件，并校验命令退出码和产物是否非空。

## 使用方法

提取 PDF：

```bash
python data_to_excel.py -i ./pdf -o extracted_parameters.xlsx -c extracted_parameters.csv
```

训练缺失值模型：

```bash
python model_training_saving_v2.py --mode imputer --data extracted_parameters.csv --output trained_imputer.joblib
```

如果输入文件包含 `review_status`，正式实验建议增加 `--require-reviewed`，仅使用状态为 `approved` 或 `confirmed` 的数据。

训练监督模型：

```bash
python model_training_saving_v2.py --mode supervised --data extracted_parameters.csv --target COD --n-splits 5 --report reports/cod.json
```

训练前审计数据：

```bash
python data_audit.py --data extracted_parameters.csv --output reports/data_audit.json
```

如果同一论文包含多组实验，请保留 `study_id`、`source_id` 或 `pdf_name`。监督训练会自动按来源使用 GroupKFold，并以完整来源为单位生成独立留出集，避免同一论文同时出现在训练折和验证折。也可使用 `--source-column` 明确指定来源列。

训练报告包括数据 SHA-256、缺失率、来源数量、交叉验证策略、实际折数、按来源留出指标、中位数基线对比、随机种子和关键依赖版本。模型相对基线没有明显改善时，不应据此作工程判断。

启动 GUI：

```bash
python user_gui.py
```

不使用模型，直接从完整参数 JSON 生成 2D 图：

```bash
python design_cli.py --params examples/design_parameters.json --output-prefix output/wetland --mode 2d
```

使用 Imputer 补全空值并生成全部结果：

```bash
python design_cli.py --params parameters.json --model trained_imputer.joblib --output-prefix output/wetland --mode all
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

## Windows 打包

在 PowerShell 中执行：

```powershell
.\build_windows.ps1
```

脚本会安装 PyInstaller、先运行测试，再生成 `dist\WetlandDesigner.exe`。OpenSCAD 不会嵌入程序包，使用 3D 功能的电脑仍需单独安装 OpenSCAD；2D、PDF 提取和模型训练不受影响。

构建脚本会自动创建并使用项目内的 `.venv`，避免调用系统中可能存在的旧版 PyInstaller。
打包前会强制安装并逐项导入检查 pandas、NumPy、scikit-learn、Pillow 等运行依赖；依赖或测试失败时不会继续生成可能无法启动的 EXE。

## 数据要求

训练表中一行应尽量对应一个独立实验组或运行工况，而不只是整篇论文。需要保留来源、实验组、采样时间、进出水角色、湿地类型、植物、填料、尺寸、HRT、HLR 和环境条件。自动抽取目前只能生成待复核数据，正式建模前应逐条校验。

自动抽取会将记录分为 `accepted_candidate`、`review_required` 和 `rejected`。只有单位匹配、关键词距离合理且水质角色明确的高置信度候选会自动进入训练宽表；这仍不代表数据已经获得科研层面的人工确认。

`examples/sample_training_data.csv` 仅用于演示格式，不是科研数据。

## 已知限制

- PDF 表格、扫描件、上下标和多栏排版可能导致抽取错误。
- 当前 Imputer 只学习数值字段；`flow_type` 等结构类别由用户输入或 schema 默认值提供。
- OpenSCAD 输出是参数化概念模型，未包含结构配筋、详细填料层、植物配置和施工节点。
- 样本过少或来源高度相似时，交叉验证指标不具有代表性。
- 当 CSV 缺少来源列或独立来源不足 3 个时，训练会降级为普通 KFold；报告中的 `validation_strategy` 会明确记录这一点。

许可证为 GPL-3.0-or-later。
