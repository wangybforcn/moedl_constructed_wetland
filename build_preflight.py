"""Windows 打包前验证所有运行依赖可导入。"""

import importlib
import sys


REQUIRED_MODULES = (
    "joblib",
    "numpy",
    "openpyxl",
    "pandas",
    "pdfminer",
    "PIL",
    "sklearn",
    "scipy",
)


def main():
    failures = []
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            print("OK {:<12} {}".format(module_name, getattr(module, "__version__", "")))
        except Exception as exc:
            failures.append("{}: {}".format(module_name, exc))
    if failures:
        print("\n打包依赖检查失败：", file=sys.stderr)
        for failure in failures:
            print("- " + failure, file=sys.stderr)
        raise SystemExit(1)
    import user_gui  # noqa: F401
    print("OK user_gui")


if __name__ == "__main__":
    main()

