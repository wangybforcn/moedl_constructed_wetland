# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ["user_gui.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        "sklearn.experimental.enable_iterative_imputer",
        "sklearn.impute._iterative",
        "PIL._tkinter_finder",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="WetlandDesigner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
