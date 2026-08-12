$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv (Join-Path $projectRoot ".venv")
}

& $venvPython -m pip install --upgrade -r requirements-build.txt
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败，停止打包。" }
& $venvPython build_preflight.py
if ($LASTEXITCODE -ne 0) { throw "运行依赖检查失败，停止打包。" }
& $venvPython -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "测试失败，停止打包。" }
Remove-Item -LiteralPath (Join-Path $projectRoot "build") -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $projectRoot "dist\WetlandDesigner.exe") -Force -ErrorAction SilentlyContinue
& $venvPython -m PyInstaller --clean --noconfirm wetland_designer.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 打包失败。" }

$exePath = Join-Path $projectRoot "dist\WetlandDesigner.exe"
if (-not (Test-Path -LiteralPath $exePath)) { throw "未生成 WetlandDesigner.exe。" }

Write-Host "构建完成: $exePath"
