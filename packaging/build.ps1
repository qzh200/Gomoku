# 一键构建 Windows 发行包：dist/MyGo/ 与 dist/MyGo-win64.zip
#
#   powershell -ExecutionPolicy Bypass -File packaging/build.ps1
#
# 前置：能离线编译 Go（GOPROXY=off）；.venv-pack 是打包用虚拟环境，装 CPU 版
# torch（GPU 版会让发行包膨胀到数 GB），创建方式见 packaging/README-构建.md。

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:GOPROXY = "off"
New-Item -ItemType Directory -Force dist, build/pyinstaller | Out-Null

Write-Host "[1/4] mygo.exe（Go 主程序：API + 桌面 UI）"
go build -buildvcs=false -ldflags "-H windowsgui -s -w" -o dist/mygo.exe ./cmd/app
if ($LASTEXITCODE -ne 0) { throw "go build ./cmd/app 失败" }

Write-Host "[2/4] Gomoku.exe（启动器：主程序 + Worker 生命周期）"
go build -buildvcs=false -ldflags "-H windowsgui -s -w" -o dist/Gomoku.exe ./packaging/launcher
if ($LASTEXITCODE -ne 0) { throw "go build ./packaging/launcher 失败" }

Write-Host "[3/4] workers/（PyInstaller 冻结 Python + torch，onedir）"
$python = ".venv-pack/Scripts/python.exe"
$pyinstaller = @(
    "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--console",
    "--name", "workers", "--distpath", "dist",
    "--workpath", "build/pyinstaller", "--specpath", "build/pyinstaller", "--paths", ".",
    "--collect-submodules", "aiworker",
    "--collect-submodules", "classical_worker",
    "--collect-submodules", "llm_worker",
    "--collect-submodules", "az_worker",
    "--exclude-module", "az_worker.train",
    "--exclude-module", "az_worker.selfplay",
    "--exclude-module", "az_worker.dataset",
    "--exclude-module", "az_worker.benchmark",
    "--exclude-module", "classical_worker.benchmark",
    "--exclude-module", "matplotlib",
    "--exclude-module", "pandas",
    "--exclude-module", "IPython",
    "--exclude-module", "tensorboard",
    "packaging/worker_entry.py"
)
& $python @pyinstaller
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 构建失败" }

Write-Host "[4/4] 裁剪权重 + 组装目录 + 打 zip"
& $python packaging/export_model.py models/az/run_last.pt dist/run_last.pt
if ($LASTEXITCODE -ne 0) { throw "权重导出失败" }
& powershell -ExecutionPolicy Bypass -File packaging/pack.ps1
if ($LASTEXITCODE -ne 0) { throw "打包失败" }
