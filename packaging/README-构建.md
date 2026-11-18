# 构建 Windows 发行包

产物：`dist/MyGo/`（可直接运行目录）与 `dist/MyGo-win64.zip`（分发包）。

```powershell
# 一次性准备打包环境（CPU 版 torch，GPU 版会把包撑到数 GB）
D:/python/python.exe -m venv .venv-pack
.venv-pack/Scripts/python.exe -m pip install pyinstaller numpy
.venv-pack/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 每次发布
powershell -ExecutionPolicy Bypass -File packaging/build.ps1
```

四个步骤（`build.ps1` 依次执行）：

| 步骤 | 命令 | 产物 |
| --- | --- | --- |
| 1 | `go build ./cmd/app` | `dist/mygo.exe`（API + 桌面 UI） |
| 2 | `go build ./packaging/launcher` | `dist/Gomoku.exe`（启动器） |
| 3 | `PyInstaller --onedir packaging/worker_entry.py` | `dist/workers/`（冻结 Python + torch） |
| 4 | `export_model.py` + `pack.ps1` | `dist/MyGo/`、`dist/MyGo-win64.zip` |

## 为什么是 CPU 版 torch

发行包里的 `workers/` 用 CPU 版 torch：

- CUDA 版 `torch` 单独 4.2 GB，PyInstaller 打出来的目录装不进普通 zip 分发；
- AlphaZero 每手 2 秒预算在 CPU 上完全够用（网络是 64 通道 × 6 残差块）；
- 想让发行包跑 GPU：把 `.venv-pack` 换成装 GPU torch 的环境重新打包 3–4 步，
  启动器无需改动。

## 排除项

`--exclude-module` 去掉训练侧代码（`az_worker.train/selfplay/dataset/benchmark`、
`classical_worker.benchmark`）与 `matplotlib/pandas/IPython/tensorboard`。
`worker_entry.py` 只按 `--worker` 名动态导入三个 `__main__`，所以要用
`--collect-submodules` 把包内模块显式收集进来。

## 校验

```powershell
# Worker 能起来（参数文件缺失时报错即可，说明入口通了）
dist/workers/workers.exe --worker classical --env-file dist/MyGo/classical_worker/.env

# 完整发行目录
cd dist/MyGo; ./MyGo.exe
```
