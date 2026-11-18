# 组装 Windows 发行包：dist/Gomoku/ 目录与 dist/Gomoku-win64.zip。
#
# 由 packaging/build.ps1 调用；也可单独运行（前置产物需已存在）：
#   powershell -ExecutionPolicy Bypass -File packaging/pack.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$out = Join-Path $root "dist/Gomoku"
$envDirs = @("classical_worker", "llm_worker", "az_worker")

if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force $out, "$out/configs", "$out/models/az" | Out-Null

foreach ($path in @("dist/Gomoku.exe", "dist/GomokuApp.exe", "dist/GomokuWorkers/GomokuWorkers.exe", "dist/run_last.pt")) {
    if (-not (Test-Path $path)) { throw "缺少 $path，先跑 packaging/build.ps1" }
}

Copy-Item dist/Gomoku.exe $out
Copy-Item dist/GomokuApp.exe $out
Copy-Item dist/GomokuWorkers (Join-Path $out "runtime") -Recurse
Copy-Item dist/run_last.pt "$out/models/az/run_last.pt"

$utf8 = New-Object System.Text.UTF8Encoding $false

# 引擎注册表：令牌留空，主程序启动时随机生成并写入 engines.runtime.json
# （auth.NewRegistry 对空 token 调 RandomToken，示例文件里的占位串会变成真令牌）。
$registry = @'
{
  "local_client_token": "",
  "engines": [
    { "engine_id": "classical", "display_name": "Classical AI", "token": "", "available": true },
    { "engine_id": "llm", "display_name": "LLM AI", "token": "", "available": true },
    { "engine_id": "az", "display_name": "AlphaZero AI", "token": "", "available": true }
  ]
}
'@
[System.IO.File]::WriteAllText((Join-Path $out "configs/engines.json"), $registry, $utf8)

# 引擎参数文件（模板不含密钥；llm 的 Key 由用户自己填）。
foreach ($dir in $envDirs) {
    $target = Join-Path $out $dir
    New-Item -ItemType Directory -Force $target | Out-Null
    $src = Join-Path $root "$dir/.env.example"
    if (Test-Path $src) { Copy-Item $src (Join-Path $target ".env") }
    else { New-Item -ItemType File -Force (Join-Path $target ".env") | Out-Null }
}

# 随包只有 run_last.pt，模板里的 current.pt 不存在。
$azEnv = Get-Content (Join-Path $out "az_worker/.env")
$azEnv = $azEnv -replace '^AZ_MODEL_PATH=.*$', 'AZ_MODEL_PATH=models/az/run_last.pt'
[System.IO.File]::WriteAllText((Join-Path $out "az_worker/.env"), ($azEnv -join "`r`n") + "`r`n", $utf8)

Copy-Item packaging/README-发行版.md "$out/使用说明.md"

$zip = Join-Path $root "dist/Gomoku-win64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path "$out/*" -DestinationPath $zip

$size = [math]::Round((Get-Item $zip).Length / 1MB)
Write-Host "发行包: $out ($size MB)"
