# 真实模型部署

本页只安装已有推理工件，不含训练步骤，也不需要数据集。
视觉模型包位于 [v0.1.0 Release](https://github.com/fusq-zt/tbx-agent/releases/tag/v0.1.0)。
当前仓库为私有：在浏览器中登录已获授权的 GitHub 账号，从 Assets 下载
`tbx-rank03-inference-v1.zip`，然后执行本地安装。公开可见性与源码许可证尚待维护者确认。

## 1. 环境

安装 Git CLI，并确认 `git --version` 可用；D-FINE 源码准备需要 Git。

完成 [快速部署](quickstart.md) 的虚拟环境与外部目录设置。下面的 `python` 指虚拟环境解释器；
Windows 可使用 `& .\.venv\Scripts\python.exe`。

```bash
python -m pip install -e ".[ui,dicom,vision,anatomy]"
```

真实推理需要相容的 PyTorch、torchvision 与目标设备驱动。此处不承诺统一显存门槛、
固定延迟或任意 CUDA 组合可用，需在目标设备检查。

## 2. 视觉包

当前包为 `tbx-rank03-inference-v1.zip`，大小 **219,595,278 bytes（209.42 MiB）**，SHA-256：

```text
0312172e649dffd774311db397066a916dc65845627aa1489049bfa36d971594
```

包内为 `manifest.json`、`classifier.pt`、`detector.pt`、`classifier_config.json`、
`detector_config.json`。权重只保留推理状态，不含优化器、数据、预测或训练日志。
导出后张量一致性检查不等于重新测得模型性能。

Windows：按提示输入已有 ZIP 的完整路径。

```powershell
$visionBundle = Read-Host '视觉推理 ZIP 的完整路径'
& .\.venv\Scripts\python.exe scripts\install_vision_bundle.py `
  --bundle $visionBundle `
  --sha256 0312172e649dffd774311db397066a916dc65845627aa1489049bfa36d971594 `
  --artifact-root $env:TBX_ARTIFACT_ROOT `
  --runtime-config $env:TBX_AGENT_RANK03_RUNTIME_CONFIG
```

Linux：

```bash
printf '视觉推理 ZIP 的完整路径: '
read -r VISION_BUNDLE
python scripts/install_vision_bundle.py \
  --bundle "$VISION_BUNDLE" \
  --sha256 0312172e649dffd774311db397066a916dc65845627aa1489049bfa36d971594 \
  --artifact-root "$TBX_ARTIFACT_ROOT" \
  --runtime-config "$TBX_AGENT_RANK03_RUNTIME_CONFIG"
```

安装器校验清单与文件哈希，写入外部模型根，并生成 `rank03_runtime.json`。
目标运行契约必须是新文件；重复安装或升级时使用新的输出路径，并在检查后切换环境变量。
当前私有 Release 请使用本地 `--bundle`。安装器的匿名 `--url` 不支持 GitHub 私有附件认证；
不要把令牌拼到 URL 中。只有日后提供可匿名访问的 HTTPS 地址时，才可使用 `--url` 并同时提供 `--sha256`。
不要把仓库模板当成已安装契约，也不要绕过模型身份校验。

## 3. D-FINE 与 PSPNet

D-FINE 权重需要固定的上游推理源码；PSPNet 用于肺野分割。显式执行：

Windows：

```powershell
& .\.venv\Scripts\python.exe scripts\bootstrap_dfine.py --cache-dir $env:TBX_ARTIFACT_ROOT
& .\.venv\Scripts\python.exe scripts\bootstrap_models.py --cache-dir $env:TBX_ARTIFACT_ROOT download anatomy
& .\.venv\Scripts\python.exe scripts\bootstrap_models.py --cache-dir $env:TBX_ARTIFACT_ROOT verify anatomy
```

Linux：

```bash
python scripts/bootstrap_dfine.py --cache-dir "$TBX_ARTIFACT_ROOT"
python scripts/bootstrap_models.py --cache-dir "$TBX_ARTIFACT_ROOT" download anatomy
python scripts/bootstrap_models.py --cache-dir "$TBX_ARTIFACT_ROOT" verify anatomy
```

PSPNet 来自 [TorchXRayVision 官方 Release](https://github.com/mlmed/torchxrayvision/releases/tag/v1)，
大小和 SHA-256 固定在 [模型清单](../../configs/model_sources.yaml)。
这些命令不下载分类/检测权重。MedSAM 是另一组可选模型，默认关闭，无需为本流程安装。

## 4. MedGemma

按 [MedGemma 运行时](medgemma_runtime.md) 使用 `--download` 或已有 GGUF。
Windows 可准备固定 Windows llama.cpp；Linux 先构建并登记本平台运行时。
本项目不加载 MedGemma 的 `mmproj`，胸片由专用工具分析。

## 5. 启动

Windows：

```powershell
$env:TBX_AGENT_VISION_BACKEND = 'rank03'
$env:TBX_AGENT_REQUIRE_REAL_INFERENCE = 'true'
$env:TBX_AGENT_NARRATOR_BACKEND = 'llama_cpp'
$env:TBX_AGENT_REQUIRE_LLM_INFERENCE = 'true'
$env:TBX_AGENT_CONTOUR_REFINEMENT_BACKEND = 'none'
& .\.venv\Scripts\python.exe scripts\check_setup.py --mode vision
& .\scripts\run_local.ps1
```

Linux：

```bash
export TBX_AGENT_VISION_BACKEND=rank03
export TBX_AGENT_REQUIRE_REAL_INFERENCE=true
export TBX_AGENT_NARRATOR_BACKEND=llama_cpp
export TBX_AGENT_REQUIRE_LLM_INFERENCE=true
export TBX_AGENT_CONTOUR_REFINEMENT_BACKEND=none
python scripts/check_setup.py --mode vision
sh scripts/run_local.sh
```

检查 [就绪状态](http://127.0.0.1:8000/readyz) 和
[能力列表](http://127.0.0.1:8000/v1/system/capabilities)。
分类 argmax、检测与二维空间描述由契约固定；安装成功不构成模型性能或临床验证。
