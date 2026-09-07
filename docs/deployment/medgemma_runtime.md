# MedGemma 与 llama.cpp

默认语言层为 MedGemma 1.5 4B 的 Q4_K_M GGUF，使用本机 llama.cpp OpenAI-compatible API。
它读取专用工具产生的文本证据，不直接分析上传图像；本项目不加载 `mmproj`。

## 固定工件

| 项目 | 值 |
| --- | --- |
| 基础模型 | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) |
| GGUF 来源 | [unsloth/medgemma-1.5-4b-it-GGUF](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF) |
| GGUF revision | `3855f948626b7ae42bccd082757f15078c53e758` |
| 文件 | `medgemma-1.5-4b-it-Q4_K_M.gguf` |
| 字节数 | `2489894976` |
| SHA-256 | `b31becdf4f39561800505514cce67681604fe449d04dd35c8c92fd7848c6d7bd` |
| 服务别名 | `tbx-medgemma-1.5-4b-it-q4-k-m` |
| llama.cpp | `b10517`，提交 `dc72703fc69698b1ea68ece8d2dd8a96e6a4e1fe` |

GGUF 来自上表量化发布者，不把它称为 Google 官方 GGUF。使用前遵守上游模型条款；
若上游要求账号与访问批准，先完成相应步骤。需要令牌时通过进程环境 `HF_TOKEN` 提供，
不要把真实令牌放进源码、命令参数、截图或报告。

## Windows

先完成 [快速部署](quickstart.md) 的环境与外部目录设置：

```powershell
& .\.venv\Scripts\python.exe scripts\setup_medgemma.py --download `
  --runtime-root $env:TBX_RUNTIME_ROOT
```

脚本下载固定 GGUF 到外部 `models` 目录，校验大小与 SHA-256，并在需要时准备固定 Windows
llama.cpp bundle。自动安装面向固定 Windows x64 bundle；其他平台或自定义构建需先登记运行时。

已有对应 GGUF 时，可不重复下载和复制：

```powershell
$ggufPath = Read-Host '已下载的 MedGemma Q4_K_M GGUF 完整路径'
& .\.venv\Scripts\python.exe scripts\setup_medgemma.py `
  --model $ggufPath --runtime-root $env:TBX_RUNTIME_ROOT
```

## Linux

Linux 不自动安装 Windows bundle。先准备 CMake、C++ 编译工具；CUDA 构建还需要匹配的 CUDA toolkit
与 `nvcc`。在虚拟环境中执行 CPU 构建：

```bash
sh scripts/build_llamacpp_linux.sh \
  --backend cpu --runtime-root "$TBX_RUNTIME_ROOT" --execute
```

如选 CUDA，将 `--backend cpu` 改成 `--backend cuda`。不加 `--execute` 时默认只展示计划。

构建完成会生成 `candidate_unregistered` 回执，并打印完整的 `register-runtime` 命令。
检查源码归档哈希、构建回执与实际二进制后，执行打印出的命令；构建成功不等于已登记。
登记入口目前仍名为 `scripts/bootstrap_qwen.py`，这里只使用通用 llama.cpp 登记子命令，
无需下载或构建 Qwen 模型。

登记完成后：

```bash
python scripts/setup_medgemma.py --download --runtime-root "$TBX_RUNTIME_ROOT"
```

已有 GGUF 时：

```bash
printf '已下载的 MedGemma Q4_K_M GGUF 完整路径: '
read -r MEDGEMMA_GGUF
python scripts/setup_medgemma.py --model "$MEDGEMMA_GGUF" --runtime-root "$TBX_RUNTIME_ROOT"
```

## 生成文件与启动

setup 写入：

- `<runtime-root>/config/llm_runtime.generated.yaml`：固定模型与运行时身份；
- `<runtime-root>/config/llm.env`：无密钥正文的路径和配置指针；
- `<runtime-root>/secrets/llama-server-api.keys`：受保护的 API key 文件。

让 `TBX_AGENT_DATA_ROOT` 与 `TBX_RUNTIME_ROOT` 保持一致，随后运行
`scripts/run_local.ps1` 或 `sh scripts/run_local.sh`，启动器会读取上述外部配置并管理本地服务。
若使用不同目录，显式设置 `TBX_AGENT_LLM_ENV_FILE` 指向生成的 `llm.env`。

默认模型 API 为 `http://127.0.0.1:11435/v1`。真实应用还需要
[视觉模型](inference_models.md) 就绪。不要将底层聊天接口的输出称为已经过 TBX-Agent 工具链验证。

切换模型或运行时时，应重新执行 [软件与运行时检查](../evaluation.md)；
旧模型报告不能作为当前配置的运行证明。
