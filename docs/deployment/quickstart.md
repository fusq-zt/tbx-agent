# 快速部署

真实模式使用
[v0.1.0 Release](https://github.com/fusq-zt/tbx-agent/releases/tag/v0.1.0) 的视觉包与另外准备的语言模型。
当前仓库和附件均为私有，需登录获授权账号；公开可见性与源码许可证尚待维护者确认。

## 获取源码

使用有访问权限的 GitHub 账号克隆：

```bash
git clone https://github.com/fusq-zt/tbx-agent.git
cd tbx-agent
```

没有 Git 时，也可登录[仓库页面](https://github.com/fusq-zt/tbx-agent)，
选择 **Code → Download ZIP**，解压后进入源码根目录。

## Python 环境

支持 Python 3.11–3.13。以下命令均在源码根目录运行。

Windows PowerShell：

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[ui,dicom,vision,anatomy]"
```

Linux：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ui,dicom,vision,anatomy]"
```

也可使用 `scripts/bootstrap.ps1` 或 `sh scripts/bootstrap.sh` 创建环境并安装默认
`ui,dicom` 依赖。bootstrap 默认不下载模型，也不覆盖已有 `.env`。
显式的 `-VisionBundle` / `--vision-bundle` 才会安装已经准备的视觉 ZIP。

## 外部目录

下面是可执行的便携路径设置，可改成其他仓库外目录。

Windows：

```powershell
$env:TBX_ARTIFACT_ROOT = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'TBX-Agent\artifacts'
$env:TBX_AGENT_DATA_ROOT = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'TBX-Agent\runtime'
$env:TBX_RUNTIME_ROOT = $env:TBX_AGENT_DATA_ROOT
$env:TBX_AGENT_RANK03_RUNTIME_CONFIG = Join-Path $env:TBX_AGENT_DATA_ROOT 'config\rank03_runtime.json'
```

Linux：

```bash
export TBX_ARTIFACT_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/tbx-agent/artifacts"
export TBX_AGENT_DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/tbx-agent"
export TBX_RUNTIME_ROOT="$TBX_AGENT_DATA_ROOT"
export TBX_AGENT_RANK03_RUNTIME_CONFIG="$TBX_AGENT_DATA_ROOT/config/rank03_runtime.json"
```

| 变量 | 用途 |
| --- | --- |
| `TBX_ARTIFACT_ROOT` | 不可变模型和推理源码缓存 |
| `TBX_AGENT_DATA_ROOT` | 数据库、日志、进程状态等运行文件 |
| `TBX_AGENT_CASE_ARTIFACT_ROOT` | 可选，病例图像与报告；默认数据根下的 `cases` |
| `TBX_RUNTIME_ROOT` | llama.cpp 准备目录；建议与数据根一致 |
| `TBX_AGENT_RANK03_RUNTIME_CONFIG` | 视觉安装器生成的运行契约 |

终端变量只在当前会话生效。持久化时，从 `.env.example` 创建自己的 `.env`，填入解析后的真实路径；
不要覆盖已有配置。dotenv 不执行 Shell 展开，不能直接依赖 `$HOME`、`$env:...` 或命令替换。
已有非空进程变量优先于 dotenv。

## 下载模型并启动

完成 [真实模型部署](inference_models.md) 后，在设置过相同变量的终端运行：

```powershell
& .\.venv\Scripts\python.exe scripts\check_setup.py --mode vision
& .\scripts\run_local.ps1
```

```bash
python scripts/check_setup.py --mode vision
sh scripts/run_local.sh
```

启动器读取项目 `.env` 和外部 `config/llm.env`，管理本地 llama.cpp、预检、API 与 UI。
真实模式不要添加 Demo 参数。
启动后打开 [Streamlit](http://127.0.0.1:8501) 或 [FastAPI 文档](http://127.0.0.1:8000/docs)。
停止时在启动终端按 Ctrl+C。

| 现象 | 检查 |
| --- | --- |
| PowerShell 拒绝运行脚本 | 按终端管理策略允许本地脚本，或使用批准的终端环境 |
| 端口占用 | 检查已有进程，或协调设置 `TBX_AGENT_BIND_PORT`、`TBX_AGENT_API_URL`、`TBX_AGENT_UI_PORT` |
| 缺模型或模板未通过 | 安装匹配视觉包，使用生成的外部契约 |
| LLM 未就绪 | 核对 GGUF、llama.cpp bundle、key 文件和外部 `llm.env` |
| 下载或哈希失败 | 核对网络与工件版本，重试准备命令；不要改掉预期哈希 |
| 向量模式失败 | 默认 BM25 无需模型；Dense 需要固定模型和匹配索引 |
| 图像拒绝 | PNG/JPEG 为基础支持；DICOM 需 extra，仅受限单帧非压缩 CR/DX |

`check_setup.py` 只读，不下载或加载模型。`/livez` 表示进程存活，`/readyz` 检查所选必需运行时；
`/v1/system/capabilities` 展示能力，`/v1/system/manifest` 展示模型、知识和策略身份。
默认仅本机访问，按单进程设计。进一步见 [Docker](docker.md) 和 [安全边界](../production_security.md)。
