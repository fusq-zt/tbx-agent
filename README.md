<div align="center">

# TBX-Agent

**面向胸片分析与指南问答的本地智能体**

本地部署的胸部 X 线分析研究原型 · 仅推理发布版

![Python](https://img.shields.io/badge/Python-3.11%E2%80%933.13-3776AB?style=flat-square)
![API](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square)
![Deployment](https://img.shields.io/badge/Deployment-Local--first-475569?style=flat-square)

[核心能力](#核心能力) · [模型性能](#模型性能) · [工作流程](#工作流程) · [使用真实模型](#使用真实模型) · [项目文档](#项目文档)

</div>

TBX-Agent 将胸片分类、候选区域定位、肺野分割和指南检索接入同一条对话流程。
用户提出问题后，Agent 根据已有信息规划任务、按需调用工具，再结合分析结果和文献证据生成回答。
项目提供 **Streamlit 可视化界面**与 **FastAPI 接口**，支持在本机部署模型与保存病例记录。

## 核心能力

| 能力 | 方法 | 使用效果 |
| --- | --- | --- |
| 胸片分类 | ConvNeXt-Tiny | 展示胸片三分类结果 |
| 候选区域定位 | D-FINE | 标出候选区域，辅助查看影像中的局部信息 |
| 肺野空间分析 | PSPNet | 分割左右肺，描述上、中、下二维区域 |
| 指南问答 | BM25 / 可选 Dense、Hybrid RAG | 检索指南证据，提供可追溯引用 |
| 任务编排 | Plan + ReAct | 根据问题和病例状态选择下一步工具 |
| 对话与记录 | MedGemma + 本地病例管理 | 组织文本回答，支持问卷、病例记录与报告 |

默认语言层为 **MedGemma 1.5 4B Q4_K_M + llama.cpp**，基于专用工具产生的文本证据组织回答。
视觉分析由用户问题触发，上传时完成影像解码、质量检查与病例建立。

## 模型性能

在 **TBX11K 官方测试集**上，本项目视觉模型的类别无关检测（合并结核子类）取得 **AP50 71.11、AP75 30.58、AP@[0.50:0.95] 35.38**，较所对照的[历史榜单](https://competitions.codalab.org/competitions/25848#results)最佳值分别提升 **9.37、10.94、8.06 个百分点**，在严格定位评价和综合 AP 上表现突出。分类 **Accuracy 为 91.31%、AUC 为 97.11%**，六项分类指标中有五项优于 Faster R-CNN-ResNet50 基线。

## 工作流程

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, Microsoft YaHei, sans-serif", "fontSize": "15px", "lineColor": "#94A3B8", "primaryTextColor": "#243247", "edgeLabelBackground": "#FFFFFF", "clusterBkg": "#F8FAFC", "clusterBorder": "#E2E8F0"}, "flowchart": {"curve": "basis", "nodeSpacing": 28, "rankSpacing": 50, "padding": 18, "wrappingWidth": 280}}}%%
flowchart TB
    subgraph INPUT["01　输入与上下文"]
        direction LR
        QUESTION("用户提问<br/>影像分析 · 指南问答")
        IMAGE("胸片 · 可选上传<br/>解码 · 质控 · 建立病例")
    end

    AGENT("02　规划与决策<br/><b>Plan + ReAct</b><br/>状态检查 · 预算约束")

    subgraph TOOLS["03　按需选择工具"]
        direction LR
        CLASSIFY("胸片分类<br/><b>ConvNeXt-Tiny</b>")
        LOCALIZE("候选区域定位<br/><b>D-FINE</b>")
        ANATOMY("肺野空间分析<br/><b>PSPNet</b>")
        KNOWLEDGE("指南证据检索<br/><b>BM25 · RAG</b>")
    end

    EVIDENCE("证据与工具回执<br/>结果 · 来源 · 执行状态")
    ANSWER("04　校验后输出<br/><b>回答 · 按需报告</b><br/>附适用引用")

    QUESTION --> AGENT
    IMAGE -. 提供病例上下文 .-> AGENT
    AGENT --> CLASSIFY & LOCALIZE & ANATOMY & KNOWLEDGE
    CLASSIFY & LOCALIZE & ANATOMY & KNOWLEDGE --> EVIDENCE
    EVIDENCE -. 更新状态 · 继续决策 .-> AGENT
    AGENT --->|无需新工具| ANSWER

    classDef input fill:#F1F5F9,stroke:#CBD5E1,color:#334155,stroke-width:1.5px
    classDef agent fill:#EEF2FF,stroke:#818CF8,color:#3730A3,stroke-width:2px
    classDef vision fill:#EFF6FF,stroke:#93C5FD,color:#1E40AF,stroke-width:1.5px
    classDef knowledge fill:#ECFDF5,stroke:#6EE7B7,color:#065F46,stroke-width:1.5px
    classDef evidence fill:#FFFBEB,stroke:#FCD34D,color:#92400E,stroke-width:1.5px
    classDef output fill:#0F766E,stroke:#0F766E,color:#FFFFFF,stroke-width:2px

    class QUESTION,IMAGE input
    class AGENT agent
    class CLASSIFY,LOCALIZE,ANATOMY vision
    class KNOWLEDGE knowledge
    class EVIDENCE evidence
    class ANSWER output
    style INPUT fill:#F8FAFC,stroke:#E2E8F0,stroke-width:1px,color:#64748B
    style TOOLS fill:#F8FAFC,stroke:#E2E8F0,stroke-width:1px,color:#64748B
```

Agent 每步最多调用一个工具，根据返回结果决定继续分析还是回答；已有信息足够时可以直接回答，
信息不足时明确说明缺口。

## 使用真实模型

准备 **Python 3.11–3.13、Git** 和可联网的安装环境。本仓库仅提供推理代码，
模型权重单独下载，无需训练或准备数据集。

| 模型 | 用途 | 获取方式 |
| --- | --- | --- |
| ConvNeXt-Tiny + D-FINE | 分类与候选定位 | 从 [Release](https://github.com/fusq-zt/tbx-agent/releases/tag/v0.1.0) 下载 `tbx-rank03-inference-v1.zip` |
| PSPNet | 肺野分割 | 下方脚本从 TorchXRayVision 下载 |
| MedGemma 1.5 4B | 对话与回答组织 | 下方脚本下载 GGUF 并准备 llama.cpp |

当前仓库和 Release 为私有，请先登录已获访问权限的 GitHub 账号。

### 1. 获取项目

```bash
git clone https://github.com/fusq-zt/tbx-agent.git
cd tbx-agent
```

### 2. 安装环境与模型

下面是 **Windows x64 / PowerShell 首次部署**的完整步骤。
Linux 请参考[视觉模型部署](docs/deployment/inference_models.md)与
[llama.cpp 构建说明](docs/deployment/medgemma_runtime.md#linux)。

先下载上表的视觉 ZIP，无需手动解压；在项目目录执行：

```powershell
# 模型与运行数据存放在源码目录外
$tbxRoot = Join-Path $env:LOCALAPPDATA 'TBX-Agent'
$env:TBX_ARTIFACT_ROOT = Join-Path $tbxRoot 'artifacts'
$env:TBX_RUNTIME_ROOT = Join-Path $tbxRoot 'runtime'
$env:TBX_AGENT_DATA_ROOT = $env:TBX_RUNTIME_ROOT
$env:TBX_AGENT_RANK03_RUNTIME_CONFIG = Join-Path $env:TBX_RUNTIME_ROOT 'config\rank03_runtime.json'
$visionBundle = Read-Host '已下载的视觉 ZIP 完整路径'

# 创建虚拟环境、安装依赖和分类 / 检测权重
& .\scripts\bootstrap.ps1 -Extras 'ui,dicom,vision,anatomy' `
  -VisionBundle $visionBundle `
  -CacheDir $env:TBX_ARTIFACT_ROOT `
  -RuntimeConfig $env:TBX_AGENT_RANK03_RUNTIME_CONFIG

# 准备检测推理源码与肺野分割模型
& .\.venv\Scripts\python.exe scripts\bootstrap_dfine.py --cache-dir $env:TBX_ARTIFACT_ROOT
& .\.venv\Scripts\python.exe scripts\bootstrap_models.py --cache-dir $env:TBX_ARTIFACT_ROOT download anatomy

# 准备 MedGemma 与本地 llama.cpp 服务
& .\.venv\Scripts\python.exe scripts\setup_medgemma.py --download --runtime-root $env:TBX_RUNTIME_ROOT
```

使用模型前需遵守上游条款；访问设置与已有 GGUF 的使用方式见[语言模型部署](docs/deployment/medgemma_runtime.md)。
如自定义存放目录，请将对应路径写入本地 `.env`，便于后续启动。

### 3. 启动应用

```powershell
& .\scripts\run_local.ps1
```

启动器会检查模型并管理本地服务。打开 **[交互界面](http://127.0.0.1:8501)**，
即可上传胸片、发起分析或进行指南问答；接口说明位于 **[API 文档](http://127.0.0.1:8000/docs)**。

默认指南检索使用 BM25。Dense / Hybrid 检索与 MedSAM 轮廓可视化均为可选扩展。

## 项目文档

| 想了解什么 | 文档 |
| --- | --- |
| 模型安装与 Linux 部署 | [真实模型部署](docs/deployment/inference_models.md) |
| MedGemma 与 Linux 运行时 | [语言模型部署](docs/deployment/medgemma_runtime.md) |
| 容器、挂载与网络 | [Docker](docs/deployment/docker.md) |
| 工具调度、病例状态与接口 | [架构](docs/architecture.md) · [API](docs/api.md) |
| 指南来源、索引与回执 | [RAG](docs/retrieval.md) · [知识摄取](docs/knowledge_ingestion.md) |
| 测试与能力限制 | [测试说明](docs/evaluation.md) · [安全边界](docs/safety_case.md) |
| 参与项目 | [贡献指南](CONTRIBUTING.md) |

## 使用说明

本项目用于研究与信息辅助，不替代临床判断。肺野分区是二维区域，不代表解剖肺叶；
软件测试通过不等于临床验证。请仅使用已获授权且去标识的数据。

项目代码许可证待确定；第三方代码、模型与指南遵循各自条款，详见[第三方来源说明](THIRD_PARTY_NOTICES.md)。
