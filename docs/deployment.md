# 部署概览

TBX-Agent 是本地研究与信息辅助软件。本发布版仅含推理代码、配置、知识快照和软件测试，
不包含模型权重、数据集、原始指南文件或训练代码。

## 选择路径

1. [快速部署](deployment/quickstart.md)：准备 Python 环境和外部模型目录，启动真实模型服务。
2. [真实模型部署](deployment/inference_models.md)：安装外部视觉包，准备 D-FINE、PSPNet 和 MedGemma。
3. [Docker](deployment/docker.md)：已准备真实运行工件后的进阶容器路径。
4. [模型工件](deployment/model_artifacts.md)：目录与哈希边界。

普通启动要求所选必需模型就绪，不把缺失权重静默替换为模拟结果。

## 运行边界

默认 API / UI / llama.cpp 都只监听本机。模型缓存与运行数据分开，病例和报告写到外部数据根。
当前仅协调单进程内同一用户/线程请求；不提供多副本共享锁、分布式队列或机构级高可用。

Agent 使用 LangGraph 执行有界 ReAct，复杂问题按需规划，每步最多一个公开工具。
上传解码和基础质控不调用模型。病例状态与审计由业务存储管理，未配置 durable checkpointer，
因此不支持在进程重启后从图中间节点继续执行。

生命周期关闭会等待已接受的后台任务，再关闭检索、数据库和 LLM 连接。Python 模型线程不能安全强杀，
不承诺固定时间内退出。调用方注入的服务和存储由调用方负责关闭。

## LLM 与 RAG

默认 [MedGemma / llama.cpp](deployment/medgemma_runtime.md) 接收文本证据，不加载图像投影器。
外部 OpenAI-compatible 服务是显式替代项，会收到允许发送的文本上下文；部署者需审查认证、日志与留存。

[默认检索配置](../configs/retrieval.yaml) 为 BM25，不加载向量模型。
Dense / Hybrid 必须使用固定的 BGE-M3 本地快照与匹配索引，详见 [RAG](retrieval.md)。
MedSAM 默认关闭，详见 [轮廓可视化](medsam_refinement.md)。

## 检查与限制

- `check_setup.py`：只读安装检查；
- `/livez`：进程存活；
- `/readyz`：所选必需运行时探针；
- `/v1/system/capabilities` 和 `/v1/system/manifest`：能力及身份。

真实运行、模型性能和软件契约是不同证据，见 [测试说明](evaluation.md)。
服务对外前须有可信入口、TLS、认证授权、限流和数据治理；
现有实现与缺口见 [安全边界](production_security.md)。
