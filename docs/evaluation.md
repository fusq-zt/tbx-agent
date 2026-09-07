# 软件测试与运行检查

本发布版保留软件回归和推理检查，不包含训练流程。报告需区分合成输入的软件契约、
已准备模型的真实运行与独立数据上的性能；这些都不能直接当作临床验证。

## v0.1.0 本地验证记录

本版在 Windows / Python 3.13.12 上完成以下检查：

| 检查 | 结果与范围 |
| --- | --- |
| 完整 pytest | 1,149 passed、3 skipped，143.38 秒；跳过的平台权限或可选依赖条件以 pytest 报告为准 |
| Ruff 与源码发布检查 | 通过；检查敏感文件、训练代码和其他排除内容 |
| 真实启动 Demo | 5 个 HTTP 端点返回 200，验证后已停止服务；视觉结果仍为模拟 |
| 视觉包安装与输出一致性 | 已实际安装；合成图像上，CPU 推理的打包前后分类与最终检测输出完全一致 |

视觉输出一致性只验证推理工件导出和安装没有改变这次输入的结果，不是模型性能评估。
Docker 未实际运行。GitHub Windows / Linux CI 状态应查看仓库的 Actions 记录，
这里不把尚未确认完成的 CI 写成通过。

## 覆盖层级

| 层级 | 检查内容 | 是否需要模型 |
| --- | --- | --- |
| 单元与 API | 解码、病例隔离、分类策略、报告、回执、失败关闭 | 默认不需要 |
| Agent 轨迹 | 四工具 allowlist、每步 0/1 工具、证据复用、预算与失败恢复 | mock 默认不需要 |
| 软件符合性 | 合成 DICOM、PSPNet / MedSAM DTO、二维空间计算、UI 门控 | 不需要 |
| RAG fixture | 固定 corpus / queries / qrels、来源过滤、引用与无证据语义 | BM25 不需要 |
| 真实运行 | 固定视觉工件、MedGemma 协议、实际加载与工具闭环 | 显式准备后执行 |

普通测试使用合成文本、临时图像和替身。不要把跳过模型测试的报告写成“真实模型已验证”。

## 最小回归

在虚拟环境中运行：

```bash
python -m pip install -e ".[ui,dev]"
python -m ruff check src ui tests scripts
python -m pytest -m "not real_rank03 and not requires_model and not slow"
python scripts/build_rag_index.py --dry-run
```

具体语义要求见 [Agent 轨迹评测](agent_trajectory_evaluation.md)。
合成测试可检查适配器输入输出契约，但不测量真实分类、检测或分割质量。

## 保存报告

Linux 示例会创建仓库外的新运行目录；Windows 在 `$env:TBX_AGENT_DATA_ROOT` 下选择新目录，
向各命令传相同参数即可。

```bash
export TBX_AGENT_DATA_ROOT="${TBX_AGENT_DATA_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/tbx-agent}"
RUN_DIR="$TBX_AGENT_DATA_ROOT/evaluation/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
python scripts/evaluate_agent_runtime.py --runtime-mode mock --output "$RUN_DIR/trajectory.json"
python scripts/evaluate_rag_retrieval.py \
  --suite-config evaluation/retrieval/smoke_v5/config.json \
  --modes bm25 --output "$RUN_DIR/bm25.json"
tbx-agent-system-bench \
  --config evaluation/system_bench_config_v1_6.json --output-dir "$RUN_DIR/system"
tbx-agent-software-conformance \
  --config evaluation/software_conformance_config_v1.json --output-dir "$RUN_DIR/software"
```

每次使用新目录并保留失败报告。系统 benchmark 绑定源码、策略、知识和固定 suite；
software-conformance 检查合成契约，不加载 PSPNet / MedSAM。
RAG 保存检索模式、配置和身份，并把成功、失败与中断记入外部台账，详见 [RAG](retrieval.md)。

fixture 是工程回归材料，不是医学专家金标准，不能用于选择模型、阈值或临床策略。
比较时应固定 suite 与 corpus 身份；不能把历史结果写成本次发布重新运行的结果。

默认部署策略为 `configs/fusion_policy.json`。
`fusion_policy_argmax_v2.json` 与 `fusion_policy_sens98_legacy.json` 保留为历史兼容回归材料；
其历史指标和报告引用不构成发布包性能背书，也不是建议部署者切换的策略。

## 真实运行

先完成 [模型部署](deployment/inference_models.md) 与 `/readyz` 检查。
评测脚本不自动下载安装模型。对已经运行的服务，可显式执行固定对话矩阵：

```bash
python scripts/evaluate_medical_dialogue_runtime.py \
  --base-url http://127.0.0.1:8000 \
  --provider local_medgemma \
  --output "$RUN_DIR/medical-dialogue.json"
```

runner 每例使用独立 thread，检查实际响应、回执、引用与错误状态；不会启动或停止服务，
也不从命令行接收 API key。输出文件必须是新文件。

HTTP 2xx、模型能加载、张量一致或输出形状正确，不等于回答契约全部通过或模型性能达标。
源码、模型、配置与知识身份应和报告一起保留；无法测得显存等字段时记录原因。
模型更新需要新的运行证据，不继承另一组权重的性能数字。
