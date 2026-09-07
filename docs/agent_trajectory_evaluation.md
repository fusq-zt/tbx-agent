# Agent 轨迹评测

当前评测对象是由真实 LangGraph `StateGraph` 执行的 Plan+ReAct：先生成不含隐藏推理的
公开 Plan，再在每次 Observation 后选择直接回答或一个公开工具。初始 Plan 是目标与证据需求，
不是预先固定的工具列表。评测同时校验实际 `graph_node_trace`，以防止只在旧循环外包一层图适配器。

## 评测问题

一条有效轨迹需要回答：

1. `initial_plan` 是否清楚表达目标和证据需求，且没有持久化 CoT；
2. ReAct 是否每步只直接回答或调用一个允许工具；
3. `model_tool_name` 是否准确反映模型动作，内部 handler 是否保持可审计；
4. 已存在且版本一致的病例证据是否被复用；
5. 工具 Observation 是否导致继续、修订 Plan 或终止；
6. 技术失败、空定位、证据缺口和前置条件缺失是否被区分；
7. 最终回答是否只使用真实 receipt、病例事实和适用的指南证据；
8. 容量饱和的单次恢复是否保留两个真实 attempt，其他失败是否未循环重试；
9. 轨迹与用户回答是否不含密钥、原始图像、完整指南正文、内部状态和隐藏推理；
10. 能力说明和已完成病例摘要是否走零工具可信投影，且不把二维肺野改写成肺叶。

## 公开动作与轨迹结构

模型动作空间严格限定为：

```text
classify_cxr
localize_cxr
analyze_lung_anatomy
search_tb_knowledge
```

上传解码/基础质控、普通问答、病例状态和纵向比较能力缺口均为零工具回答。当前没有经过验证的
纵向比较模型，不能输出“好转、恶化或稳定”。

`execution_plan` 是 Plan+ReAct 的主脱敏轨迹：

```text
source=plan_react
initial_plan
  plan_id / revision / goal
  steps[]: id / objective / evidence_need / status
plan_metadata
plan_revisions[]
  revision / trigger / reason_code / steps
react_steps[]
  step_index / plan_revision
  outcome=tool_call|answer
  tool_name|null / selection_mode / status / observation_code / recovery
tool_names[]                  # 实际公开工具历史，不是初始计划
graph_node_trace[]            # 实际 LangGraph 节点名，不含状态内容或 CoT
hidden_reasoning_persisted=false
```

每个实际执行 attempt 另有 `ToolReceipt`。评测优先使用 `model_tool_name`；`tool_name` 保留内部 handler
身份。receipt 还绑定 plan/step/call/request/trace ID、状态、attempt、输入/上下文/输出哈希、权限、
前置条件、Observation code 和指南 `resolved_*`。`AgentRunTrace` v2 继续提供高层 TaskSpec、预算和
终止原因，但不再被用来伪造旧式逐步决策。

## 必须覆盖的对照轨迹

同一个“病灶在哪里”任务至少覆盖三条真实 Observation 路径：

| 工具结果 | 后续行为 |
| --- | --- |
| 定位成功且有框 | 保存候选区域并回答位置 |
| 定位成功但无框 | 明确“本次无达到门槛的候选”，不重跑到有框、不改变分类 |
| 定位失败或不可用 | 保留技术失败并说明未完成，不伪装为空结果 |

其他关键轨迹：

- 上传后无工具；分类问题只运行 `classify_cxr`；
- 分类解释复用既有证据，不自动运行定位或 RAG；
- 肺区问题按需逐步执行 `localize_cxr -> Observation -> analyze_lung_anatomy`；
- 只要求肺野时不必运行定位，默认界面不展示肺野图层；
- 图像质量请求读取上传质控，不运行四工具中的任何一个；
- 复合问题在多个 ReAct step 中依次补齐证据，每步最多一个工具；
- 普通问答、能力说明、病例状态不误调视觉或指南工具；
- 能力说明必须描述 TBX-Agent 的实际能力，不能退化成“可完成各种任务”的通用 AI 自述；
- “汇总已完成分析”必须只投影现有公开证据，不输出 `thought/思考`、`case_state`、Plan 载荷、
  “诊断为 TB”或“上叶/中叶/下叶”等越界改写；
- 原生无工具正文、JSON fallback 和通用问答均覆盖推理/内部状态泄漏拒绝测试；
- 无既往片或纵向能力不可用时零工具返回明确缺口；
- 最大步数/工具数/成本到达上限后受控终止；
- 仅 `receipt.retryable=true` 的容量饱和在预算内恰好重试一次；
- 任意定位/解剖路径保持原始 `predicted_class` 不变；
- 指南问题只调用 `search_tb_knowledge(query)`，检索维度由工具内解析并写入 `resolved_*`。

## 指标

Agent 工程评测至少分别报告：

- 高层工具选择准确率；
- 无关工具调用率、前置条件违规数和重复工具调用率；
- Plan 目标/步骤质量与 Plan revision 记录有效性；
- ReAct step 的 0/1 工具基数、公开 allowlist 合规和 receipt 对齐率；
- Observation 闭环覆盖率与失败恢复/受控降级率；
- evidence reuse、预算违规、时延和成本；
- 指南 scope/subtopic/population/scenario 解析准确率；
- 指南 evidence applicability、citation 完整率和无证据回答一致性；
- 最终回答与 receipt 一致率；
- secret/PHI/hidden-reasoning 泄漏率。

这些是软件与 Agent 控制质量指标，不是诊断性能或临床安全指标。

## 本地验证

不加载真实模型的核心回归：

```bash
pytest tests/test_plan_react.py tests/test_plan_react_runtime.py tests/test_llm_tool_calling.py
pytest tests/test_medical_dialogue_runtime_evaluator.py tests/test_medical_dialogue_qa_matrix.py
pytest tests/test_trajectory_evaluation.py tests/test_agent_runtime_evaluation_cli.py
pytest tests/test_tb0077_followup_regression.py
```

默认真实控制器回归使用 `evaluation/suites/trajectory_v3`，公开动作使用上述四工具，receipt 必须满足
`tbx-tool-contract-v7`。v1/v2 fixture 保持只读，只用于显式历史重放；旧工具名只能由 evaluator 的
兼容投影读取，不能重新进入当前模型动作空间。

真实模型端到端评测还需显式准备视觉权重、固定病例和 LLM 服务，并分别标记 `mock/unit`、
`integration`、`requires_model` 与 `real_rank03`。不得把 mock 轨迹称为真实模型闭环，也不得使用
locked local test 或 official hidden test 选择 Agent 策略、阈值或提示词。
