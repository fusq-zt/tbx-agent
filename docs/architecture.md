# TBX-Agent 架构

本文描述当前源码中的主运行路径。TBX-Agent 是单进程、单病例上下文的研究原型，不是生产级
医疗平台。它不持久化 Controller 执行状态，也不提供分布式调度、多智能体或自由思维链。

## 1. 责任边界

```text
视觉模型            产生胸片证据，不读取指南、不生成临床结论
指南 RAG            返回经准入的指南 chunk 与 citation，不修改病例视觉事实
CaseRecord/SQLite    保存病例、工具结果、问卷和审计所需业务状态
CaseState            从 CaseRecord 派生的一轮运行时视图，不是第二个数据库
Plan 节点           生成目标与 evidence_need，不预授权整条工具链、不记录隐藏思维
ReAct 选择层        每次观察后只选择零个或一个公开工具，或直接回答
确定性代码           校验工具前置条件、证据适用性、预算与安全硬门
Agent Controller     根据结构化状态选择一个最小必要动作
可信零工具投影       在模型规划前确认能力/病例状态请求，由纯函数投影公开事实
回答组成层           病例/RAG 回答只组合已批准事实；模型直答先经过输出清洗与泄漏拒绝
```

模型输出与系统处置分开。`predicted_class=tb` 是 ConvNeXt 的三分类 argmax 输出；
`screening_disposition` 是交互结果状态，永远不能回写或覆盖前者。分类与 D-FINE 定位差异不构成
分类冲突，也不会把单病例对话送入复核工作台；只有显式批量筛查入口可以创建复核任务。

## 2. 输入和病例建立

上传 PNG/JPEG 或受限 DICOM 后，输入层完成解码、EXIF 方向、DICOM 窗宽窗位、MONOCHROME 反相、
基础质量检查、SHA-256 和去元数据像素制品。随后建立或幂等复用 `CaseRecord`。

上传阶段不应默认运行：

- ConvNeXt 分类；
- D-FINE 定位；
- PSPNet 肺野分割；
- 指南检索。

因此病例必须显式保存“未请求”，不能用空数组表示“模型运行后没有结果”。

## 3. 病例事实与运行时状态

### 3.1 CaseRecord

`schemas.py` 中的 `CaseRecord` 是 SQLite 的持久化病例事实。与 Agent 相关的关键字段包括：

- `classification_status` 和不可变 `vision_evidence`；
- 独立的 `localization_evidence`；
- 输入质量状态与代码；
- `screening_disposition`；
- `conflict_flags`、`uncertainty_flags`、`evidence_gaps`；
- `human_review_required` 与原因代码；
- `record_version`，用于原子写回与并发比较。

定位结果通过独立字段保存。D-FINE 执行不得改变分类分数、`predicted_class` 或原始分类制品身份。

### 3.2 CaseState

`agent_state.py` 的 `build_case_state()` 每次从当前 `CaseRecord` 和本轮临时 Observation 派生
`CaseState`。它只包含 Controller 需要的结构化信息：

- 分类、定位、质量、解剖、既往和指南证据状态；
- 内部分类状态与不向主模型/界面公开的数值审计字段；
- 技术状态、不确定性标志和证据缺口；
- 已完成/失败动作；
- 工具预算、当前 disposition 与终止状态。

`CaseState` 不复制原始图像、指南全文、密钥或完整对话，也不独立持久化。

## 4. LangGraph Plan+ReAct 与四个公开工具

主编排由官方 LangGraph `StateGraph` 编译和执行。固定节点为 `load_context`、`plan`、
`decide`、`execute_tool`、`observe`、`replan`和 `finalize`；继续、重规划与终止由条件边决定，
而不是由项目内另一个手写图引擎或 `while` 主循环执行。
`initial_plan` 只包含可展示的 `goal` 和最多四个 `{objective, evidence_need, status}`；它不是固定工具
列表，也不授权批量执行。每次获得 Observation 后，ReAct 节点重新选择下一步或回答；
工具失败且 Plan 修订次数少于两次时才进入 `replan`，重新生成 Plan。成功 Observation 只更新
步骤与条件状态，不调用规划模型。公开 provenance 保留 `replans_after_each_observation=false`，
另用 `decides_after_each_observation=true` 和 `replan_policy` 明确这两种行为。

模型可见的工具严格限定为四个：

```text
classify_cxr             localize_cxr
analyze_lung_anatomy     search_tb_knowledge
```

一个 ReAct step 的 `outcome` 只能是 `tool_call` 或 `answer`；前者必须且只能携带一个公开工具名，
后者必须没有工具。多个需求通过“单动作 -> Observation -> 下一动作”依次完成，不把最初 Plan 当作
一次性完整调用列表。可信运行时绑定原始 `query`、病例身份、权限和预算，模型不能改写这些参数。
主模型使用 Plan/ReAct Schema；`TaskSpec` 仍承担兼容轨迹投影和确定性意图解析。运行时通过
`parse_task_spec()` 校验显式影像需求、增补或移除 Plan 中的证据需求，并识别能力/病例状态等零工具
回答。因此它实际参与证据与工具准入，不能作为纯历史代码删除。旧 `BoundedAgentController` 决策循环
不在生产主路径。指南目标统一为 `SEARCH_TB_KNOWLEDGE`，不再要求主模型输出
`guideline_scope`、`subtopic`、`population`、`product_terms` 或 `scenario_tags`。

`search_tb_knowledge(query)` 映射到内部指南 handler，在工具内建立 `GuidanceQueryProfile`，完成召回、
实体/否定关系提取、章节与人群过滤、rerank、知识快照 attestation 和 claim-scope 校验。解析维度只
写入脱敏 `ToolReceipt.resolved_*`，供审计、记忆和评测使用。

OpenAI-compatible provider 优先走原生 tool calling。原生接口不受支持、调用失败或返回不合约时，
运行时才使用严格 JSON Schema fallback；两条路径共享相同四工具 allowlist、原问题绑定和本地校验。
公开 `model_tool_name` 与内部 handler `tool_name` 同时写入 receipt，便于区分模型选择和执行适配器。
原生响应没有 `tool_calls` 时的正文并不自动可信；它与 JSON fallback 的 `direct_answer`、通用问答正文
共用同一输出校验。封闭的 `<think>/<analysis>` 块可被剥离；残留的 `thought/思考` 前缀、内部
`case_state`/Plan/ReAct 载荷或与问题无关的病例状态会被拒绝，不能直接成为用户消息。

LangGraph `context_schema` 仅注入 service、生成器和 tenant/thread 身份等不可变运行依赖；
可演化 state 只保留脱敏 Plan、有界 Observation、预算、选定动作和终止结果。密钥、原始像素、
完整指南正文和思维链不进入图状态。`graph_node_trace` 只记录实际节点名，供评测和故障定位，
不向前端展示模型的隐藏推理。

上传质控不是模型工具：解码和基础质控在病例创建时完成，询问时直接读取已有状态。普通问答、能力/
病例状态说明，以及缺少可验证纵向模型的比较请求也直接回答，不伪造工具 receipt。其中产品能力说明
使用代码拥有的固定能力事实；“汇总已完成分析”只投影当前公开分类、定位和二维肺野摘要，不把私有
`case_state` 交给模型复述，也不能把“上/中/下肺野”改写成肺叶。规则层只承担权限、
前置条件、预算、最低限度托底和检索证据适用性硬校验；当前证据校验也依赖 `TaskSpec` 的短语规则，
这些规则可能增补或否决模型计划中的动作，需与语义路由一起验证。

`response_projection.py` 是无模型、无工具、无存储副作用的纯投影边界。授权病例读取与急症硬门
先执行，随后 `plan` 为完整的能力/病例状态请求建立确定性零证据计划，`decide` 投影答案，
`finalize` 保留原响应轨迹和线程审计；provider 未配置、离线或正常时都不调用规划与回答模型。
该路径标记 `deterministic_trusted_projection`、`narrator_generation_invoked=false` 和
`narration_status=skipped_response_kind`。含显式分类、定位、肺野或指南需求的复合请求继续进入
正常 Plan/ReAct；状态短语不能使同一请求的其他子句失去工具需求。续问词本身也不授权检索，
只有存在指南任务记忆时才启用指南续问快捷判定。

通用问答在同一 API 进程内按 `owner_scope + user_id + thread_id` 保留最多三组问答，供指代和
短追问使用；它不进入 SQLite、审计记录或病例上下文，最多保留 256 个最近线程，进程重启或
更换 thread 后即失效。持久层仍执行 `tbx-thread-memory-digest-only-v1`，只保存不可逆摘要。

“继续”“展开”等短续接使用线程中最近成功的工具意图和指南任务维度，以及有界的近期问答。
Plan/ReAct 仍会接收这些历史 user/assistant 消息以理解指代；历史消息按不可信数据处理，
当前问题与证据准入规则限制本轮动作。完整聊天记录不持久化或无限累积进提示词。

## 5. 受限 Plan+ReAct Controller

当前主入口为：

```text
TBXAgentGraph.invoke_with_receipt()
  -> TBXAgentService.respond_with_controller()
  -> agent_runtime.run_agent_turn()
```

`TBXAgentGraph` 是为兼容已有 API 保留的类名；它的生产 `invoke_with_receipt()` 会调用已编译的
LangGraph，而不是在适配器中复制状态循环。当前 graph 未配置 durable checkpointer：业务 SQLite/病例存储
仍是病例事实、有限线程记忆与审计的唯一权威来源，因此尚不支持图中间节点的跨进程 resume。

模型只能选择四个公开工具。内部 `AgentAction` 枚举仍负责 handler、权限和历史兼容，例如：

```text
CLASSIFY_CURRENT_CXR
GET_CLASSIFICATION_EVIDENCE
LOCALIZE_CURRENT_CXR
INSPECT_ANATOMICAL_CONTEXT
SEARCH_TB_KNOWLEDGE
STOP
REFER_TO_HUMAN
```

这些内部名称不构成额外模型工具。每次 ReAct 决策最多返回一个工具动作；上传质控、代码拥有的能力
说明、当前病例证据摘要、急症规则提示和未实现的纵向比较由代码直接组成零工具回答。模型输出无效、
越权、包含未公开运行时状态或暴露推理正文时，
运行时按失败类型执行受限恢复：无效选择可被当前计划中的合法动作替代，无效直答可进入独立通用回答，
工具失败可进入 `replan` 边。模型规划失败时也会使用确定性最小计划托底，再执行同一证据准入校验；
并非只有整个 LLM 服务完全不可用时才托底。

急症硬门在任何模型规划前执行，直接建立无工具的转交计划；最终回答也优先保留急症提示，
不被既有病例证据覆盖。工具前置条件、权限、技术失败和预算耗尽由确定性硬门处理，不能由 LLM 放宽。非技术性
分类不确定性只作为回答上下文，不自动触发单病例复核。

## 6. 真实运行循环

```text
读取问题和授权病例 -> 构建 CaseState
  -> PLAN：生成公开 initial_plan（目标/evidence_need）
  -> REACT：选择 answer 或一个公开工具
       |- answer：通过输出校验后组成零工具回答，或使用可信状态投影
       `- tool_call：映射为受限内部 ToolInvocation
  -> ToolRegistry 执行并返回 ToolResult + ToolReceipt
  -> 必要时原子写回 CaseRecord
  -> 重新读取病例并重建 CaseState
  -> 记录 Observation / StateTransition
  -> 必要时生成 PlanRevisionRecord
  -> 下一次 REACT（仍然最多一个工具）
  -> 达成目标、冲突、失败或预算上限后终止
  -> 只合并本轮真实工具 Observation，组成最终回答
```

预算来自 `configs/app.yaml` 或环境变量：

- `max_agent_steps`；
- `max_tool_calls`；
- `max_expensive_vision_calls`；
- `agent_tool_cost_budget`。

到达上限会受控停止，不会无限循环。普通模型、工具、契约或后端失败不会自动重试；
只有容量饱和且 receipt 标记 `retryable=true` 时，预算内恰好再尝试一次，并保留两份真实 receipt。
定位为空不会反复运行 D-FINE 直到出现框。

## 7. Observation 如何改变行为

位置问题的三个典型分支：

```text
LOCALIZE_CURRENT_CXR
  |- COMPLETED + detections
  |    -> 保存候选区域 -> 位置目标完成 -> STOP
  |- COMPLETED_NO_DETECTION
  |    -> 保存“运行成功但无候选”
  |    -> 不生成第二个分类结论，也不改变既有分类
  |    -> 位置目标正常 STOP
  `- FAILED / UNAVAILABLE
       -> 保留技术失败，不转换成空结果
       -> 在对话中说明定位未完成，不改变分类，也不创建复核任务
```

这是状态感知 Agent 与固定 workflow 的关键差异：下一动作由刚产生的 Observation 决定。

## 8. 与 MedRAX 的关系：借鉴范式，不复刻实现

参考 [MedRAX 的 Agent 实现](https://github.com/bowang-lab/MedRAX/blob/main/medrax/agent/agent.py)、
[入口与线程配置](https://github.com/bowang-lab/MedRAX/blob/main/main.py)、
[界面](https://github.com/bowang-lab/MedRAX/blob/main/interface.py) 和
[ChestAgentBench 论文](https://arxiv.org/abs/2502.02673) 后，本项目吸收的是下列可验证的设计原则：

- **工具调用后的 Observation 循环**：MedRAX 以绑定工具的模型调用、`ToolMessage` 和再次调用模型形成
  循环。TBX-Agent 用 LangGraph `StateGraph` 实现“公开 Plan -> 一个受限动作 ->
  `ToolReceipt`/Observation -> 修订或回答”；TBX 自身仅保留病例权限、预算和收据等领域逻辑。
- **按 thread 隔离的短期上下文**：保留当前病例绑定、最近一次可续接意图和最多三组进程内通用问答；
  `owner_scope + user_id + thread_id` 是隔离键。它类似 MedRAX `thread_id` 上下文的使用目的，但不是
  `MemorySaver` 的等价实现，也不是可恢复 checkpoint。
- **展示实际发生的工具事件**：前端只根据真实 receipt 展示已调用的工具及状态，不展示预演计划、
  模型隐藏思维或伪造的“执行完成”步骤。
- **工具契约**：每个动作受固定枚举、参数 Schema、权限、病例身份、预算和输出契约约束；回答只能消费
  已批准事实与 citation。工具名或参数不能由自由文本动态执行。
- **失败后的有限恢复**：工具失败成为下一轮 Observation；互不依赖的目标可以继续，依赖失败证据的分支
  停止。只有明确标为可重试的容量饱和允许预算内重试一次，并同时保留失败与成功 receipt。
- **分维度评测**：当前评测分别记录任务/工具选择、工具契约、恢复、上下文来源、证据忠实度、注入拒绝、
  资源预算、延迟和成本口径，而不只报告一个最终回答分数。

以下做法不采用：用 `eval()` 或同类动态求值解析工具参数；没有退出条件的工具循环；向前端、日志或
评测集输出原始思维链；没有步骤、工具和成本预算的自由 Agent；无稳定 citation 的医学事实扩写。
公开 trace 只保留结构化 reason code、状态哈希、receipt 和终止原因。

这项参考不代表功能对等。当前缺口包括：

- Agent 请求仍是同步整轮返回，尚无逐 token/逐工具事件的 SSE 流；OpenAI-compatible 的底层 LLM
  端点即使支持 SSE，也不等于 Agent 轨迹流。
- 已使用真实 LangGraph 执行单轮图，但没有配置 durable checkpointer；尚不支持进程重启后的图中断点续跑、
  人工 interrupt/resume 或跨 worker 恢复。
- 没有经验证的纵向影像变化模型，不能比较“好转、恶化或稳定”。
- PSPNet 肺野工具依赖用户提供的可选权重，不是所有部署的基础能力。
- 没有临床外部验证、医学专家 qrels 或医疗器械级验证。
- 轨迹套件目前主要是合成场景与选定运行时回放，虽覆盖工具契约、故障和证据一致性，但尚未复现
  MedRAX/ChestAgentBench 的完整病例规模、任务轴、模型对照与专家评审，二者结果不可直接比较。

## 9. 工具语义

### 分类

规划器看到的 `classify_cxr` 映射到内部 `classify_current_cxr` ConvNeXt-Tiny handler。相同病例图像、模型和预处理
版本已有成功证据时由 handler 复用，但本轮仍生成真实工具 receipt；分类解释不自动运行 D-FINE
或 RAG。

### 定位

`localize_cxr` 映射到内部 `localize_current_cxr`，按需运行 D-FINE。`localization_evidence.status` 区分：

- `not_requested`；
- `completed`；
- `completed_no_detection`；
- `failed`；
- `unsupported` / `stale`。

成功写回使用病例版本比较和内容身份，避免同一定位请求在并发下重复覆盖。失败尝试不得覆盖同一
generation 已成功的证据。

### 解剖

`analyze_lung_anatomy` 映射到内部 `inspect_anatomical_context`，按需运行 PSPNet 并计算候选框与肺野的二维空间关系。请求候选区域肺区
而定位未运行时，Controller 可先定位；只要求显示肺野时不必运行 D-FINE。当前不是肺叶分割。

### 图像质量

上传阶段已生成基础质控证据；回答质量问题时直接读取该状态，不产生工具调用。它不是完整放射摄影
质量评价，也不应被包装成第五个模型工具。

### 既往影像

当前没有经过验证的纵向变化模型，也没有公开纵向比较工具。无既往片直接回答
`PRIOR_IMAGE_NOT_FOUND`；即使用户提供两张片，也必须说明 `LONGITUDINAL_MODEL_UNAVAILABLE`，不得
生成“好转/恶化/稳定”。

### 指南

ReAct 只决定是否需要 `search_tb_knowledge(query)`，不选择 scope、检索实体或
BM25/Dense/Hybrid。问题解析和证据适用性校验在工具内部完成；具体检索模式由 RAG 配置控制。
工具回答必须带稳定 chunk citation；无满足准入与相关性条件的证据时返回 evidence gap。

## 10. 一致性、证据缺口和人工复核

确定性 post-action 检查至少识别：

- 定位/分类/质量工具失败；
- 分类 margin 过低或 argmax 并列（仅作内部 uncertainty）；
- 既往片不存在或纵向能力不可用；
- 指南证据/索引不可用；
- 工具预算耗尽。

分类与 D-FINE 有无候选框的差异不是冲突。上述状态用于解释与故障恢复，不改变视觉模型原始输出，
也不创建交互病例复核任务。复核工作台只接受显式批量筛查入口创建的任务。

## 11. RAG

统一 corpus 由 `knowledge/source_manifest.json` 与 `knowledge/chunks.jsonl` 构成。稳定 chunk ID、
来源、版本/日期、地区、主题、claim scope、章节/locator、正文与内容哈希是 BM25、Dense、Hybrid
共同事实源。

- BM25：依赖少、确定性排序，是已接入基线；
- Dense：BGE-M3 dense vector，可选懒加载/显式本地服务，索引身份绑定模型和 corpus；
- Hybrid：分别召回 BM25/Dense 候选，以 RRF 合并，不直接相加原始分数；
- Qdrant Local：单机持久化向量索引，不要求启动 Qdrant Server；
- fallback：若配置允许，Dense 不可用可显式回退 BM25，回执必须记录实际模式。

启用 Dense/Hybrid 前必须构建匹配 manifest，并在同一固定 qrels 上运行检索评测。当前 checked-in
BM25 工程 fixture 不是医学相关性金标准，也不能证明 Hybrid 更优。

## 12. 回答、上下文与审计

同一个选定的 MedGemma/OpenAI-compatible 连接可承担公开 Plan、单步 ReAct 选择和回答表述。
ReAct 选择优先走原生 tool calling，并以严格 JSON 作为兼容 fallback；各环节仍有独立 Schema 与
本地校验边界。回答合并器读取本轮实际工具 response、citation，以及当前请求明确需要且已经完成的
病例缓存证据。缓存分类、定位和二维肺野摘要可直接进行确定性投影，不运行模型、不增加事实，
也不生成伪造的工具 receipt；新增证据仍必须通过 ToolRegistry 获取。
narrator 只接收最小 approved-fact 视图，不能决定工具结果或改写 predicted class。

Plan/ReAct 会将当前问题、病例状态投影、本轮有界 Observation 与同一线程的近期对话发送给选定的
provider，普通问答也可能先经过这些阶段；使用远程 provider 时这些文本可能离开本机。

`GENERAL_CHAT` 使用独立 Schema `{answer: string}`，直接调用本轮选定 provider；它不调用视觉、
RAG 或病例工具，也不生成 receipt。请求只包含当前问题与同一
`owner_scope + user_id + thread_id` 下最近三组进程内问答；不另外附加病例结构、图像、指南证据、工具回执或
其他线程内容。历史问答可能包括此前的病例或指南回答，不能据此宣称整个请求完全不发送病例信息。
模型不可用、输出不合法、暴露思考/内部状态、给出个体化药物指令或谎称已执行工具时，返回明确的通用模型
故障，不退回病例状态。这份有界短期上下文只存在于 API 进程内，不进入 SQLite、审计日志或病例记忆。

线程只保存有限摘要事件、当前病例绑定、最近一次成功且可续接的语义意图和工具调用计数；工具回答
与缓存回答都会按白名单更新该意图，以便“继续”“展开”等短追问只续接刚完成的任务。线程不保存
原始图像、完整提示词、API key 或模型思维链。公开轨迹包含：

- `initial_plan`：公开 goal、objective、evidence need 和状态；
- `plan_revisions`：修订号、触发码、原因码及前后 Plan 哈希；
- `react_steps`：每步的公开工具名或 answer、选择模式、状态和 observation code；
- receipt 中并列的 `model_tool_name` 与内部 `tool_name`；
- 工具状态、错误码和证据缺口；
- 预算使用；
- 终止原因与最终 `STOP`。

兼容 `AgentRunTrace` 的 `decisions` 和 `state_transitions` 在当前 Plan/ReAct 主路径中为空，
不应将旧 Controller 的状态迁移哈希当成本轮已记录数据。实际执行过程以 `react_steps`、
`graph_node_trace` 和真实工具 receipt 为准，Plan 哈希与最终回答哈希进入审计事件。

`hidden_reasoning_persisted` 固定为 `false`。该字段描述图状态和公开轨迹不持久化自由推理，并不假设
provider 会遵守“不要输出思考”的提示；provider 正文仍须通过上述清洗、拒绝和回退边界。

## 13. 前端、主动筛查与批量

Streamlit 左侧是对话、简短公开 Plan 和实际工具链，右侧显示当前胸片及按需图层。Plan 步骤是目标，
不能渲染成已执行工具；只有真实 receipt 才显示成功/失败。定位或肺野工具完成后重新读取病例证据。
默认不显示肺野图层。

主动筛查是确定性问卷，可绑定当前胸片病例，但问卷答案与 ConvNeXt 分数不融合成未经验证的疾病
概率。批量筛查默认只分类；定位、PSPNet 和 RAG 只在打开单病例并明确请求时运行。

## 14. 并发与持久化

当前是单服务进程原型：

- 同一 owner/user/thread 的 Agent turn 使用进程内锁串行；
- 病例和复核使用 SQLite；
- 定位写回使用事务与 `record_version` 比较；
- Qdrant Local 索引由离线构建器写，应用侧读取；
- 不实现分布式锁、跨组件事务、Controller 执行恢复或自动后台索引重建。

Windows 与 Linux 共用 Python 核心；路径由项目相对位置、环境变量或命令行参数提供，不依赖开发者
电脑绝对路径。

## 15. 当前限制

- 视觉权重通过独立推理包安装，不随源码分发；下载入口和访问范围以发布说明为准；
- softmax 未做临床概率校准；
- 无经过验证的纵向比较能力；
- PSPNet 是可选二维肺野工具；MedSAM 适配器默认关闭且不属于当前闭环；
- Dense/Hybrid 的可用性取决于实际模型、索引和同 corpus 评测；
- 没有医学专家 qrels 或临床外部验证；
- Agent 层无 SSE 事件流；Plan+ReAct 由 LangGraph 执行，但未配置 durable checkpointer，不能跨进程断点续跑；
- 当前轨迹评测不等同于 MedRAX/ChestAgentBench 的完整外部基准；
- 没有生产 IAM、分布式恢复、在线学习、RL Controller 或医疗器械准入。
