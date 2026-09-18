# TBX-Agent V1 HTTP API

> 本文描述当前 `0.1.0` FastAPI 源码，而不是规划接口。V1 未经临床验证、未达到临床生产就绪，
> 不用于确诊或排除肺结核，也不提供个体化处方。应用已有 fail-closed production HMAC 身份档，
> 但没有 RBAC，不能据此宣称完整生产授权。

## 1. 基本约定

- 默认地址：`http://127.0.0.1:8000`
- OpenAPI：research/development 可用 `GET /openapi.json`
- Swagger UI / ReDoc：research/development 可用 `GET /docs` / `GET /redoc`；production 三者关闭
- 除胸片上传外，请求和响应使用 `application/json`。
- 胸片上传使用 `multipart/form-data`。
- Pydantic 请求模型禁止额外字段（`extra="forbid"`）。路径和 query 参数的额外项仍遵循 FastAPI 默认行为。
- 时间为 ISO 8601 UTC 字符串。
- 所有医学输出都是信息支持；HTTP 200 不表示医学阴性、确诊、临床批准或人工复核通过。
- checked-in 默认运行契约是 `rank03 + llama_cpp`，且
  `require_real_inference=true`、`require_llm_inference=true`。必需视觉或 MedGemma 生成失败返回 503，
  不会生成 mock、静态模板或确定性替代回答。

### 1.1 身份、subject 与 RBAC 边界

`research/development` 兼容本地演示：客户端提交 `owner_scope`、`user_id`、`reviewer_id` 或
`actor_id`。这些值未经认证，即使存储层做 exact tenant + subject 比较也不能暴露到不受信网络。

`production` 在应用构造时 fail-close，要求可信反向代理提供 HMAC-SHA256 签名的
`X-TBX-Tenant`、`X-TBX-User`、`X-TBX-Actor`、method、完整 path/query、timestamp 与 nonce。
应用派生 `owner_scope=tenant:<X-TBX-Tenant>`；`X-TBX-User` 是病例资源 subject，
`X-TBX-Actor` 是操作者。body/form/query 中的身份字段可省略，若提供则必须与签名一致，否则 403。
缺失、错误、过期或重放签名为 401。完整契约见 `docs/production_security.md`。

当前签名没有 role/permission claim。review 列表/读取/完成和报告操作均固定
`allow_cross_subject=false`；actor 不因名字是 clinician/reviewer 就能访问其他 subject。因此当前
production 档有真实身份完整性和 same-subject 隔离，却仍是跨 subject 临床复核的 NO-GO，不能称为
完整 RBAC。

## 2. 通用枚举与响应对象

### 2.1 `AgentResponse`

Agent、胸片和主动筛查响应使用同一个结构：

| 字段 | 类型 | 说明 |
|---|---|---|
| `request_id` | string | 本次业务请求 UUID；只读筛查 GET 固定为 `read-only` |
| `trace_id` | string | 本次业务 trace UUID；只读筛查 GET 固定为 `read-only` |
| `thread_id` | string | 对话/问卷线程；直接胸片评估固定为 `assessment` |
| `case_id` | string/null | 关联病例；不是诊断标识 |
| `response_kind` | enum | 见下表 |
| `summary` | string | 面向用户的主摘要；视觉结果可在主对话中压缩为一句准确模型结论 |
| `visual_result` | enum/null | 仅视觉相关响应使用 |
| `predicted_class` | enum/null | 分类器原生 argmax 训练类：`healthy`、`sick_non_tb`、`tb`；无可靠证据时为 null；并列按固定类别顺序处理；不是诊断 |
| `visual_evidence_notes` | string[] | exact-case 解释可附带的代码生成空间事实；仅视觉响应使用，MedGemma 只能从中选择，不能改写定位 |
| `diagnostic_information` | string[] | 诊断证据链教育，不是诊断结论 |
| `next_step_information` | string[] | 进一步检查、筛查或就医信息，不是检查单 |
| `treatment_education` | string[] | 非处方性教育，不含个体方案 |
| `limitations` | string[] | 后端完整返回的输出边界和限制；客户端可分层披露，但病例详情必须持续可见且不得丢弃 |
| `citations` | `Citation[]` | 结构化知识来源；本地急症规则故意不借指南 citation |
| `review_status` | enum/null | `not_required`、`pending` 或 `completed` |
| `next_question` | `ScreeningQuestion`/null | 主动筛查收集阶段的下一题 |
| `urgency` | enum/null | `emergency`、`prompt_evaluation`、`priority_screening`、`routine_information` |
| `reused_existing_assessment` | boolean | 当前调用未生成新视觉证据；病例解释路径也会为 `true` |
| `safety_policy_id` | string | 当前为 `tbx-agent-safety-v1` |

`response_kind` 枚举定义了：

```text
general_answer
visual_screening_result
localization_result
diagnostic_information
next_test_information
treatment_education
active_screening_question
active_screening_summary
screening_report
case_explanation
capability_statement
emergency_escalation
safe_abstention
```

客户端应按返回的 `response_kind` 渲染：分类结论为 `visual_screening_result`，候选区域为
`localization_result`，病例状态或缓存证据解释可为 `case_explanation`，纯系统能力说明为
`capability_statement`，急症升级为 `emergency_escalation`。`general_answer` 是普通问答，不代表
病例证据。复合问题可在同一响应中保留病例事实和指南引用；不要仅凭类型丢弃其他非空字段。
报告端点返回制品元数据；枚举值不意味着存在同名路由。

`visual_result` 枚举：

```text
model_flagged
model_not_flagged
non_tb_abnormal
indeterminate
technical_failure
pending_human_review
```

这些值都是辅助筛查/工作流状态，不是肺结核诊断结果。

视觉结果采用分层披露。主对话应优先显示一句可核对的模型结论；例如
`model_not_flagged` 可表述为“模型没有识别出结核”，`non_tb_abnormal` 可表述为“模型识别为非结核
异常”。用户侧只显示分类结论；原始分类分数只保留在内部审计与离线评测链路，不通过
公共病例接口、病例详情或用户报告展示。
完整的非诊断边界仍由 `limitations` 返回，并在病例详情中持续可见。客户端不得因主对话简化而删除、
覆盖或停止保存 `limitations`。

### 2.2 `Citation`

```json
{
  "chunk_id": "who-m3-rapid-diagnostic-001",
  "source_id": "who_tb_diagnosis_module3_2025",
  "title": "WHO consolidated guidelines on tuberculosis: Module 3: Diagnosis, fourth edition",
  "organization": "World Health Organization",
  "publication_year": 2025,
  "section": "示例章节",
  "locator": "示例定位符",
  "url": "https://www.who.int/publications/i/item/9789240107984",
  "support_text": "经审核的单一主张支持文本"
}
```

citation 表示回答使用了哪个审核片段，不表示该片段已经证明适用于当前个体。

### 2.3 `CaseRecord`

| 字段 | 类型 | 说明 |
|---|---|---|
| `case_id` | string | UUID |
| `owner_scope` | string | tenant scope；research/dev 自报，production 由签名 tenant 派生 |
| `user_id` | string/null | subject 绑定；新记录必须非空，无法证明 subject 的 legacy 行 fail closed |
| `image_artifact_ref` | server-only | 内部模型字段；所有 API/公开 JSON 响应递归删除，不返回本地路径 |
| `image_sha256` | string | 上传原始字节 SHA256 |
| `image_width`, `image_height` | integer | 解码尺寸 |
| `consent_scope` | string | 当前为 `cxr_auxiliary_screening` |
| `vision_evidence` | object/null | 分类器 argmax 与 advisory 检测候选框的结构化证据 |
| `fusion_decision` | object/null | 冻结融合结果；`clinical_validation` 固定 false |
| `review_id` | string/null | 待/已复核记录 ID |
| `review_status` | enum | `not_required`、`pending`、`completed` |
| `active_screening_session_id` | string/null | schema 预留；当前服务未把新问卷 ID 回写病例 |
| `created_at`, `updated_at` | datetime | UTC 时间 |
| `record_version` | integer | 病例版本；复核完成时递增 |

公共 `vision_evidence` 包含运行 ID、图像哈希/尺寸/质量状态及稳定 `image_quality_codes`、分类和检测模型 ID、checkpoint SHA256、
`classifier_decision_rule`、`predicted_class`、`classifier_argmax_tied`、
`detector_decision_role`、候选框（与 D-FINE 300 object-query 合同一致，最多 300 个）、预处理/策略版本、
运行毫秒数和 artifact refs。原始类别顺序、softmax、top-1/top-2、margin 与分类阈值只保存在
服务端审计记录中，不出现在公共病例响应或用户报告。当前规则是 `native_three_class_argmax`；
检测器角色是 `advisory_localization_only`。`tb`、`healthy`、
`sick_non_tb` 分别路由为模型标记、模型未标记和 `non_tb_abnormal`；完全并列路由为
`pending_human_review`。非技术性图像质量 warning 保留在 `image_quality_status` 和
`image_quality_codes` 中供解释与审计，但不覆盖 argmax 分流，也不让单病例进入复核；技术失败仍
返回 `technical_failure`。交互评估仍不创建中央复核记录。

`fusion_decision` 回显 `predicted_class`、分类规则、检测器角色，并包含 `policy_id`、
`visual_result`、`review_required`、`review_reasons`、最大检测分数及固定为 false 的
`clinical_validation`。D-FINE 候选框和检测分数只用于定位，不覆盖分类器 argmax 分流；无框也不能排除肺结核。

病例精确复用键是 `(owner_scope, user_id, image_sha256)`，不是仅 owner + hash。不同 subject 即使上传
相同字节也不共享病例。旧数据库中无法从既有 payload 证明 subject 的行保留为隔离状态，API 不会
把它自动归给当前调用者。

### 2.4 `ActiveScreeningSession`

| 字段 | 类型 | 说明 |
|---|---|---|
| `session_id` | string | `screen-` 前缀 ID |
| `thread_id`, `user_id`, `owner_scope` | string | tenant + subject 绑定；production 来自签名身份 |
| `case_id` | string/null | 可选关联病例 |
| `consent` | boolean | 当前问卷同意状态 |
| `guideline_rule_version` | string | 当前问题库规则版本 |
| `status` | enum | `consent_pending`、`collecting`、`complete`、`cancelled`、`needs_clarification` |
| `answers` | object | 已规范化答案；可能含 `__unknown__` 或 `__skipped__` |
| `next_question_id` | string/null | 当前必须回答的问题 |
| `result` | object/null | 完成时的 `urgency`、`triggers`、`information_gaps`、`next_steps`、`citations` |
| `created_at`, `updated_at` | datetime | UTC 时间 |

启动 HTTP 请求的 `consent` 是必填 boolean，因此 API 当前不会创建 `consent_pending`；`false` 会得到 `cancelled`。其余状态值属于内部 schema/未来路径，不应据此假设已有对应 API 动作。

### 2.5 `ReviewRecord`

```json
{
  "review_id": "review-uuid",
  "case_id": "case-uuid",
  "owner_scope": "local-demo:user-1",
  "trigger_reasons": ["batch_model_flagged"],
  "status": "pending",
  "reviewer_decision": null,
  "reviewer_note": null,
  "reviewed_by": null,
  "reviewed_at": null,
  "version": 1,
  "created_at": "2026-08-28T10:00:00Z"
}
```

允许的 `reviewer_decision`：

```text
keep_model_flagged
keep_model_not_flagged
indeterminate
technical_repeat_required
```

复核完成不会替换病例中的 `fusion_decision.visual_result`。completed review 可按 exact ID 读取，报告会把它作为独立记录附加；调用方必须分别呈现原始模型融合结果和人工复核记录，不能把后者伪装成对模型证据的覆盖。

## 3. 错误模型

手工抛出的 HTTP 错误通常是：

```json
{"detail": "错误说明"}
```

FastAPI/Pydantic 请求校验错误的 `detail` 是结构化数组。当前稳定语义如下：

| HTTP | 含义 | 典型场景 |
|---|---|---|
| 200 | 请求按当前契约完成 | 仍不代表医学阴性、确诊或临床批准；默认强制档的视觉/LLM 失败不会返回 200 |
| 401 | production 可信代理认证失败 | 签名缺失/错误/过期、重复 header 或 nonce 重放；响应不泄露 claim/secret |
| 403 | 身份声明冲突或 tenant/subject 范围不匹配 | 签名身份与 body/query/form 冲突，或 exact subject 授权失败；筛查业务异常映射仍有例外 |
| 404 | 记录不存在 | 病例、复核、问卷 session 或可选关联病例不存在 |
| 409 | 版本/状态冲突 | 复核已不再 pending、乐观锁竞争，或 legacy assessment 需要显式 generation migration |
| 411/413 | 请求 framing/大小不允许 | production body 请求无明确 Content-Length，或完整 body 超限 |
| 429 | 进程内主体速率限制 | 带 `Retry-After`；多副本前仍需共享网关限流 |
| 422 | 请求字段、同意或图像无效；筛查状态/答案无效 | 多余字段、长度/类型、不同意、不支持的图像/DICOM、过大/无法解码、乱序答题、空多选等 |
| 503 | 必需视觉/LLM 推理失败、进程内并发背压或 readiness 未通过 | `required_vision_inference_failed`、`required_llm_inference_failed` 或稳定非敏感 blocker；无模拟/静态回退 |
| 500 | 未映射的初始化、依赖或安全异常 | 例如知识文件缺失或未映射安全校验异常 |

当前有一个必须由客户端特别处理的不一致：

1. `POST .../answers` 和 `POST .../cancel` 把 `PermissionError` 一并转换为 422，所以这两个端点的 owner 或 user 不匹配可能返回 422，而不是其他端点常见的 403。
除 migration conflict 外，多数业务错误仍缺统一机器可读 `error_code`；生产前应避免通过 403/404
差异泄露对象是否存在。每个响应带验证/生成的 `X-Request-ID` 和 no-store/nosniff 等安全 header。

## 4. 端点总览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/livez` | 仅进程存活，不初始化可选依赖 |
| GET | `/readyz` | 实际加载必需 rank03 runtime，并执行 MedGemma 结构化 generation probe |
| GET | `/healthz` | 后向兼容的聚合健康摘要 |
| GET | `/metrics` | 默认关闭；仅 loopback 或管理员 bearer token 的去标识化进程指标 |
| GET | `/v1/system/capabilities` | 非敏感、非阻塞的组件/工具状态快照 |
| GET | `/v1/system/manifest` | 模型、策略、split、知识快照与来源决策 |
| POST | `/v1/assessments/cxr` | 上传胸片并运行辅助筛查 |
| POST | `/v1/batches/{batch_id}/assessments/cxr` | 批量筛查一项，并把非 healthy 直接路由幂等登记到复核工作台 |
| GET | `/v1/cases/{case_id}` | exact ID + tenant + subject 读取病例 |
| POST | `/v1/cases/{case_id}/anatomy-runs` | 创建可选肺野分割与空间关系 worker run |
| GET | `/v1/cases/{case_id}/anatomy-runs` | 列出 exact subject 下该病例的 anatomy runs |
| GET | `/v1/cases/{case_id}/anatomy-runs/{run_id}` | 读取 run、肺野 QC、结构化空间关系与安全摘要 |
| GET | `/v1/cases/{case_id}/anatomy-runs/{run_id}/boundary.png` | 获取源坐标透明肺野边界图层 |
| POST | `/v1/llm/connections` | 测试并创建线程绑定、进程内短期保存的 OpenAI-compatible 连接 |
| DELETE | `/v1/llm/connections/{connection_id}` | 撤销当前 tenant + subject + thread 的临时 LLM 连接 |
| POST | `/v1/agent/respond` | 对话式结果解释、检查信息、治疗教育或急症升级 |
| POST | `/v1/screening/sessions` | 显式同意/拒绝主动筛查 |
| POST | `/v1/screening/sessions/{session_id}/answers` | 提交当前问题答案 |
| POST | `/v1/screening/sessions/{session_id}/cancel` | 取消并清空本次答案 |
| GET | `/v1/screening/sessions/{session_id}` | 读取问卷状态和可渲染响应 |
| GET | `/v1/reviews/pending` | 列出 signed/local subject 自己的待复核记录 |
| GET | `/v1/reviews/{review_id}` | exact ID + tenant + subject 读取复核记录 |
| PATCH | `/v1/reviews/{review_id}` | 用乐观锁完成复核 |
| POST | `/v1/cases/{case_id}/reports` | 生成 Markdown/JSON 报告制品 |
| GET | `/v1/cases/{case_id}/reports/{report_id}/{artifact_format}` | tenant + subject 校验后下载报告 |

## 5. 系统端点

### `GET /livez`、`GET /readyz` 与 `GET /metrics`

`/livez` 只返回进程存活。默认强制档的 `/readyz` 会实际加载 rank03 分类器、检测器和 D-FINE
运行时，并执行一次有界、非医学、严格 JSON Schema 的 MedGemma generation probe。成功示意：

```json
{
  "status": "ready",
  "service": "TBX-Agent",
  "version": "0.1.0",
  "deployment_profile": "research",
  "blockers": [],
  "runtime_verified": true,
  "llm_generation_probed": true,
  "optional_narrator_probed": true,
  "clinical_validation": false
}
```

`optional_narrator_probed` 是兼容字段；它现在与实际结构化生成探针结果一致，不能再解释为“语言层
未探测”。无效 production HMAC 配置通常已使 `create_app` 直接失败；运行时加载或生成探针失败使
readiness 返回 503 和稳定 `blockers`。探针不对患者图像执行推理，也不证明筛查性能或临床有效性。

`/metrics` 默认 404。启用后只允许直接 loopback 客户端，或正确的管理员 bearer token；输出只有
route template、状态类、聚合延迟、guard 拒绝、fallback 和 in-flight 数，不含 raw URL、病例、
tenant/user/actor、prompt、图像或 secret。该状态是单进程的，不能替代多副本网关指标。

### `GET /healthz`

响应：

```json
{
  "status": "ok",
  "service": "TBX-Agent",
  "version": "0.1.0",
  "vision_backend": "rank03",
  "narrator_backend": "llama_cpp",
  "mode": "real_rank03_with_local_llm",
  "deployment_profile": "research",
  "required_components_ready": true,
  "clinical_validation": false
}
```

该端点会触发 service 初始化和 capability 构建，因此默认强制档也会加载 rank03 runtime 并执行缓存
受限的 MedGemma generation probe。它不是轻量 liveness；探针成功仍不代表患者图像推理成功或临床可用。

### `GET /v1/system/capabilities`

无请求参数。返回 `status`、`mode`、`runtime_verified`、`checked_at` 和 `components`。每个组件包含
`component_id`、`state`、`required`、`implementation`、`detail`、`loaded` 和 `synthetic`。
当前快照覆盖默认必需的 rank03 影像评估和 llama.cpp composer、离线指南检索、主动筛查状态机、
SQLite 存储，以及四个公开 allowlist Agent 工具。工具条目的 `detail` 显示截止时间和步数上限；
工具状态可以因最近失败从 `ready` 降为 `degraded`。

默认强制档会调用 rank03 `probe_runtime()` 和 MedGemma `probe_generation()`；只有真实分类器/检测器已
加载、固定 llama.cpp alias/provenance 与结构化生成均成功，且其他必需组件 ready 时，
`runtime_verified=true`。`mode` 为 `real_rank03_with_local_llm`。隔离测试档可以显式出现
`synthetic_demo` 或非必需语言层，但不能冒充默认真实契约。`status=ok`/`runtime_verified=true` 是
工程运行时证据，不是患者影像推理、医学正确性或临床有效性证据。

### `GET /v1/system/manifest`

无请求参数。响应字段：

```json
{
  "vision_backend": "rank03",
  "inference_contract": "real_required",
  "runtime_verified": true,
  "inference_mode": "real_runtime_verified",
  "model_bundle_id": "tbx11k_user_rank03_<bundle-id>",
  "model_source_revision": "source-revision",
  "classifier_checkpoint_sha256": "sha256",
  "detector_checkpoint_sha256": "sha256",
  "fusion_policy_id": "rank03-user-trained-native-argmax-v2",
  "classifier_rule": "native_three_class_argmax",
  "classifier_routes": {
    "healthy": "model_not_flagged",
    "sick_non_tb": "non_tb_abnormal",
    "tb": "model_flagged"
  },
  "detector_role": "advisory_localization_only",
  "input_contract": {
    "max_upload_bytes": 20971520,
    "raster_formats": ["PNG", "JPEG"],
    "dicom": {
      "enabled": true,
      "part10_preamble_required": true,
      "modalities": ["CR", "DX"],
      "single_frame": true,
      "monochrome_only": true,
      "compressed_transfer_syntax": false,
      "raw_dicom_persisted_as_case_artifact": false,
      "persisted_derivative": "metadata_free_png",
      "input_transform_id": "dicom-crdx-windowed-rgb-v1"
    },
    "raster_input_transform_id": "raster-exif-transpose-rgb-v1",
    "quality_contract": "technical_and_coarse_domain_checks_only"
  },
  "policy_selection_design": null,
  "policy_selection_metrics": null,
  "policy_heldout_metrics": null,
  "reference_split_sha256": null,
  "selection_split_sha256": null,
  "knowledge_snapshot_id": "tbx-guidelines-curated-2026-09-01-v6",
  "knowledge_manifest_sha256": "sha256",
  "knowledge_chunks_sha256": "sha256",
  "narrator": {
    "backend": "llama_cpp",
    "model": "tbx-medgemma-1.5-4b-it-q4-k-m",
    "expected_model_digest": "sha256",
    "policy_id": "tbx-grounded-evidence-synthesis-v2",
    "local_only_client_policy": true
  },
  "source_decisions": [],
  "clinical_validation": false,
  "warning": "The repository contains inference code only, with no model weights or clinical performance claim. Install the independently distributed, hash-verified inference bundle and validate its runtime identity before use."
}
```

活动策略没有 checkpoint 无关的筛查阈值，因此 selection/heldout 指标和参考 split 字段为 null。
视觉包通过配置和权重哈希绑定模型身份；本源码发布不含训练记录或性能背书，
不同权重不能继承其他模型的性能声明。
`source_decisions` 是完整 manifest 来源对象数组，
包含状态、允许 claim scope、是否获得/审核全文和纳入/排除理由。`inference_contract=real_required`
表示本进程同时要求 rank03 与 llama.cpp；`runtime_verified` 是本次 capability 探针结果。
`narrator` 只公开非敏感配置身份，不返回模型服务 endpoint 或 API key。默认选择是 direct pinned
llama.cpp + MedGemma 1.5 4B；Ollama 与 Qwen 仅历史对照。

## 6. 胸片与病例

### `POST /v1/assessments/cxr`

内容类型：`multipart/form-data`

| part | 类型/限制 | 说明 |
|---|---|---|
| `file` | binary：PNG/JPEG，或带 Part-10 前导的无压缩单帧单色 CR/DX DICOM | 必须是已去标识化胸部 X 线；DICOM 原始字节/标签不持久化为病例制品，只保留确定性 PNG 衍生图。HTTP 层可能使用受控临时文件；压缩、多帧、其他模态及明确烧录标注会被拒绝 |
| `user_id` | string，1..128 | research/dev 必填的自报 subject；production 可省略并由签名 user 派生 |
| `owner_scope` | string，1..256 | research/dev 必填的自报 scope；production 可省略并由签名 tenant 派生 |
| `consent_to_process` | boolean | 必须为 true |
| `attested_chest_radiograph` | boolean | 必须为 true；只是调用方声明，不是系统影像识别 |

默认最大上传 20 MiB；服务只读取 `max_upload_bytes + 1` 后验证。示例：

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/assessments/cxr `
  -F "file=@<deidentified-cxr.png>" `
  -F "user_id=local-user" `
  -F "owner_scope=local-demo:local-user" `
  -F "consent_to_process=true" `
  -F "attested_chest_radiograph=true"
```

成功响应：

```json
{
  "case": {
    "case_id": "case-uuid",
    "owner_scope": "local-demo:local-user",
    "user_id": "local-user",
    "image_sha256": "sha256",
    "image_width": 512,
    "image_height": 512,
    "consent_scope": "cxr_auxiliary_screening",
    "vision_evidence": {},
    "fusion_decision": {},
    "review_id": null,
    "review_status": "not_required",
    "active_screening_session_id": null,
    "created_at": "2026-08-28T10:00:00Z",
    "updated_at": "2026-08-28T10:00:00Z",
    "record_version": 1
  },
  "response": {
    "request_id": "request-uuid",
    "trace_id": "trace-uuid",
    "thread_id": "assessment",
    "case_id": "case-uuid",
    "response_kind": "visual_screening_result",
    "summary": "【演示后端】模型识别为结核类，建议进一步检查。",
    "visual_result": "model_flagged",
    "predicted_class": "tb",
    "diagnostic_information": [],
    "next_step_information": ["..."],
    "treatment_education": [],
    "limitations": ["辅助筛查，不用于确诊或排除肺结核。"],
    "citations": [{
      "chunk_id": "guideline-chunk-id",
      "source_id": "approved-source-id",
      "title": "经审核的指南标题",
      "organization": "发布机构",
      "publication_year": 2025,
      "section": "章节",
      "locator": "定位符",
      "url": "https://example.org/official-guideline",
      "support_text": "经审核的支持文本"
    }],
    "review_status": "not_required",
    "next_question": null,
    "urgency": null,
    "reused_existing_assessment": false,
    "safety_policy_id": "tbx-agent-safety-v1"
  }
}
```

示例省略的 `{}`/`[]` 内部字段由第 2 节 schema 定义；实际响应不会省略默认字段。视觉结果取决于后端，不能依赖示例值。

相同 `owner_scope + user_id + image_sha256` 只有在上传制品完整、策略 ID 一致、后端类型和 checkpoint 哈希匹配时才复用，并把 `reused_existing_assessment` 设为 true。系统不做跨 tenant/subject 复用，不做相似图像/病例检索。公开响应不含 `image_artifact_ref`。

错误：422（同意/声明为 false、空文件、过大、格式不支持、解码失败或 schema 错误）；默认强制档
的推理期 `VisionBackendError` 返回 503 `required_vision_inference_failed`，不保存或返回 mock、静态或
`technical_failure` 替代结果。

### `POST /v1/batches/{batch_id}/assessments/cxr`

multipart 字段与交互评估相同，并增加必填 `batch_item_id`。服务先运行或复用同一真实推理，再以
`owner_scope + batch_id + case_id` 生成稳定复核身份：healthy 的 `model_not_flagged` 不登记；其他
结果登记 `origin=batch_screening`、`batch_id`、`batch_item_id` 的 ReviewRecord。重试同一项是幂等的，
不会因病例先在交互模式创建而漏登。响应包含 `case`、`response`、可能为 null 的 `review` 及 batch
标识。中央复核工作台只消费该端点产生的 batch provenance；普通 `/v1/assessments/cxr` 始终返回
病例自身 `review_status=not_required`。

### `GET /v1/cases/{case_id}`

research/dev Query：`owner_scope` 与 `user_id` 均必填、非空。production 由签名派生，两者可省略；若
提供必须一致。

成功返回去除 server-only artifact ref 的 `CaseRecord`。不存在为 404；tenant 或 subject 不匹配为
403。该端点不会运行模型，也不会读取相似病例。

### `POST /v1/cases/{case_id}/anatomy-runs`

为已完成胸片病例创建可选、异步的肺野分割 run。query 必须提供与病例一致的 `owner_scope` 和 `user_id`；未保留原图时，multipart `file` 必须重新上传完全相同的输入。新任务返回 202；命中相同有效 generation 时返回 200 且 `reused_existing_run=true`。可选工具失败不会改变病例的 `fusion_decision`。

### `GET /v1/cases/{case_id}/anatomy-runs`

按更新时间倒序列出 exact tenant + subject + case 下最多 100 条 run。`limit` 默认为 20。不会跨 subject 返回记录。

### `GET /v1/cases/{case_id}/anatomy-runs/{run_id}`

状态为 `pending`、`running`、`completed`、`completed_with_refinement_failure` 或
`technical_failure`。`completed_with_refinement_failure` 表示 PSPNet 肺野与确定性空间证据已成功，
但可选 MedSAM 子分支失败；它不是整个 anatomy run 失败。证据可用的两种完成记录都包含左右肺野
RLE、分割 QC、与每个 D-FINE 框一一对应的 `detector_locations`、定位策略 provenance 和
`spatial_summary`。配置 MedSAM 时还会返回 `refinement_status`、稳定的
`refinement_error_code` 或完整 `refinement_evidence`。`localized` 只表示像素几何交叠；
`outside_lungs` 不等于肺外病变；`invalid_anatomy` 明确停止空间结论。

### `GET /v1/cases/{case_id}/anatomy-runs/{run_id}/boundary.png`

允许 `completed` 和 `completed_with_refinement_failure` run。`structure` 为 `combined`、
`left_lung` 或 `right_lung`，返回与源图同宽高的透明 PNG。响应头
`X-Anatomy-Routing-Effect: none` 明确该图层不参与 rank03 分流。

### `GET /v1/cases/{case_id}/anatomy-runs/{run_id}/contours.png`

仅当 `refinement_status=completed` 且存在经契约校验的 MedSAM 证据时返回与源图同宽高的透明
PNG；未启用、子分支失败或证据不可用返回 409。响应头
`X-Refinement-Routing-Effect: none` 和 `X-Clinical-Validation: false` 明确该图层只是未经验证的
可视化轮廓。结构化 `refinement_evidence.items` 与 D-FINE 框一一对应；超出受审单次提示上限的框
保留为 `capacity_abstained`，不会静默丢失或生成 mask。完整数据与安全边界见
[肺野分割与确定性空间证据](anatomy_spatial_evidence.md)和
[可选 MedSAM 候选框轮廓细化](medsam_refinement.md)。

## 7. Agent 对话

### `POST /v1/llm/connections`

前端选择 OpenAI-compatible 模型时，先提交 `thread_id`、`base_url`、`model`、`api_key` 以及当前
身份字段。服务会实际执行一次严格结构化的非医学连接测试，成功后返回不透明 `connection_id`、
非敏感端点/模型元数据和过期时间。API key 不写 SQLite、审计或响应，只保存在当前
API 进程内，并绑定完整 `(owner_scope, user_id, thread_id)`；进程重启、到期或 DELETE 都会使连接
失效。后续 `/v1/agent/respond` 只发送 `connection_id`，不重复发送 secret。

### `DELETE /v1/llm/connections/{connection_id}`

query 必须包含 `thread_id`，research/development 档另可提供 `owner_scope` 与 `user_id`；production
身份仍由可信代理导出。成功返回 `{ "revoked": true }`，已不存在返回 `false`，跨身份访问按不可用
处理。

### `POST /v1/agent/respond`

请求：

```json
{
  "thread_id": "thread-1",
  "user_id": "local-user",
  "owner_scope": "local-demo:local-user",
  "message": "胸片异常后痰NAAT、培养和药敏分别有什么用？",
  "case_id": null,
  "llm_provider": "local_medgemma"
}
```

限制：`thread_id`、`user_id` 为 1..128；`owner_scope` 为 1..256；`message` 为 1..4000；`case_id`
可省略或为 null。`llm_provider` 默认为 `local_medgemma`；旧客户端的 `local_qwen` 仅作为同一本地
服务的兼容别名继续接受。选 `openai_compatible` 时必须传当前线程有效的 `llm_connection_id`，本地
模式禁止携带该字段。

成功返回完整 `AgentResponse` 顶层字段，并增加以下过程字段：

| 字段 | 语义 |
|---|---|
| `execution_receipt` | 兼容字段；有工具调用时等于最后一个 receipt，无调用时为 null |
| `execution_receipts` | 本轮实际执行的全部工具收据，按执行顺序排列 |
| `execution_plan` | ReAct 脱敏轨迹：`framework=langgraph`、`strategy=react_first_optional_plan`、实际节点/工具历史、可选计划及决策来源；`plan_metadata.planning_used` 表示是否实际规划 |
| `reflection` | 正常轮次为 null；仅在硬门拒绝、必需 Observation 缺失或工具失败触发修订时返回简短触发/修订计数，不含 CoT |
| `agent_trace` | v2 兼容轨迹：高层 TaskSpec、预算和终止原因；逐步事实以 `execution_plan` 与 receipt 为准，不含 CoT |

主链从 `load -> decide` 开始。LLM 通过 `tbx_react_decision` 的结构化语义任务判断当前需求，
执行器结合缓存状态和权限将任务解析为直接回答、单次工具调用或可选规划。复杂问题需要规划时才经过
`plan` 节点；兼容字段 `initial_plan` 的存在不代表发生过额外规划调用。四个公开工具是
`classify_cxr`、`localize_cxr`、`analyze_lung_anatomy`、`search_tb_knowledge`。工具返回
Observation 后服务端重新读取病例，再做下一次决策。一轮决策不会批量执行多个工具。
默认上限为 5 个决策步骤、4 次工具调用、3 次高成本视觉调用和 10 个成本单位；急症、权限、
前置条件、显式禁止项与预算由代码硬门控制。

当前主决策链使用严格 JSON Schema，不先尝试原生 tool calling，也不再增加独立意图解析调用。
provider 不可用或输出不合约时才采用受限规则托底；托底仍受同一权限和预算约束。模型不能改写
服务端绑定的病例身份、用户原始问题或工具参数。所有用户正文经过输出校验；内部思考、图状态、
结构化决策载荷和没有证据支持的病例判断不会直接进入 `AgentResponse.summary`。

上传时的解码和基础质量检查不是模型工具；质量询问直接读取病例已有证据。普通问答、能力/病例状态
以及纵向比较能力缺口可以零工具回答。产品能力说明和“汇总已完成分析”是代码拥有的受信投影：前者
只列出本系统实际能力，后者只读取公开分类、定位和二维肺野摘要，不调用模型复述私有病例状态，也
不产生工具 receipt。当前没有经过验证的纵向比较模型，即使提供前后片也不会输出
“好转、恶化或稳定”，也不会伪造比较 receipt。

与病例/TB 工具无关的问题可由 LLM 选择为通用问答，v4 可直接使用本轮结构化决策中的 `answer`，
不必额外调用独立回答模型。该路径返回 `response_kind=general_answer`、`execution_receipt=null`、
空 `execution_receipts/tool_names`。决策请求包含授权病例的简洁结构化状态和最近一对交互，不发送原始
图像；零工具调用并不表示没有病例上下文。服务的进程内聊天窗口最多保存三对交互，
原文不进入 SQLite 或审计，换 thread/重启后失效。provider 未配置或调用失败时返回明确模型状态，
不会返回“分类未运行；定位未运行”。明确询问病例状态时仍走 `CASE_STATUS`。

`search_tb_knowledge` 只接收原始问题。人群、检查实体、scope、subtopic 和适用场景在工具内部解析、
过滤并写入 receipt 的 `resolved_*`，主 Agent 不生成这些检索参数。请求体也不接受工具名、计划、
工具参数或内部状态。

每个 receipt 同时包含公开 `model_tool_name` 与内部 handler `tool_name`，并记录
plan/step/call/request/trace ID、selection source、幂等键、状态、权限与 v7 契约版本、截止时间、
运行时、Observation code、citation 数及输入/上下文/输出摘要，不包含消息原文。
普通模型、工具、契约或后端失败直接成为 Observation；依赖该证据的分支停止，但同一问题中独立的
指南或其他证据目标仍可继续。唯一自动重试是工具执行容量饱和且 receipt 明确标记
`retryable=true` 时，在预算允许的前提下对同一受限 invocation 恰好重试一次；第一次失败与第二次
尝试各自保留真实 receipt，最终回答只采用该逻辑步骤的最后一次 attempt。

`execution_plan.source=plan_react` 且 `framework=langgraph`；`graph_node_trace` 只保存本轮实际节点名。
`initial_plan` 保存公开目标，`plan_metadata.planning_used` 区分可选规划与直接决策，
`plan_revisions` 保存触发码/原因码，
`react_steps` 保存每步的公开工具或 answer、选择模式、状态、Observation code 和受限恢复标志。
`agent_trace.trace_version` 仍为 `tbx-agent-trace-v2`，用于兼容高层 TaskSpec、预算与终止原因。
`hidden_reasoning_persisted` 固定为 `false`。这表示图状态和轨迹不持久化自由推理，不表示信任 provider
自行隐藏思考；provider 正文仍经过上述输出校验。Streamlit 只显示真实 receipt，不把 Plan 目标模拟成已执行进度。

线程还保存最近一次成功且在白名单内的语义意图。正常 v4 每轮仍由模型结合有限上下文判断当前目标，
“继续”“展开”等短追问并非跳过模型的固定继承。运行时继续检查权限、缓存及未完成义务，
规则降级路径可以使用受限意图记忆。决策/计划来源、Schema 状态、模型和 token usage（如适用）记录在
`execution_plan.plan_metadata` 与 `react_steps`；工具 receipt 保留实际执行来源。客户端应读取
这些字段，不能把规则托底或缓存读取显示为新的模型推理。

指南工具先按来源状态、topic、jurisdiction、时间和 `allowed_claim_scope` 做准入，再按配置选择实际
检索模式。无合格命中时返回明确证据缺口；若 Dense/Hybrid 允许降级，receipt 必须记录实际模式与
原因，不能把 BM25 降级结果标记为 Hybrid。

如请求携带 `case_id`，服务会在动作选择前按 tenant + subject exact 校验；不存在或越权不能作为
普通线程关联继续执行。首次使用新 `thread_id` 会创建业务线程，同一 tenant/user/thread 在单进程内
串行。当前不持久化 Controller 执行状态，也不支持中断后恢复一轮 Agent 执行；线程、病例、审计
和制品仍是不同持久化边界。

医学表述层只能组织已经批准的结构化事实。通用回答层可以使用模型基础知识，但不能声称执行了没有
receipt 的工具，也不能绕过个体化用药和结核确诊边界。传输、provenance、Schema、枚举或安全检查
失败时丢弃候选文本并保留安全故障响应，可能标记 `narration_status=fallback_error`。原始图像、密钥
和完整指南正文不发送给文本 LLM。

## 8. 主动筛查

### `POST /v1/screening/sessions`

请求：

```json
{
  "thread_id": "thread-1",
  "user_id": "local-user",
  "owner_scope": "local-demo:local-user",
  "case_id": null,
  "consent": true
}
```

`consent=true` 创建 `collecting` 会话，第一题固定为急症警示；`consent=false` 创建 `cancelled` 会话，并返回“已取消、答案未保留”的摘要。API 不支持省略同意后先停在 `consent_pending`。若提供 `case_id`，必须存在且 owner 匹配。

响应：

```json
{
  "session": {
    "session_id": "screen-uuid",
    "thread_id": "thread-1",
    "user_id": "local-user",
    "owner_scope": "local-demo:local-user",
    "case_id": null,
    "consent": true,
    "guideline_rule_version": "tb-active-screening-2026-v1",
    "status": "collecting",
    "answers": {},
    "next_question_id": "emergency_red_flags",
    "result": null,
    "created_at": "2026-08-28T10:00:00Z",
    "updated_at": "2026-08-28T10:00:00Z"
  },
  "response": {
    "request_id": "request-uuid",
    "trace_id": "trace-uuid",
    "thread_id": "thread-1",
    "case_id": null,
    "response_kind": "active_screening_question",
    "summary": "请回答下一项；也可以回答不知道、跳过或取消。",
    "visual_result": null,
    "diagnostic_information": [],
    "next_step_information": [],
    "treatment_education": [],
    "limitations": ["本结果仅用于主动筛查分层，不能确诊或排除肺结核，也不计算或合并疾病概率。"],
    "citations": [],
    "review_status": null,
    "next_question": {
      "question_id": "emergency_red_flags",
      "text_zh": "目前是否有以下任一紧急警示表现？可多选；若均无请选择‘以上均无’。",
      "answer_type": "multi_choice",
      "choices": ["严重呼吸困难", "口唇或面色发青", "意识模糊或晕厥", "大量或持续咯血", "严重胸痛", "已测得血氧饱和度低于90%", "以上均无"],
      "sensitive": true,
      "source_id": "TBX_AGENT_SAFETY_V1",
      "locator": "emergency_red_flags",
      "ask_if": {}
    },
    "urgency": null,
    "reused_existing_assessment": false,
    "safety_policy_id": "tbx-agent-safety-v1"
  }
}
```

错误：关联病例/线程相关对象不存在为 404；现有线程 tenant/subject 不匹配为 403；请求 schema 为 422。

### `POST /v1/screening/sessions/{session_id}/answers`

请求：

```json
{
  "user_id": "local-user",
  "owner_scope": "local-demo:local-user",
  "question_id": "emergency_red_flags",
  "answer": ["以上均无"]
}
```

`question_id` 必须等于 session 的 `next_question_id`，不允许乱序。`answer` 按题型解释：

- boolean：JSON `true/false`，或 `yes/no`、`y/n`、`1/0`、`是/否`、`有/没有/无`；
- single choice：必须与 `choices` 中一个字符串完全相等；
- multi choice：一个非空字符串或非空字符串集合；“以上均无”等排他选项不能与其他项同时提交；
- 特殊答案：`unknown/不知道/不清楚/不确定`；`skip/跳过/不愿回答/暂不回答`；
- 取消答案：`cancel/取消/退出/停止问询`，会取消并清空全部本次答案。

成功返回更新后的 `{session, response}`。若急症题选择任何非“以上均无”项，会立即完成，会话 `urgency=emergency`、`citations=[]`，并在 `triggers` 中记录 `local_clinical_safety_policy:emergency:*`。否则按年龄（15 岁界限）、HIV、免疫抑制、症状、高风险和重点人群继续。

错误：session 不存在为 404；乱序、错误题型/选项、空多选、已完成/已取消状态、规则版本不符为 422。当前该端点的 user 或 owner 不匹配也可能是 422。

### `POST /v1/screening/sessions/{session_id}/cancel`

请求：

```json
{
  "user_id": "local-user",
  "owner_scope": "local-demo:local-user"
}
```

成功返回 `{session, response}`；session 为 `cancelled`、`consent=false`、`answers={}`、`result=null`。这是业务 payload 的逻辑清空，不保证 SQLite WAL、备份或磁盘取证层面的安全擦除。已完成 session 不能取消，返回 422。不存在为 404；当前 user/owner 不匹配也可能是 422。

### `GET /v1/screening/sessions/{session_id}`

research/dev Query：`owner_scope` 与 `user_id` 必填。production 由签名派生，可省略兼容字段。

响应为 `{session, response}`。这是只读渲染，`response.request_id` 和 `trace_id` 固定为 `read-only`，不会新增审计事件；固定模板输出仍会再次通过 `SafetyVerifier`。不存在为 404，tenant/subject 不匹配为 403。

## 9. 人工复核

### `GET /v1/reviews/pending`

research/dev Query：`owner_scope` 与 subject `user_id` 必填；production 由签名派生。

响应为 `ReviewRecord[]`，按 `updated_at` 升序，只返回 exact tenant + subject 且
`status=pending` 的记录。无记录返回 `[]`。production 能认证 actor/subject 断言，但没有 reviewer
role，actor 不能列出其他 subject 的队列。

### `GET /v1/reviews/{review_id}`

research/dev Query：`owner_scope` 与 subject `user_id` 必填；production 由签名派生。按 exact review
ID 读取 pending/completed 记录；tenant/subject 不匹配为 403。当前没有角色授权。

### `PATCH /v1/reviews/{review_id}`

请求：

```json
{
  "owner_scope": "local-demo:local-user",
  "user_id": "local-user",
  "reviewer_id": "reviewer-1",
  "expected_version": 1,
  "decision": "indeterminate",
  "note": "需核对原始影像与临床信息"
}
```

约束：身份字段最长分别为 128/256，`expected_version >= 1`，`note` 最长 2,000；`decision` 必须是第 2.5 节四个枚举之一。research/dev 三个身份值均为不可信演示声明；production 可省略，由签名
subject/actor 派生并拒绝冲突。

成功直接返回完成后的 `ReviewRecord`：`status=completed`、`version=expected_version+1`，并记录 actor
和时间。病例 `review_status` 同事务改为 completed、`record_version` 加一，但模型融合视觉结果不会
被替换。API 固定 `allow_cross_subject=false`；当前身份 actor 若不是病例 subject，仍不能因
`reviewer_id` 名称或职业自报获得访问。真实临床 reviewer workflow 要等签名 RBAC。

错误：不存在为 404；owner 不匹配为 403；过期版本、已完成或竞争失败为 409；非法字段/枚举为 422。

## 10. 报告

### `POST /v1/cases/{case_id}/reports`

请求：

```json
{
  "owner_scope": "local-demo:local-user",
  "user_id": "local-user",
  "actor_id": "local-user"
}
```

成功响应：

```json
{
  "report_id": "tbx-report-case-uuid-20260828T100000Z",
  "generated_at": "2026-08-28T10:00:00Z",
  "markdown_download_url": "/v1/cases/case-uuid/reports/tbx-report-case-uuid-20260828T100000Z/markdown",
  "json_download_url": "/v1/cases/case-uuid/reports/tbx-report-case-uuid-20260828T100000Z/json"
}
```

Markdown 报告包含辅助筛查标签、分类训练类/规则、检测器 advisory 角色、候选区域数、质量状态、
运行 ID、独立人工复核记录、下一步限制、引用和审计哈希。JSON 报告包含完整病例快照、独立 review
快照、citation、知识快照及固定 `clinical_validation=false`；公开 JSON 递归删除
`image_artifact_ref`。复核记录不会覆盖原始模型证据或融合标签。

生成响应不返回 `markdown_path`/`json_path` 或其他服务器本地路径。下载 URL 是相对
API 路径。research/dev 客户端请求时必须附加 `owner_scope` 和 `user_id` query，例如：

```http
GET /v1/cases/case-uuid/reports/tbx-report-case-uuid-20260828T100000Z/json?owner_scope=local-demo%3Alocal-user&user_id=local-user
```

production 下载端点从签名派生 tenant + subject，兼容 query 可省略；若提供必须一致。随后只允许
`artifact_format=markdown|json`，并把解析文件限制在该病例 reports 目录。HMAC 不是限时报告 URL
或报告签名：制品仍无撤回/WORM/时间戳服务，且跨 subject 下载因缺 RBAC fail closed。病例不存在、
格式/制品不存在为 404，tenant/subject 不匹配为 403，文件系统失败目前可能为 500。

## 11. 客户端必须执行的最小规则

1. 界面必须持续提供病例详情入口，并在病例详情中显示“辅助筛查、非确诊/排除、无个体化处方”等
   完整边界；主对话不要求每轮重复这些长警示。
2. 同时检查 HTTP 状态、`visual_result`、`review_status`、`urgency` 和 `limitations`；不得把 200 当医学成功或阴性。
3. `technical_failure` 必须走技术重试/人工处理，`pending_human_review` 必须保持待处理状态。
4. `model_not_flagged` 的主对话应只显示一句准确模型结论，例如“模型没有识别出结核”；
   `non_tb_abnormal` 可显示“模型识别为非结核异常”。不得写成“胸片正常”或排除结论。完整的
   “不用于确诊或排除”等 `limitations` 必须保留，并在病例详情持续可见，不要求与主结论同屏重复。
5. `predicted_class` 只能显示为模型训练类；不得把 `healthy` 改写为“胸片正常”，把
   `sick_non_tb` 改写为具体疾病/已排除结核，或把 `tb` 改写为肺结核诊断。
6. `urgency=emergency` 必须停止继续问卷/模型流程，并优先显示当地急救/急诊提示。
7. 隔离测试档若启用 mock，不得去掉 synthetic 标识；所有档位均不得去掉 citation、知识版本、
   模型/策略版本、`inference_contract`、`runtime_verified` 或临床验证 false 标志。
8. 不得在日志、URL、分析平台或外部 tracing 中写入原图、原始问卷答案或可识别信息。
9. research/dev 不得把自报 `owner_scope/user_id` 当生产授权；production 必须经 HMAC 代理，且不得把
   same-subject 隔离冒充 reviewer RBAC。跨 subject 功能在签名 permission 契约完成前必须保持关闭。

## 12. OpenAI-compatible 原始模型协议测试

这一节描述 `llama-server` 在 `http://127.0.0.1:11435/v1` 上的原生协议测试面，不是本页其余部分所述
的 `http://127.0.0.1:8000` TBX-Agent FastAPI。两者不能混称：

| 接口 | 地址 | 输出边界 |
|---|---|---|
| TBX-Agent API | `http://127.0.0.1:8000/v1/...` | 受硬门约束的 ReAct 与可选规划、四公开工具、RAG、视觉/安全/病例边界后的结构化 Agent 输出 |
| raw LLM protocol test | `http://127.0.0.1:11435/v1` | 原始 MedGemma 文本生成；不经过 Agent 安全链，不持久化到病例/记忆/报告 |

活动原始模型使用固定 alias `tbx-medgemma-1.5-4b-it-q4-k-m`、纯文本、thinking=false、输入不超过
4096 tokens、输出不超过 512 tokens，`clinical_authority=false`。已提交的
[`configs/openai_compat_test.yaml`](../configs/openai_compat_test.yaml) 仍是旧 Qwen 运行时的历史 smoke
收据，不能当作 MedGemma 已通过协议评测。MedGemma 的最小协议范围保持为：

- `GET /v1/models`；
- `POST /v1/chat/completions` 的同步 JSON；
- 同一路由 `stream=true` 的 SSE，并以 `data: [DONE]` 正常结束。

不把原始端口当 Agent 工具面；不得通过它调用 Agent tools、函数、图像、多模态、embeddings、files、
audio 或 OpenAI Responses API。原始输出不能用于筛查、诊断/排除、治疗建议或发布结论。

### 12.1 凭据与网络边界

11435 只监听 loopback。`llama-server b10517` 的鉴权是端点特定行为，不可笼统称为“整个 `/v1`
均已鉴权”：

- 原生 `GET /v1/models` 是公共端点，实测携带伪 Bearer 仍返回 HTTP 200。它不能作为鉴权探针；该
  信息暴露风险目前只由 `127.0.0.1` loopback 边界缓解。
- `POST /v1/chat/completions` 生成端点强制 Bearer，实测错误 key 返回 HTTP 401。内部 key 只能从 ACL
  限制的 `LLAMA_CPP_API_KEY_FILE` 或一个进程的 `LLAMA_CPP_API_KEY` 环境变量读取。

禁止把内部 key 写入 YAML、源码、`.env` 提交、日志、截图、命令历史或错误响应，也不要
`print`/`echo` 它。

远程用户不得直连 11435，也不得拿到或复用内部 key。远程兼容访问需要一个尚未实现的独立 gateway：
gateway 必须终止 TLS，使用另一套可轮换 Bearer 身份、限流和审计，再通过受控内部边界访问模型。
因此本文没有远程 URL 或公共 key 示例。当前唯一获准范围是同一主机上的 loopback 协议测试；其中
只有生成端点由内部 Bearer 保护。

### 12.2 OpenAI Python SDK：models、同步与 SSE

```python
import os
from pathlib import Path

from openai import DefaultHttpxClient, OpenAI


def load_local_llama_key() -> str:
    direct = os.environ.get("LLAMA_CPP_API_KEY", "").strip()
    if direct:
        return direct
    path = Path(os.environ["LLAMA_CPP_API_KEY_FILE"])
    keys = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(keys) != 1 or len(keys[0]) < 32:
        raise RuntimeError("ACL key file must contain exactly one strong key")
    return keys[0]


with OpenAI(
    base_url="http://127.0.0.1:11435/v1",
    api_key=load_local_llama_key(),
    max_retries=0,
    http_client=DefaultHttpxClient(trust_env=False, follow_redirects=False),
) as client:
    # 此调用核对 alias；models 原生公开，成功不代表通过了鉴权。
    models = client.models.list()
    assert [item.id for item in models.data] == ["tbx-medgemma-1.5-4b-it-q4-k-m"]

    common = {
        "model": "tbx-medgemma-1.5-4b-it-q4-k-m",
        "messages": [{"role": "user", "content": "只回复：协议测试成功"}],
        "temperature": 0,
        "max_tokens": 16,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    completion = client.chat.completions.create(**common)
    print(completion.choices[0].message.content)

    usage_seen = False
    with client.chat.completions.create(
        **common,
        stream=True,
        stream_options={"include_usage": True},
    ) as stream:
        for chunk in stream:
            usage_seen = usage_seen or chunk.usage is not None
            text = chunk.choices[0].delta.content if chunk.choices else None
            if text:
                print(text, end="", flush=True)
    assert usage_seen
    print()
```

示例固定 `max_retries=0` 并禁用环境代理与重定向，避免内部 ACL key 被代理或跳转到非预期地址。它只
打印模型输出，不打印 key。不要把真实患者、影像、身份、问卷答案或其他 PHI 放入原始协议测试。

### 12.3 PowerShell + curl：models 与 SSE

下例先在不带 key 的情况下读取公共 models，再从 ACL 文件把 key 放入当前 PowerShell 进程环境，仅供
生成请求使用。curl 自己展开环境变量，Bearer 值不会出现在文档或 curl 命令参数中；结束后立即清理
该进程变量。curl 需要支持 `--variable` 与 `--expand-header`；旧版 curl 应改用上面的 SDK 示例，
不能退回把 key 拼在命令行中。

```powershell
# b10517 原生公共端点；HTTP 200 不能证明鉴权成功。
curl.exe --silent --show-error --fail-with-body --noproxy '*' `
  http://127.0.0.1:11435/v1/models

$env:LLAMA_CPP_API_KEY_FILE = '<protected-key-file>'
$env:LLAMA_CPP_API_KEY = (Get-Content -LiteralPath $env:LLAMA_CPP_API_KEY_FILE |
  Where-Object { $_.Trim() -and -not $_.TrimStart().StartsWith('#') } |
  Select-Object -First 1).Trim()

try {
  $body = @{
    model = 'tbx-medgemma-1.5-4b-it-q4-k-m'
    messages = @(@{ role = 'user'; content = 'Reply only: protocol test ok' })
    temperature = 0
    max_tokens = 64
    stream = $true
    chat_template_kwargs = @{ enable_thinking = $false }
  } | ConvertTo-Json -Depth 6 -Compress

  $body | curl.exe --silent --show-error --fail-with-body --no-buffer --noproxy '*' `
    --variable '%LLAMA_CPP_API_KEY' `
    --expand-header 'Authorization: Bearer {{LLAMA_CPP_API_KEY}}' `
    --header 'Content-Type: application/json' `
    --data-binary '@-' `
    http://127.0.0.1:11435/v1/chat/completions
} finally {
  Remove-Item Env:LLAMA_CPP_API_KEY -ErrorAction SilentlyContinue
}
```

对生成请求，HTTP 非 2xx、SSE 中断、缺少 `[DONE]`/终止 usage、模型 alias 漂移或
thinking/content/token 边界不符都应视为失败，不能补写成功结果。models 在伪 key 下返回 200 是已知
原生行为，不是生成鉴权证据。原生 llama-server 目前没有 TBX gateway 的统一 OpenAI error adapter；
不要把其错误文本直接展示给远程用户。

### 12.4 运行本地协议烟测

```powershell
& .\.venv\Scripts\python.exe scripts\test_openai_compatible_api.py --runtime-config configs\llm_runtime.yaml
```

命令会在外部运行目录保存 models、同步 chat、SSE、终止 usage、错误 key 和运行时间的工程记录。
失败记录必须保留，峰值显存未测量时明确记为 unavailable，不能填 0 或从其他基准补录。

成功结果只证明指定主机、运行配置和固定非医学输入下的协议兼容性。它不证明开放式生成、
TBX-Agent 安全链、应用发布或临床质量，也不能作为 rank03 模型选择证据。
