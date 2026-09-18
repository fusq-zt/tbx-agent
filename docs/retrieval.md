# 指南检索子系统

TBX-Agent 的在线指南工具支持 `sparse`、`dense` 和 `hybrid` 三种模式。默认配置是
metadata-gated BM25（`sparse`）；BGE-M3/Qdrant Local 只有在模型制品和索引身份完整匹配时才会
懒加载。检索返回的是候选证据，不是诊断、处方或医学事实验证。

## 1. 当前实现状态

| 能力 | 当前状态 |
| --- | --- |
| 统一 corpus 与稳定 chunk/citation ID | 已实现 |
| 来源 ID、审核状态、地区、主题、日期、claim scope 排序前准入 | 已实现 |
| BM25 sparse | 已接入在线 Agent；默认模式 |
| BGE-M3 dense embedding | 已实现离线本地适配器；显式模型目录，懒加载，不自动下载 |
| Qdrant Local | 已实现离线 generation 构建、原子 current pointer 和在线只读查询 |
| Hybrid | 已实现 BM25 + dense 分别召回、加权 RRF、确定性去重 |
| 在线模式选择与降级状态 | 已实现；requested/effective mode 和稳定原因码分开记录 |
| 固定 query/qrels 评测 | 已实现 BM25/Dense/Hybrid 同套输入、逐模式失败分类和 JSON/Markdown 报告 |
| sqlite-vec | 保留精确小语料适配器与测试，不是默认，也不是当前索引 CLI 的目标 |
| Qdrant Server | 仅保留查询边界；当前在线 Agent 与构建 CLI 不接入 |
| reranker | 代码边界存在，在线 `GuidelineRetriever` 明确拒绝启用 |
| BGE-M3 sparse/ColBERT | 未实现 |
| 医学专家 qrels/临床验证 | 未完成，且临床验证不属于本仓库现有证据 |

## 2. 数据流

```text
reviewed source_manifest.json + chunks.jsonl
  -> RetrievalDocument（稳定 ID、内容哈希、来源、locator、准入元数据）
  -> RetrievalFilter 先执行来源/审核/地区/主题/日期/claim-scope 准入
  -> sparse: BM25
     dense: BGE-M3 query vector -> promoted Qdrant Local generation
     hybrid: BM25 + dense -> weighted RRF
  -> 精确/保守近重复抑制
  -> top-k RetrievalHit
  -> 重新校验不可变 chunk identity 与 caller claim scope
  -> citation + content-free retrieval receipt
```

三种模式必须读取同一 corpus。Dense 不允许用另一份“语义优化文本”绕过 sparse 的审核元数据；
Hybrid 也不直接相加 BM25 与 cosine 原始分数，而是用 rank 进行 RRF。

## 3. Corpus 与准入契约

`knowledge/source_manifest.json` 登记来源和准入属性；`knowledge/chunks.jsonl` 保存经审核的片段。
每个 `RetrievalDocument` 至少绑定：

- `chunk_id`、`source_id`、source/content SHA-256；
- 标题、机构、URL、章节或页码 locator；
- jurisdiction、发布日期、语言、topics；
- `allowed_claim_scopes`、`review_status` 与 `retrievable`。

摄取流水线输出默认仍是 `pending_medical_review` / `retrievable=false`。只有单独审核并明确提升后，
内容才可进入在线快照。检索后还会用可信 corpus 重新证明 hit identity；未知 chunk、被篡改支持文本、
非法分数、重复 ID 或 scope/jurisdiction/topic 不匹配都会被拒绝。

用户明确指定《指南名称》或“第 N 页”时，检索必须命中该来源与覆盖页码的 locator；不能用标题相似
的其他真实指南替代不存在的来源。

调用方传入的 `required_source_ids` 已进入 `RetrievalFilter`，在 BM25、内存 Dense/Hybrid、
sqlite-vec 和 Qdrant 的召回截断之前执行。多个来源 ID 表示允许其中任一来源，空元组表示不额外限制。
来源准入与其他 metadata 条件共同生效；不能先全局取 top-k，再因其他来源占满候选而返回伪空集。
知识层仍在返回引用前复核来源，未知的必需来源保持空结果。

## 4. Sparse、Dense 与 Hybrid

### 4.1 Sparse

`BM25Index` 无额外模型依赖，按 `chunk_id` 做稳定 tie-break。tokenizer 保留拉丁/数字 token，并为
中文生成 unigram、bigram 和 trigram。它适合做可审计基线，但没有学习到的同义词泛化，也不是医学
相关性判定器。

### 4.2 Dense

本地适配器使用 `FlagEmbedding.BGEM3FlagModel` 的 1,024 维 normalized dense output。当前只请求
`return_dense=True`；没有把 BGE-M3 sparse 或 ColBERT 向量写进索引。

加载遵守以下约束：

- 模型必须已经位于显式本地目录；`local_files_only=True`、`trust_remote_code=False`；
- `model_sha256` 是目录中“相对路径 + 文件内容”的聚合哈希，而不是随意填写的模型名；
- revision、维度、normalization、query/document prefix 都进入 provenance；
- `FlagEmbedding`、模型和 torch 只在第一次真实 embedding 时懒加载；
- 也可配置 loopback OpenAI-compatible embedding endpoint，但仍必须固定 model/revision/hash。

前缀身份迁移：`EmbeddingProvenance` 现在实际包含 `query_prefix` 和 `document_prefix`，适配器输出、
在线配置、构建器与评测器使用同一份有效前缀。旧版本未记录前缀的索引即使使用空前缀也会因
`embedding_fingerprint` 不匹配而失效；升级后必须显式重建 dense 索引，不能只修改旧 manifest。

### 4.3 Qdrant Local generation

当前构建器只写 Qdrant Local。文档按 `chunk_id` 排序后向量化，generation identity 绑定：

- corpus SHA-256；
- embedding fingerprint；
- 全部向量的 SHA-256；
- qdrant-client 版本、维度与数量。

新 generation 先写临时目录，再移动为完整 generation，最后原子替换 `current.json`。另外写入
`index-manifest.json`，把 source manifest、chunks、模型和 generation 再次绑定。在线查询先验证这些
身份；旧索引不能因目录存在就被静默接受。

`runtime://retrieval/qdrant_local` 会解析到 `TBX_AGENT_DATA_ROOT`（其次 `TBX_RUNTIME_ROOT`，否则平台
默认数据目录）下，不会写入 Git checkout。

Qdrant Local 的数据库对只读客户端也使用独占锁。当前按解析后的根目录共享进程内锁，把不同实例的
客户端打开、查询、校验、关闭串行执行；每次查询在 `finally` 中关闭客户端，异常后下一次请求可重新
打开。构建/提升也使用同一个根目录锁。`QdrantLocalVectorStore.close()` 等待正在执行的操作结束，
然后禁止该实例继续查询、构建或读取 manifest；重复关闭安全。`RetrievalEngine.close()` 和
`GuidelineRetriever.close()` 向下释放已创建资源，不会为了关闭而加载模型。

实例首次读取的 generation 会固定到其生命周期结束；外部提升新 generation 后应关闭旧在线实例并
创建新实例，重新执行 corpus/model/generation 身份验证。已有请求因此不会在两代索引间隐式切换。
此锁只协调同一进程；多个 API worker 不应共用同一个 Qdrant Local 根目录。

来源过滤在 Local 使用嵌套 payload `document.source_id`；
Qdrant Server 查询边界要求远端 payload 提供顶层 `source_id`。前述 embedding prefix 身份迁移仍需
按其规则显式重建旧索引。

### 4.4 Hybrid 与去重

Hybrid 分别取 BM25 与 dense 候选，以 `weight / (rank_constant + rank)` 融合。原始 backend rank 与
score 仍保留在 hit 中。完全相同内容会抑制；近重复用 token-shingle Jaccard，默认不跨独立来源激进
合并，以免丢掉互相印证的指南。所有被抑制 ID 写入 receipt。

RRF 权重、candidate limit 和近重复阈值是评测变量，不是 UI 偏好；调整后应使用同一开发 qrels
重新报告。

## 5. 在线模式与故障恢复

在线服务只通过 `TBX_AGENT_RETRIEVAL_CONFIG` 选择一份严格 YAML 运行契约；未设置时使用项目
`configs/retrieval.yaml`。该变量可以指向仓库外绝对路径，也可以使用相对于项目根目录的路径。其他
环境变量不会覆盖契约中的模式、向量路径或 collection；当前真实模式选择项是所选 YAML 中的
`engine.retrieval_mode`：

```yaml
engine:
  retrieval_mode: sparse  # sparse | dense | hybrid
  fallback_to_sparse: true
dense:
  enabled: false
vector_store:
  backend: qdrant_local
  qdrant_path: runtime://retrieval/qdrant_local
```

默认 `sparse + dense.enabled=false` 不加载 BGE/Qdrant。请求 Dense/Hybrid 时，在线层第一次查询才构建
adapter、读取 current pointer 并校验 corpus/model/generation；成功后在进程内复用。

`RetrievalRuntimeState` 始终区分：

- `requested_mode`：配置请求的模式；
- `effective_mode`：本次实际使用的模式；
- `dense_initialized`；
- 稳定 `status_code` 与 `fallback_reason`。

稳定状态包括 `sparse_ok`、`dense_ok`、`hybrid_ok`，以及 dense disabled、依赖缺失、索引过期、初始化
不可用、契约错误、query mismatch/query outage 等 sparse fallback。receipt 同时记录每个 backend 的
`used / not_queried / disabled / unavailable / mismatch`。因此 UI/日志不能只显示 requested mode。

索引初始化失败会在当前进程隔离，不会每个用户问题都重复加载损坏 generation。初始化和 query 阶段
失败是否降级都由 `fallback_to_sparse` 控制。设为 `false` 时，Dense/Hybrid 模式遇到 dense 被禁用、
依赖缺失、索引失配或初始化失败都会报错；同一进程后续请求继续报出初始化错误。允许降级时，BM25
的 generation/manifest 重新成为本次 receipt 身份。

### 5.1 检索回执 schema v2

新 `RetrievalReceipt` 使用 `schema_version=2`。保留原有配置、query/corpus/generation、后端状态、
降级原因与去重记录，并新增实际引擎请求的 `top_k`、全部 `filters`、`request_sha256`，以及有序
`returned_chunks`。每个结果身份包含 chunk ID、source ID、content/source SHA-256，不包含证据正文。
集合型 filters 按值排序；参数顺序不同但含义相同的请求得到相同请求身份。`retrieval_id` 绑定整个
回执主体，因此 top-k、过滤条件或返回 chunk 身份/顺序变化均会改变 ID。

该回执记录引擎召回结果。知识层为相关性门、明确页码约束和 scope 软排序取得的候选上限，可能大于
最终引用数；最终交给回答的引用仍以知识层返回值和重新校验为准。

使用 `RetrievalReceipt.from_dict()` 读取回执：v1 可原样读取并校验历史 retrieval ID；新增字段是
`None`（未知），`to_dict()` 保留其 v1 形状与身份。不得把旧回执填上当前默认 filters、top-k 或结果
而宣称已具备 v2 证明。v2 读取同时校验请求哈希与完整回执哈希。此次升级会生成新的 retrieval ID，
不改写历史回执、知识快照或 qrels。

## 6. 准备 BGE-M3 与构建索引

仓库不下载或分发 BGE-M3 权重。先在合规位置准备固定 revision 的完整本地 snapshot，再安装可选依赖：

```bash
python -m pip install -e '.[retrieval]'
```

计算实现实际校验的聚合哈希：

```bash
python -c "from pathlib import Path; from tbx_agent.retrieval.embeddings import local_artifact_sha256; print(local_artifact_sha256(Path('<external-models>/bge-m3')))"
```

复制 `configs/retrieval.yaml` 为部署配置并填写：

```yaml
engine:
  retrieval_mode: hybrid
dense:
  enabled: true
  adapter: bge_m3_local
  model_id: BAAI/bge-m3
  model_sha256: <64-hex-aggregate-hash>
  revision: <immutable-revision>
  dimensions: 1024
vector_store:
  backend: qdrant_local
  qdrant_path: <external-runtime>/retrieval/qdrant_local
  qdrant_collection: tbx-guidelines
```

可选的 `configs/retrieval_bge_m3_eval.yaml` 记录一组固定 snapshot 身份。只有本地模型树实际匹配时
才可使用，不能仅复制其中哈希。模型目录布局或内容变化后应生成新配置并重建索引。

设置本地运行输入：

```dotenv
TBX_AGENT_RETRIEVAL_CONFIG=<external-runtime>/config/retrieval.yaml
RAG_EMBEDDING_MODEL_PATH=<external-models>/bge-m3
RAG_EMBEDDING_CACHE_DIR=<external-cache>/huggingface
RAG_EMBEDDING_DEVICE=cpu
```

CLI 的 `--config` 与在线服务的 `TBX_AGENT_RETRIEVAL_CONFIG` 应指向同一文件。环境变量中的模型目录、
缓存目录和设备只向 BGE 适配器提供本地运行输入；它们不会切换检索模式、向量后端或 collection。

先做只读 dry-run；默认 sparse-safe 配置也可以运行这一步：

```bash
python scripts/build_rag_index.py \
  --knowledge-dir knowledge \
  --config <external-runtime>/config/retrieval.yaml \
  --dry-run
```

真实构建：

```bash
python scripts/build_rag_index.py \
  --knowledge-dir knowledge \
  --config <external-runtime>/config/retrieval.yaml \
  --output <external-runtime>/retrieval/qdrant_local \
  --model-path <external-models>/bge-m3 \
  --cache-dir <external-cache>/huggingface \
  --device cpu \
  --batch-size 8
```

`--output` 必须与在线配置的 `qdrant_path` 指向同一根目录。已有 promoted index 时命令拒绝覆盖；确认
新 corpus/model 身份后显式加 `--force`。命令不会删除历史 generation，也不会联网获取模型。

## 7. 检索检查

当前回归场景为 `smoke_v5`、`core_guideline_seven_v2` 和 `cdc_supplemental_v2`。
配置绑定问题、相关性标签和知识快照。版本化 fixture 用于兼容性与回归检查，
不同语料版本的结果不能混用。

BM25 不需要向量模型。固定场景的运行输出应放在仓库外：

```bash
python scripts/build_rag_index.py --dry-run
python scripts/evaluate_rag_retrieval.py \
  --queries evaluation/retrieval/smoke_v5/queries.jsonl \
  --qrels evaluation/retrieval/smoke_v5/qrels.jsonl \
  --suite-config evaluation/retrieval/smoke_v5/config.json \
  --config configs/retrieval.yaml \
  --modes bm25 \
  --output <external-runtime>/evaluation/rag-smoke.json
pytest tests/test_retrieval.py tests/test_knowledge.py \
  tests/test_rag_index_build.py tests/test_rag_retrieval_evaluation.py
```

报告区分检索指标、引用覆盖和后端失败，并在外部目录保留配置、源码身份、耗时及失败记录。
进程正常退出或合成向量测试通过，不代表真实 Dense/Hybrid 检索质量；真实评测需要显式准备
上文所述模型与匹配索引。

已公布结果及适用范围见[测试概览](evaluation.md)。
