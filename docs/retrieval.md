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
| reranker | 代码边界存在，但在线 `GuidelineRetriever` 明确拒绝启用；本轮未评测 |
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

来源过滤在 Local 使用已有嵌套 payload `document.source_id`，本次来源/生命周期变更无需重建索引；
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

## 7. 检索评测

当前 v7 知识快照使用 `smoke_v5`；七个核心指南问题使用
`core_guideline_seven_v2`；19 个美国 CDC 补充证据缺口使用 `cdc_supplemental_v2`。后者专门检验
Xpert、培养、TST/IGRA、胸片正常、孕期评估、密接、漏服、不良反应、传染性复评和返工返校等
实体级检索，不能把美国补充证据当作中国或 WHO 路径的替代。`smoke_v1/v2/v3` 与
`core_guideline_six_v1/v2`、`smoke_v4`、`core_guideline_seven_v1` 和 `cdc_supplemental_v1`
只保留作历史审计，因绑定旧 corpus generation，不能直接重放到 v7，
也不能拿旧结果选择当前检索器。

每个固定 fixture 的 `config.json` 是评测运行契约。命令行在 queries 与 qrels 位于同一目录时会自动
发现它，但可审计运行仍建议显式传入 `--suite-config`。运行前会严格校验 query/qrels 文件哈希、
query count、seed、suite/split 身份、knowledge snapshot、source manifest、chunks 与 corpus generation。
其中旧 smoke fixture 没有单独保存 `query_count` 字段，评测器会从已通过 SHA-256 验证的 queries 文件
计算并校验条数，同时在报告中记录该来源。

suite 还绑定一份 retrieval config SHA-256。使用 Dense/Hybrid 评测配置替换该基线配置时，必须显式
传入 `--allow-retrieval-config-override`；否则命令失败。报告会同时保留绑定配置和实际配置的路径、
哈希、是否匹配以及 override 是否实际发生。该开关只授权比较另一份检索配置，不会放宽 corpus、
queries、qrels 或 split 身份校验。

BM25 不需要模型或向量索引：

```bash
python scripts/evaluate_rag_retrieval.py \
  --queries evaluation/retrieval/smoke_v5/queries.jsonl \
  --qrels evaluation/retrieval/smoke_v5/qrels.jsonl \
  --suite-config evaluation/retrieval/smoke_v5/config.json \
  --knowledge-dir knowledge \
  --config configs/retrieval.yaml \
  --modes bm25 \
  --k 1,3,5,10 \
  --output <external-runtime>/evaluation/bm25.json \
  --markdown-output <external-runtime>/evaluation/bm25.md
```

CDC 补充证据回归使用同一 corpus 和 BM25 配置：

```bash
python scripts/evaluate_rag_retrieval.py \
  --queries evaluation/retrieval/cdc_supplemental_v2/queries.jsonl \
  --qrels evaluation/retrieval/cdc_supplemental_v2/qrels.jsonl \
  --suite-config evaluation/retrieval/cdc_supplemental_v2/config.json \
  --knowledge-dir knowledge \
  --config configs/retrieval.yaml \
  --modes bm25 \
  --k 1,3,5,10 \
  --output <external-runtime>/evaluation/cdc-supplemental-bm25.json \
  --markdown-output <external-runtime>/evaluation/cdc-supplemental-bm25.md
```

完成 Dense index 后，在同一 corpus/qrels 上运行：

```bash
python scripts/evaluate_rag_retrieval.py \
  --queries evaluation/retrieval/smoke_v5/queries.jsonl \
  --qrels evaluation/retrieval/smoke_v5/qrels.jsonl \
  --suite-config evaluation/retrieval/smoke_v5/config.json \
  --config <external-runtime>/config/retrieval.yaml \
  --allow-retrieval-config-override \
  --modes bm25,dense,hybrid \
  --k 1,3,5,10 \
  --model-path <external-models>/bge-m3 \
  --cache-dir <external-cache>/huggingface \
  --device cpu \
  --output <external-runtime>/evaluation/three-mode.json \
  --markdown-output <external-runtime>/evaluation/three-mode.md
```

报告包含 Recall、MRR、graded nDCG、citation/source recall、hard-negative rejection、p50/p95 latency、
逐 query retrieved IDs 和失败分类。依赖缺失、stale generation、契约错误和后端不可用作为模式失败
写入报告；调用者必须检查每个 mode 的 `status`，不能只看进程退出码。

报告还记录 suite 校验结果、Python/平台和关键依赖版本、完整运行 wall time、进程 peak RSS，以及
torch 已加载且 CUDA 可用时的逐设备 peak allocated/reserved VRAM。报告另外计算 `pyproject.toml + src/**/*.py + scripts/**/*.py` 的确定性
source-tree manifest SHA-256；它用于标识实际运行源码范围，不能替代正式提交和干净工作树。

同一进程用 `--modes bm25,dense,hybrid` 适合比较确定性质量指标，但不适合直接宣称三种模式的公平
延迟：Dense 首次查询包含模型校验与冷加载，随后 Hybrid 会复用已热模型。延迟/资源晋级测试应把每个
mode 放在独立进程中，并分别报告冷启动、显式 warm-up 后延迟、目标并发和峰值资源；当前 CLI 尚未
自动执行 warm-up 或并发压测。

### 7.1 结果解读

本发布版不附开发工作站历史报告，也不从旧 corpus 的成绩推出当前版本效果。
BM25 是默认配置；Dense / Hybrid 为显式可选项。fixture 仅供工程回归，不能替代独立、
经过审核的相关性数据。比较性能需固定语料、模型、配置与硬件，并保留失败结果。

## 8. 本地检查

```bash
python scripts/build_rag_index.py --dry-run
python scripts/evaluate_rag_retrieval.py \
  --suite-config evaluation/retrieval/smoke_v5/config.json \
  --modes bm25 \
  --output <external-runtime>/evaluation/rag-smoke.json
pytest tests/test_retrieval.py tests/test_knowledge.py \
  tests/test_rag_index_build.py tests/test_rag_retrieval_evaluation.py
```

第一条只验证 corpus/config/identity；第二条是真实 BM25 检索评测；测试中的 fake embedding/Qdrant 只
验证契约，不代表真实 BGE-M3 性能。真实 Dense/Hybrid 需要另外保存模型/索引身份和运行报告。

无模型回归另用已安装的真实 qdrant-client 与合成二维向量验证来源过滤先于截断、跨实例并发串行、
异常释放/再次查询、关闭等待与 generation 切换。这些用例不下载或加载 embedding 模型，也不构成
真实 Dense 质量或负载测试。

## 9. 评测台账与报告保留

`evaluate_rag_retrieval.py` 在开始评测前建立独立运行目录
`<output-parent>/evaluation-runs/rag-<uuid>/`，保存 `experiment.sqlite3`、不可覆盖的
`report.json` / `report.md`。默认中央台账为 `<output-parent>/experiment-ledger.sqlite3`；
可用 `--ledger <external-runtime>/experiment_ledger.sqlite3` 将多次评测写入同一台账。
中央台账与单轮事件库必须在同一文件系统，才能用 SQLite 附加数据库事务原子追加。
台账与报告应使用仓库外的运行目录。

台账保存配置全文、输入文件哈希、suite 假设/seed、源码身份、指标、运行环境、耗时与显存信息。
RAG 的 `split_hash` 标明是固定评测 queries 的哈希，不冒充训练划分。
没有测到的字段记录原因。正常完成、检索失败、配置/发布异常及 KeyboardInterrupt 都有终态事件；
底层审计目录本身不可写，或输出要求修改既有实验目录时，会拒绝执行并报告错误。

退出码维持 0=执行完成、1=模式内检索失败、2=配置或报告写入失败；质量分数仍为描述性指标。
中断会先记录 interrupted 再向调用方传播。低分、失败及中断记录不会自动删除。

`--output` 与 `--markdown-output` 是便于查看的命名报告；不用 `--force` 时拒绝覆盖。
使用 `--force` 时，先把旧文件原样归档至新轮目录的 `previous_reports/`，再发布新文件。
命名报告不可指向中央台账、事件库或任一已登记实验目录。无论发布是否成功，已生成的本轮
报告、路径与哈希会保留在运行目录/失败事件中。JSON stdout 的 `experiment` 给出固定运行目录。
