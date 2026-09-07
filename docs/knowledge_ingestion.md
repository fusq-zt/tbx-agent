# 指南摄取与知识发布

本模块把指南文件转换为可审计、可重跑的中间语料，但**不会自动把原始抽取结果发布到线上 RAG**。这是一个刻意设置的生产安全边界：PDF/HTML 抽取完成后，仍须经过医学审核、适用范围标注、冲突检查和批准，才能进入现有 `knowledge/source_manifest.json` 与 `knowledge/chunks.jsonl`。

## 数据流与信任边界

```text
本地 PDF / HTML / Markdown
          │  SHA-256 校验；禁止网络获取
          ▼
Canonical Markdown
  标题层级、段落、列表、表格、页码
          │  章节优先 + 长度边界
          ▼
ingested_chunks.jsonl
  稳定 chunk_id、内容哈希、来源定位
          │
          ├── manifest.json（确定性构建清单）
          └── receipt.json（运行状态与抽取器审计）
          │
          ▼
医学审核 / claim scope / 时效与冲突审核
          │
          ▼
已批准的线上知识快照（另一个显式发布步骤）
```

摄取输出统一带有：

- `review_status: pending_medical_review`
- `retrievable: false`
- `deployment_gate.eligible: false`
- `deployment_gate.automatic_promotion: false`

因此，运行摄取命令本身不会改变 Agent 的医疗回答依据。

## 支持格式

| 输入 | 转换方式 | 结构保留 | 生产约束 |
| --- | --- | --- | --- |
| Markdown | UTF-8 严格解码 | ATX 标题、段落、列表、表格、代码块、显式页码标记 | 非 UTF-8 直接失败 |
| HTML | BeautifulSoup 清理后由 markdownify 转换 | 标题、段落、列表、表格、链接 | 不执行脚本，不下载图片或外链资源 |
| PDF | 首选 pdfplumber；可显式选择 pypdf | pdfplumber 保留表格及页码；pypdf 保留可验证文本及页码 | 扫描页默认 fail-closed；pypdf 不承诺表格还原 |

PDF 还必须处理阅读顺序。`pdf_layout_mode` 支持 `single_column`、`two_column`
和 `auto`；对已知双栏期刊应显式锁定 `two_column`。实现根据页面中部栏间距识别
双栏纵向区域，先输出完整左栏再输出右栏，并在 receipt 中逐页写入“必须视觉复核”
警告。仅仅成功提取文字不代表版面顺序正确；发布审核需将 Canonical Markdown 与
渲染页并排抽查，栏顺序、图表、上下标或公式异常均应阻止上线。

HTML 与 PDF 解析器属于离线构建依赖，目前不加入 Agent 在线服务的最小依赖集合。请在隔离的知识构建环境中，从受控镜像或内部制品库安装：

```powershell
python -m pip install -e ".[ingestion]"
```

默认配置显式使用 `pdfplumber` 且开启 `require_pdf_table_detection`，避免在依赖缺失时悄悄退化为丢失表格结构的抽取。探索性构建可以选择 `pdf_engine: auto`，但回执会记录 pypdf 回退警告，且该构建不应直接进入审核发布流程。

## Canonical Markdown 约定

每份文档转换成 UTF-8、LF 换行、稳定空行的 Markdown。文档标题作为一级标题。PDF 页边界使用内部标记：

```markdown
<!-- tbx:page=12 -->
```

标记不会作为正文块写入 chunk，但会变成 `page_start`、`page_end` 和 `locator`。结构解析得到以下块类型：

- `heading`：保留 1-6 级标题，并为后续块维护完整 `heading_path`；
- `paragraph`：连续正文行；
- `list`：连续有序或无序列表及缩进续行；
- `table`：Markdown 表头、分隔行与数据行；
- `code`：围栏代码块，通常只用于技术附录。

PDF 的标题识别采用保守规则，只把明确的“第……章/节”、中文编号或多级数字编号识别为标题。无法可靠判断的行保留为普通段落，不根据字体或语义猜测标题。

## 扫描 PDF 与 OCR

当某页几乎没有可提取文本且包含图像时，默认抛出 `ScannedPdfError`，整个来源构建失败并写入失败回执。系统不会：

- 把文件名、邻页文字或模型生成内容填成该页正文；
- 静默跳过疑似扫描页后宣称抽取完成；
- 自动调用云端 OCR 服务。

如确需 OCR，调用方必须显式实现本地 `OcrAdapter`，提供适配器名称和按页抽取方法，再通过 `run_ingestion(..., ocr_adapter=...)` 注入。返回文本必须通过非空长度校验；使用页码、适配器名称和 `ocr_used` 会写入回执。OCR 输出仍处于待审核状态。

纯空白页不会被制造文本；它会产生警告。若整份文档没有足够的可验证文本，构建仍会失败。

## Chunk 策略

配置使用字符边界而非依赖特定 tokenizer，以保证构建环境变化时结果稳定：

```yaml
chunking:
  min_characters: 240
  target_characters: 700
  max_characters: 1100
  overlap_blocks: 1
```

处理顺序如下：

1. 新章节标题优先开启新 chunk，即使上一段未达到目标长度；
2. 在段落、列表、表格等块边界优先合并到 `target_characters`；
3. 超长段落按可观察的句末标点切分；超长列表按行切分；
4. 超长表格重复表头后按行切分；单行或表头超过硬上限则失败，不截断；
5. 不超过 `max_characters` 时，可从同一章节复制最多 `overlap_blocks` 个块；
6. 每个 chunk 记录章节路径、块序号范围、页码范围和人类可读 locator。

`chunk_id` 由 `source_id + section_path + content_sha256` 生成，并只在完全相同内容发生碰撞时添加确定性的出现序号。修改无关章节不会使未变化 chunk 的 ID 全部漂移。每条记录同时保存：

- 原文件 `source_sha256`；
- Canonical Markdown `canonical_markdown_sha256`；
- chunk 正文 `content_sha256`；
- `block_start` / `block_end`；
- `page_start` / `page_end`；
- `section_path` / `block_types` / `locator`。

这些字段可用于后续向量库的幂等 upsert、删除旧版本、引用回链和审核差异比较。

## 配置与运行

生产骨架配置位于 `configs/knowledge_ingestion.yaml`。输入必须是本地文件路径；`http://`、`https://`、对象存储 URI 和 UNC 网络路径都会在打开输入前被拒绝。`url` 字段只是引用元数据，绝不会被抓取。

配置支持给每个来源设置 `expected_sha256`。如果文件内容与锁定哈希不同，来源立即失败，从而避免网页下载版本被静默替换。

```powershell
$env:PYTHONPATH = "src"
python -m tbx_agent.ingestion --config configs/knowledge_ingestion.yaml
```

大型构建产物应写入仓库外的 `TBX_AGENT_DATA_ROOT/knowledge_ingestion`：

```text
knowledge_ingestion/
├── receipts/
│   └── <run_id>.json
└── snapshots/
    └── <snapshot_id>/
        ├── canonical/<source_id>.md
        ├── ingested_chunks.jsonl
        └── manifest.json
```

`build_id` 由流水线版本、配置哈希和所有启用来源的内容哈希计算；`snapshot_id` 还绑定 chunk、Canonical Markdown 哈希以及实际抽取器版本。相同输入、配置和抽取器重复执行会指向同一快照，manifest 与 chunks 字节完全相同。每次执行另有唯一 `run_id` 和独立 receipt，历史运行不会被覆盖。运行时间只记录在 receipt，不污染确定性 manifest。

任何失败（缺文件、哈希不符、扫描页、依赖缺失、抽取错误、chunk 越界）都会写 `status: failed` 的回执，且不会产生一个看似可用的 manifest。

## 审核与向量化交接

`ingested_chunks.jsonl` 是与向量后端无关的中间格式。后续无论使用 BGE 系列 embedding、FAISS、Qdrant、pgvector 或其他方案，都应把以下步骤作为独立、版本化的发布流水线：

1. 医学人员逐来源确认版本、章节、表格和公式没有抽取遗漏；
2. 为每个 chunk 标注可回答的 `claim_scope`、适用人群、地区、时效和证据等级；
3. 排除旧方案、具体个体化剂量、新闻转述和未获全文的来源；
4. 对重复与冲突条款建立显式优先级，不依赖向量相似度决定规范效力；
5. 冻结 embedding 模型名称、revision、归一化方式、向量维度和索引参数；
6. 建立检索回归集，评测 Recall@k、nDCG、引用精确率、拒答率及错误来源率；
7. 审批后才生成现有在线 retriever 可读取的已审核快照。

向量库不应成为规范内容的唯一存储：Canonical Markdown、manifest、chunk JSONL 和内容哈希始终是可恢复的事实来源。

## 验证

摄取测试只使用临时合成 Markdown、HTML 和伪 PDF，不接触 TBX11K locked local test 或 hidden test：

```powershell
python -m pytest tests/test_ingestion.py -q
python -m ruff check src/tbx_agent/ingestion tests/test_ingestion.py
```

测试覆盖结构和页码保留、chunk 硬边界与稳定 ID、相同输入重跑、来源变更、远程路径拒绝、缺失可选依赖、扫描 PDF fail-closed、显式 OCR 适配器及失败回执。
