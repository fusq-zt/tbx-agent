# 指南证据审计摘要

审计日期：2026-09-01。正式运行以 `knowledge/source_manifest.json` 的 snapshot 与哈希为准。

## 保留与优先级

1. `WS 288—2017`：中国诊断原则、术语和证据链。
2. 《肺结核主动筛查循证指南》（2026）：主动筛查对象、年龄/风险分支与转诊路径。
3. WHO *Module 3: Diagnosis, fourth edition*（2025）：低复杂度自动 NAAT、快速利福平及其他耐药检测。
4. WHO *Module 4: treatment and care*（2025）：仅使用已审核的 WHO 发布页和 TB Knowledge
   Sharing Platform 药物敏感性肺结核章节，回答一般治疗选项、疗程边界、方案选择因素与治疗
   支持/监测；不提供剂量或个体方案。
5. WHO *Operational handbook, Module 4: treatment and care*（2025）：作为独立来源，仅使用
   TBKSP Chapter 3 §5.1 与官方 PDF 第 236—240 页已审核内容，解释多数患者的门诊/去中心化照护、
   可能需要住院的情况，以及医学安全时尽早衔接门诊；不作个人入院或出院决定。
6. 《中国结核病防治工作技术指南》（2021）：报告、定点机构转诊和基层管理；临床条款在新来源
   冲突时降权。

有限使用：

- 《肺结核基层诊疗指南（2018年）》2019 年发表：只作基层沟通/转诊解释；不采用抗体规则、旧复治
  或注射剂方案。
- ATS/CDC/IDSA 2017：只作美国高资源低发病环境的补充解释，不能覆盖中国/WHO 优先来源。
- 美国 CDC 2024—2025 官方页面：`Clinical and Laboratory Diagnosis`、`Tuberculosis in Pregnancy`、
  `Exposure to Tuberculosis`、`Treating Tuberculosis`、`Adverse Events During TB Treatment` 和
  `Preventing Tuberculosis` 均标记为 `US`、`included_limited`、`supplemental`。只回答已审核小节中的
  检测角色与阴性边界、孕期评估、密接/感染状态、单次漏服、不良反应、传染性复评和返工返校；
  不替代中国/WHO路径，不授权个体治疗方案或用药调整。
- 《肺结核影像诊断标准》（2021）：用户提供的正式 PDF 共 6 页，已逐页渲染审核；只用于影像
  作用、非特异性、鉴别背景和确诊边界，不生成个体影像诊断。

## 剔除或用户跳过

- 2012 肺结核门诊诊疗规范：引用已被替代的 WS 288-2008 并含旧耐药方案，完全不检索。
- 2020 基层合理用药指南：含旧复治、注射剂和具体剂量，完全不检索。
- QQ 新闻：只用于发现正式指南，完全不作为医学证据。
- 中国 MDR/RR-TB 2026 正式指南：用户明确要求跳过；正式全文未审核，既有摘要级片段也已移除。

## 固化的负向规则

- TST/IGRA 反映结核感染，不能区分潜伏感染与活动性结核。
- 结核抗体不得作为活动性肺结核规则。
- 涂片阴性不能排除；涂片阳性不能单独确认结核分枝杆菌。
- Xpert/NAAT 阴性不能单独排除且不替代培养；培养是实验室确认金标准，但培养阴性仍不一定排除。
- “胸片正常可排除”的 CDC 表述只适用于感染检测阳性且无症状的特定美国场景，并有免疫抑制例外；
  不得外推为有症状或仍有临床怀疑者的通用排除规则。
- CXR/CT/CAD/rank03 不能确认活动性、传染性、菌阳性或耐药；筛查未标记也不等于排除。
- Agent 不赋予用户“疑似/临床诊断/确诊”等专业病例分类标签。
- 可解释 WHO 当前药物敏感性肺结核的一般 6 个月疗程及特定人群 4 个月疗程选项；不生成
  个体化方案、剂量、个人疗程或启停/换药建议。
- 查询涉及耐药、MDR/RR-TB、XDR-TB 时，药物敏感性治疗片段不得参与回答；当前快照因用户
  跳过中国 MDR/RR-TB 2026 指南而返回证据缺口。
- WHO 对 MDR-TB“主要采用门诊而非住院照护”的正式推荐为有条件推荐、证据确定性极低；系统不把
  该推荐外推为所有药物敏感性肺结核的同等级推荐，只引用操作手册面向所有结核患者的实施说明。

## 本次来源变更

- 影像标准 PDF：`knowledge/sources/china_tb_imaging_standard_2021.pdf`，SHA256
  `1ba4807eae2a5ee0e5c6f2085a53a19d0235dc090707907d6dfd0fec6c80368c`，审核页 1—6。
- 新知识快照：`tbx-guidelines-curated-2026-09-01-v6`。
- 新增 WHO *Module 4: treatment and care* 受限来源。`full_text_reviewed=false`：仅审核 WHO
  发布页与 TBKSP node 3019、2963、3021；线上 chunk 均为这些章节的中文审核摘要。
- 新增独立 WHO *Operational handbook, Module 4* 受限来源。`full_text_reviewed=false`：仅审核
  TBKSP node 3047 §5.1 和官方 PDF 第 236—240 页；三个照护场景 chunk 与药物方案 chunk 通过
  claim scope 硬隔离。
- 新增六个 CDC 受限补充来源。`full_text_reviewed=false`：manifest 逐项记录已审核页面和小节，
  所有来源 `treatment_details_allowed=false`。新增 19 问 `cdc_supplemental_v1` 固定检索回归集，
  专门检查 Xpert、涂片、培养、TST/IGRA、孕期、密接、漏服、不良反应与传染性相关实体不被替换。
- 修正主动筛查重点人群条款中的年龄摄取错误：以指南表1的“老年人（年龄≥65岁）”
  和摘要列出的四类优先筛查对象为准；不再保留“15岁及以上老年人”这一错误表述。
- 被跳过的 MDR/RR-TB 来源保留在 manifest 中作为审计记录，但 `retrievable=false`，不含任何 chunk。

## 急症规则的来源边界

用户提供的十份文件没有统一的公众急诊分诊表。系统中的大量/持续咯血、严重呼吸困难、发绀、
意识异常等规则是隔离于指南 RAG 的 `local_clinical_safety_policy`。其保守目的合理，但生产使用前仍
须急诊/呼吸/结核临床专家签署；系统不会用主动筛查指南 citation 冒充其直接依据。
