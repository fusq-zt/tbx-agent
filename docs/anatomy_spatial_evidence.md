# 肺野分割与确定性空间证据

该能力把 D-FINE 候选框与 ChestX-Det PSPNet 的左右肺野掩膜放在同一原图像素坐标系中，生成可审计的二维空间关系。它是可选的复核与可视化工具，`routing_effect` 固定为 `none`，不能改变 rank03 三分类原生 argmax 分流。

## 运行链路

1. 胸片主流程先完成输入校验、rank03 三分类和 D-FINE 候选框生成并保存不可变病例。
2. 客户端为该病例创建 anatomy run；服务用受限线程池异步运行 PSPNet，失败不会修改病例或原分流。
3. PSPNet 输出恢复到源图尺寸，分别编码左右肺野掩膜，并执行面积、左右比例、掩膜交叠、边界接触和左右次序检查。
4. 只有分割 QC 没有失败时，确定性引擎才计算候选框与每侧肺野的像素交叠，并按该侧肺野纵向范围划分上、中、下三段。每侧 RLE 在单次 run 内只解码一次，再用确定性积分图完成全部框的矩形交叠查询。
5. 若显式启用 MedSAM，D-FINE 源图坐标框作为 box prompt；结果与 PSPNet 左右肺野并集相交并
   通过固定 QC 后，才保存为未经验证的可视化轮廓。该可选分支失败不会丢弃已完成的肺野或空间摘要。
6. 服务持久化原始结构化位置、定位策略版本以及代码生成的安全摘要。前端叠加肺野边界、可选轮廓，
   并以表格显示候选框、左右侧、二维肺野区和框内交叠比例。
7. 用户请求解释该 exact case 时，最新证据可用的空间摘要作为 `visual_evidence_notes` 进入 MedGemma 的
   approved-fact allowlist。MedGemma 只能选择代码已生成的字符串，拿不到肺野/MedSAM 掩膜、轮廓 RLE、
   候选框坐标、图像哈希或任意自由定位权。

上、中、下是二维肺野分区，不是肺叶。候选框与肺野掩膜相交也不是影像征象、病灶、活动性、传染性、菌阳性或耐药判断。

## 结构化契约

完成的 `AnatomyRunRecord` 包含：

- `evidence.masks`：左右肺野各一个 `tbx-rle-v1` 掩膜，尺寸必须等于源图；
- `evidence.qc`：`pass`、`warning` 或 `fail`，附稳定代码与有限数值指标；
- `detector_locations`：与 D-FINE 候选框一一对应；
- `localization_policy_id` 与 `localization_minimum_box_overlap_fraction`；
- `spatial_summary`：候选框总数、已定位数、肺野外/未定位数、因分割无效而弃权数，以及代码生成的陈述 allowlist；
- `routing_effect: none` 与 `clinical_validation: false`。

单个位置状态只有三种：

- `localized`：包含一侧或双侧 `assignments`，每项给出主要二维肺野区、三区交叠分布、交叠像素数、框内交叠比例和占该侧肺野比例；
- `outside_lungs`：没有达到最小框内肺野交叠门槛。这里只能显示“肺野外/未定位候选”，不能称为肺外病变；
- `invalid_anatomy`：肺野分割 QC 失败，禁止输出左右或上/中/下位置结论。

与冻结 D-FINE 的 300 object-query 合同一致，`VisionEvidence.detections` 和空间引擎批处理都硬限最多 300 个候选框；更长输入被视为不可能的后端合同偏离并在解码肺野前拒绝。为限制语言模型上下文和前端噪声，结构化数组保留该合同内的全部位置，代码文本层最多逐一展开 24 个候选框，剩余数量会明确记录。D-FINE 没有返回候选框时仍生成一条明确的不可排除说明。

## API

创建并轮询：

```text
POST /v1/cases/{case_id}/anatomy-runs
GET  /v1/cases/{case_id}/anatomy-runs/{run_id}
GET  /v1/cases/{case_id}/anatomy-runs/{run_id}/boundary.png
GET  /v1/cases/{case_id}/anatomy-runs/{run_id}/contours.png
```

所有请求都需要与病例完全一致的 `owner_scope + user_id`。未保留源图时，创建请求必须重新上传与病例哈希、宽高完全一致的输入。边界端点可用 `structure=combined|left_lung|right_lung`；返回透明 PNG，并带 `X-Anatomy-Routing-Effect: none`。轮廓端点仅在 `refinement_status=completed` 时可用，并带 `X-Refinement-Routing-Effect: none` 与 `X-Clinical-Validation: false`。

## 失败与验证边界

- PSPNet 依赖或权重不可用、分割身份不符、分割证据漂移、并发版本冲突和未知主 worker 异常进入
  `technical_failure`，不暴露部分肺野掩膜。
- MedSAM 独立分支的依赖、权重、加载或推理契约失败进入
  `completed_with_refinement_failure`：肺野、空间摘要和 boundary.png 仍可用，轮廓证据为空且带稳定
  `refinement_error_code`。该降级 run 不作为可复用 generation，修复后可重试。
- 服务启动会在一个 SQLite 事务中把上个进程遗留的 `pending/running` run 标记为 `technical_failure`，错误代码为 `worker_interrupted`，并记录系统审计事件；不会让中断任务永久悬挂。
- 同一病例、图像、模型、预处理和分割策略的有效 run 可复用；失败 run 不复用。
- 该恢复语义基于“每个 SQLite 数据库只运行一个 API 进程”的当前部署约束；多副本部署必须改用带租约的共享队列，不能共享此数据库并宣称 worker 高可用。
- 当前 QC 是工程启发式检查，不是诊断质量评价，也没有外部临床验证。
- PSPNet 与 rank03/D-FINE 的组合尚未完成前瞻性临床验证、人因验证或监管评估。

定向回归覆盖 `tests/test_anatomy_core.py`、`tests/test_anatomy_integration.py`、`tests/test_ui_anatomy_client.py`、`tests/test_streamlit_ui.py` 和 `tests/test_narrator.py`。
