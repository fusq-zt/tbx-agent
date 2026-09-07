# 可选 MedSAM 候选框轮廓细化

MedSAM 细化是 D-FINE 候选框之后的可选可视化分支。它不重新分类、不投票、不改变
`healthy / sick_non_tb / tb` 原生 argmax，也不把候选框变成“病灶确诊”或“精确病灶边界”。所有
结构化结果固定带 `routing_effect: none`、`clinical_validation: false` 和
`visualization_only_nonvalidated_contour`。

**默认关闭。** 本分支使用 `wanglab/medsam-vit-base` 固定工件与本项目的受限适配器。

## 数据流与证据语义

```text
D-FINE 源图 xyxy 框 ─┐
                     ├─ MedSAM box prompt ─ 原始候选 mask ─ 与左右肺野并集相交 ─ QC
PSPNet 左右肺野掩膜 ─┘                                                   │
                                                   refined / outside_lungs / mask_qc_failed
```

- box prompt 与 `VisionEvidence.detections` 保持原顺序，并记录 box digest；
- 按 D-FINE 原顺序最多处理前 24 个提示，每批最多 4 个，限制内存峰值；其余框仍一一保留为
  `capacity_abstained`，不静默丢框或伪造 mask；
- PSPNet QC 失败时整次细化失败关闭，不生成部分轮廓；
- 框与肺野完全无交叠时不调用 MedSAM，显式返回 `outside_lungs`；
- 推理 mask 必须与肺野并集相交，并通过最小像素数、相对框面积和框内交叠 QC；
- 单框 QC 不通过时返回 `mask_qc_failed` 且不暴露部分 mask；
- 依赖、工件哈希、模型加载、输出形状或身份契约失败会使 anatomy run 进入
  `completed_with_refinement_failure`：已完成的 PSPNet 肺野、确定性空间摘要和 boundary.png 保留，
  MedSAM 证据为空，原病例和 rank03 分流保持不变。PSPNet 主分支自身失败才是
  `technical_failure`。

轮廓是“给定 D-FINE 框后模型建议的可视区域”，不是影像征象、病原学证据、病灶类别、活动性、
传染性或耐药性。D-FINE 没有候选框时也不会凭空生成轮廓。

## 固定工件与安装

可选依赖独立于默认 PSPNet 环境。先按 [快速部署](deployment/quickstart.md) 设置外部目录；
以下为 Linux 命令，PowerShell 使用 `$env:TBX_ARTIFACT_ROOT`：

```bash
python -m pip install -e ".[anatomy,vision,medsam]"
python scripts/bootstrap_models.py \
  --cache-dir "$TBX_ARTIFACT_ROOT" download anatomy medsam_refinement
python scripts/bootstrap_models.py \
  --cache-dir "$TBX_ARTIFACT_ROOT" verify anatomy medsam_refinement
```

Docker 的默认 real image 只安装 rank03、PSPNet 与 DICOM 运行依赖；要显式启用该可选分支，构建前
设置 `TBX_REAL_INSTALL_EXTRAS=vision,anatomy,dicom,medsam`，并确保同一只读
`TBX_MODEL_CACHE` 已下载、校验上述 `medsam_refinement` 组。未安装依赖或缺少任一固定工件时必须
报告技术失败，不能退回伪造轮廓。

随后显式设置 `TBX_AGENT_CONTOUR_REFINEMENT_BACKEND=medsam_hf`；设备可用
`TBX_AGENT_CONTOUR_REFINEMENT_DEVICE=auto|cpu|cuda` 选择。只安装 extra 或下载权重不会自动开启
该分支。

`configs/model_sources.yaml` 固定 `wanglab/medsam-vit-base` revision
`de8488bca37bb1d4fb190f612c516126d739ce3b`。模型权重为 375,045,749 bytes，SHA256 为
`0ef67838fba16f16c0e308eaf7a3ae6fbf4fe8e13d7a120ca932671ed9c47db0`；配套
`config.json` 和 `preprocessor_config.json` 也分别固定字节数与哈希。所有文件在反序列化前验证；
源码包和 Git 均不包含权重。

运行时配置默认 `contour_refinement_backend: none`。只有显式设为 `medsam_hf` 才加载该分支；启用后
不能回退到模拟 mask 或无提示自动分割。概率、mask QC、24 个提示上限和 4 个提示批大小都进入
`refinement_generation_key`，策略变化不会复用旧证据。

## 验证边界

官方 MedSAM 是通用医疗图像的 box-prompt 模型。本项目尚无 TBX11K 像素级病灶轮廓真值，不能报告
Dice、IoU、边界误差或临床分割性能，也不能称为“精确病灶分割”。发布前的真实 forward smoke 只
验证固定工件能加载、D-FINE box prompt 能产生正确形状的源坐标 mask、肺野约束与失败关闭工作；
它不是模型性能或临床有效性评测。

若要升级为经验证的病灶轮廓功能，需要独立、专家标注且不参与模型选择的像素级测试集，预注册
Dice/IoU/HD95、按图像来源和病变大小分层的失效分析，并冻结新的模型、预处理、提示和 QC 策略
身份。该评测不得使用 TBX11K 锁定测试集做阈值或策略选择。
