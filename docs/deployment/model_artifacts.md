# 模型工件与目录

所有权重、GGUF、下载缓存和外部推理源码均位于 Git 目录之外。
`TBX_ARTIFACT_ROOT` 是模型/源码缓存，不是病例存储；病例默认写入
`TBX_AGENT_DATA_ROOT/cases`，也可用 `TBX_AGENT_CASE_ARTIFACT_ROOT` 单独指定。

## 工件入口

| 工件 | 安装入口 |
| --- | --- |
| ConvNeXt-Tiny + D-FINE | [视觉推理包安装器](inference_models.md) |
| 固定 D-FINE 推理源码 | `python scripts/bootstrap_dfine.py --cache-dir <external-artifact-root>` |
| PSPNet | 模型清单中的 `anatomy` 组 |
| MedGemma GGUF + llama.cpp | [MedGemma setup](medgemma_runtime.md) |
| MedSAM | 可选 `medsam_refinement` 组，默认关闭 |
| BGE-M3 | 部署者自备固定本地快照，详见 [RAG](../retrieval.md) |

分类/检测包独立发布，不存在“下载通用初始化权重即可替代”的路径。
安装器生成新的外部运行契约；旧模板或不匹配的权重会被拒绝。

## 官方 PSPNet 下载

使用 [快速部署](quickstart.md) 配置的外部目录。以下是 Linux 示例；
PowerShell 将 `"$TBX_ARTIFACT_ROOT"` 换成 `$env:TBX_ARTIFACT_ROOT`：

```bash
python scripts/bootstrap_models.py list
python scripts/bootstrap_models.py --cache-dir "$TBX_ARTIFACT_ROOT" dry-run anatomy
python scripts/bootstrap_models.py --cache-dir "$TBX_ARTIFACT_ROOT" download anatomy
python scripts/bootstrap_models.py --cache-dir "$TBX_ARTIFACT_ROOT" verify anatomy
```

全局 `--cache-dir`、`--json` 必须放在子命令之前。
下载器使用文件锁、续传、有限重试、精确大小/SHA-256 检查与原子安装。
来源见 [configs/model_sources.yaml](../../configs/model_sources.yaml)。

## 可选 MedSAM

仅在需要轮廓可视化时安装 `medsam` extra 并显式下载 `medsam_refinement`。
模型、配置和处理器固定到同一 revision，全部核验后才反序列化。
只安装依赖或权重不会开启功能，参见 [MedSAM 边界](../medsam_refinement.md)。

## 更新与信任

`artifact://` 路径相对于模型根解析并拒绝目录穿越。
模型、配置或策略字节变化应产生新身份，不继承旧性能声明。
下载校验只证明文件符合清单，不证明来源许可、模型有效性或临床用途已获批准。
密钥只通过环境或受保护文件提供，不进入源码包。
