# 第三方来源说明

本文件记录本项目使用或可选接入的第三方来源，不替代上游许可证、模型条款或版权声明，
也不构成统一的再分发许可。项目自身代码的发布许可证尚待维护者确定。

| 组件 | 来源 | 本仓库的处理 |
| --- | --- | --- |
| D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | 固定推理源码下载到外部目录 |
| ConvNeXt / timm | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | Python 依赖；独立分类权重不随源码分发 |
| TorchXRayVision PSPNet | [mlmed/torchxrayvision](https://github.com/mlmed/torchxrayvision) | 官方 Release 工件，外部下载与哈希核验 |
| MedGemma 基础模型 | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) | 遵守模型使用条款与可能的访问要求 |
| MedGemma GGUF | [unsloth/medgemma-1.5-4b-it-GGUF](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF) | 固定第三方量化版本，不称为官方 GGUF |
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | 固定源码或运行时 bundle，仓库外准备 |
| BGE-M3 | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 可选本地向量模型，默认不加载 |
| MedSAM | [wanglab/medsam-vit-base](https://huggingface.co/wanglab/medsam-vit-base) | 可选轮廓可视化，默认关闭 |

模型工件的固定 revision、大小、SHA-256 和已记录来源信息见
[模型清单](configs/model_sources.yaml)、[语言运行时](configs/llm_runtime.yaml) 与
[MedGemma 部署](docs/deployment/medgemma_runtime.md)。
下载前应查看相应上游条款；不能由“可下载”推定“可公开再分发”。

独立分类/检测 ZIP 的发布权限应由维护者另行确认。源码公开不自动授予权重或原训练数据的访问、
使用或再分发权。该源码包不包含训练数据、权重或原始指南全文。

指南知识的来源与引用定位记录在 [source_manifest.json](knowledge/source_manifest.json)；
片段用于受限检索，不能把软件中的审核元数据当作医学专家签署或版权授权。
新增、镜像或修改任何第三方资源时，应保留原有声明并记录确切来源。
