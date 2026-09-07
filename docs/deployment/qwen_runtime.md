# 旧 Qwen 运行时兼容说明

当前默认语言层是 [MedGemma 1.5 4B](medgemma_runtime.md)。
本页保留旧入口的定位说明，不再提供 Qwen 权重转换与研究复现流程。

`scripts/bootstrap_qwen.py` 的部分名称来自旧运行时。Linux 部署仍使用其中的
`register-runtime` 子命令登记固定 llama.cpp bundle；完整步骤见
[MedGemma 的 Linux 部署](medgemma_runtime.md#linux)。

既有 Qwen 配置和协议 fixture 仅供显式兼容回归，不代表 MedGemma 已通过同样的真实运行检查。
此发布版的视觉模型通过 [独立推理包](inference_models.md) 准备，与 Qwen 无关。
