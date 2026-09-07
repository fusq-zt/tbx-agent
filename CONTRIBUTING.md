# 参与维护

本仓库维护推理、部署、指南检索与软件契约。提交应围绕明确问题，说明触发条件、改变后的行为与验证方式。
模型训练和数据集构建不属于此发布版范围。

## 开发检查

在项目根目录的虚拟环境中运行：

```bash
python -m pip install -e ".[ui,dev]"
python -m ruff check src ui tests scripts
python -m pytest -m "not real_rank03 and not requires_model and not slow"
python scripts/build_source_release.py --dry-run
python scripts/check_release.py --root .
```

普通测试使用合成输入、临时目录与替身，不应触发模型下载或真实训练。
真实模型测试必须显式选择，并在仓库外准备工件。报告应区分软件契约、真实运行和数据集性能。

## 提交内容

不要提交权重、GGUF、数据集、患者影像、凭据、数据库、缓存、生成报告或大日志。
模型身份变化需更新对应清单和哈希契约；旧权重的结果不能用于新的模型。
新增公开文件须同步维护源码发布 allowlist，并通过发布检查。

修改工具行为时，保留公开参数边界、真实回执、失败状态和证据适用范围。
修改面向用户的医学表述时，需提供精确可审查的指南引用，并保留辅助研究用途边界。
不要把二维肺野区写成肺叶，也不要把模拟结果写成模型实测。

代码许可证尚待维护者确定。提交第三方代码或资源前应记录来源与条款，不能以仓库公开为由推定授权。
