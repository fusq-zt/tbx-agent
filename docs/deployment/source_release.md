# 干净源码发布

源码包从显式 allowlist 构建，不把目录中的所有文件直接归档。
发布内容为推理代码、部署配置、文档、软件测试与经选择的知识/评测 fixture。

## 检查与构建

```bash
python scripts/build_source_release.py --dry-run
python scripts/check_release.py --root .
python scripts/build_source_release.py
python scripts/check_release.py --zip dist/tbx-agent-source.zip --manifest dist/tbx-agent-source.manifest.json
```

Git 候选检查核对 tracked 与未忽略 untracked 文件，要求和公开 allowlist 一致。
新增文档、配置或测试需同时更新 allowlist。
ZIP 使用排序路径、固定时间戳、统一权限和固定根前缀；相邻 manifest 记录逐文件大小与 SHA-256、
源码树 SHA-256 和归档 SHA-256。

## 排除内容

训练代码、数据集、权重、优化器状态、患者影像、数据库、缓存、凭据、运行报告和大日志不进入源码包。
原始指南 PDF / HTML 也不随源码分发；使用者需依法取得，放在外部目录。

构建器与检查器还拒绝敏感令牌或私钥文本、工作站用户路径、目录穿越、大小写重复路径、符号链接、
加密 ZIP 项目和非普通文件；默认限制单文件 5 MiB、源码树 50 MiB。
测试和 fixture 是软件回归材料，不是患者数据或模型临床验证。

`dist/` 是生成输出，不提交到 Git。独立视觉 ZIP 属于单独的模型发布工件，
由 [视觉安装器](inference_models.md) 处理；它不应放进源码归档。
正式上传与模型托管见 [发布说明](publishing.md)。
