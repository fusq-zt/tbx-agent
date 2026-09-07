# GitHub 与权重发布

仓库代码、独立模型包与第三方模型分别管理。本页给出发布方案，不表示已完成公开发布或许可审批。

## 当前状态

- 视觉包 `tbx-rank03-inference-v1.zip` 已准备，大小与 SHA-256 见 [模型部署](inference_models.md)。
- 公开下载地址尚未配置；不要把占位地址视为可用链接。
- 源码许可证尚待维护者决定。模型与数据来源的条款单独适用。
- 私有仓库及其 Release 仅供有访问权限的账号下载；默认匿名 HTTPS 安装不负责 GitHub 私有登录，
  可在浏览器登录下载后使用 `--bundle` 安装。

## 推荐托管方式

| 方式 | 适用情况 | 本项目建议 |
| --- | --- | --- |
| GitHub Release | 同仓库版本化分发小于单文件限制的二进制附件 | 视觉 ZIP 约 209.42 MiB，可与源码版本对应 |
| Hugging Face 模型仓库 | 较大模型、模型卡、固定 revision、访问申请 | 后续需要独立模型发布与门控时考虑 |
| Google Cloud Storage | 私有对象与限时下载控制 | 需要云项目、权限和运维，当前无需额外引入 |

GitHub 官方当前规定：每个 Release 附件必须 **小于 2 GiB**，单 Release 最多 1,000 个附件，
不限制该 Release 总大小或下载带宽。因此本视觉 ZIP 适合此路径。
[GitHub Releases 文档](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

MedGemma GGUF 为 2,489,894,976 bytes，超过上述单附件门槛；本项目继续引用上游固定版本，
不重复托管。Hugging Face 提供模型仓库和访问申请机制，但存储额度与使用规则需按账号类型核对。
[存储规则](https://huggingface.co/docs/hub/storage-limits) ·
[模型门控](https://huggingface.co/docs/hub/models-gated)

GCS signed URL 让持有链接的人在限定时间访问对象，适合需要临时授权的交付。
不要把带签名的临时 URL、云凭据或访问令牌写入源码。
[Google Cloud Storage signed URL](https://cloud.google.com/storage/docs/access-control/signed-urls)

## 发布清单

1. 确定 GitHub 仓库、可见性、源码许可证和视觉权重的分发依据。
2. 运行 [源码检查](source_release.md)，仅上传推理发布目录。
3. 为源码创建对应版本标签，先准备 Release 草稿。
4. 以独立附件上传视觉 ZIP 和 SHA-256 清单，说明模型身份、输入尺寸、用途限制及是否经过真实运行验证。
5. 从有权限的新环境下载附件，校验 SHA-256，验证安装器生成的契约。
6. 确认可见性与访问方式后，在 README / 模型部署页写入真实下载链接，再完成发布。

GitHub 的 Release 以 Git tag 对应源码版本，可先保存草稿并添加二进制附件。
[管理 Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)

发布说明不得将张量一致性检查、Demo 或软件测试写成重新测得的模型性能。
第三方模型仍由上游入口准备，出处与独立条款见 [第三方来源](../../THIRD_PARTY_NOTICES.md)。
