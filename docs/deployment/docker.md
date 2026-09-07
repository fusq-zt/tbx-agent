# Docker 部署

The Compose file has no implicit service and no mock-inference profile. This
prevents a base Python image with no model mounts from being mistaken for a
working rank03 deployment. Maintainers can still build an image without running
it using `docker build --target api .`.

## Real rank03 + external OpenAI-compatible llama.cpp profile

先完成 [真实模型准备](inference_models.md)。本页是进阶容器配置：主机路径与容器内路径不同，
不能直接复制原生运行的环境变量后认为模型已经挂载。MedSAM 默认关闭。

Run the host model bootstrap first, then provide absolute host paths in the
shell or a local, uncommitted Compose `.env`:

```text
TBX_MODEL_CACHE=<external-registered-artifact-root>
TBX_DFINE_ROOT=<external-artifact-root>/sources/D-FINE
TBX_RANK03_RUNTIME_CONFIG_HOST=<external-runtime-root>/config/rank03_runtime.json
TBX_LLM_RUNTIME_CONFIG_HOST=<external-runtime-root>/config/llm_runtime.generated.yaml
TBX_LLAMA_BUNDLE_DIR=<external-runtime-root>/llama.cpp/b10517/bin
TBX_LLAMA_BUNDLE_MANIFEST_HOST=<external-runtime-root>/provenance/llama-b10517-bundle.json
TBX_LLAMA_SERVER_BINARY_NAME=llama-server
LLAMA_CPP_API_KEY_FILE_HOST=<protected-llama-key-file>
LLAMA_CPP_MODEL_SHA256=<verified-64-hex-digest>
TBX_EXTERNAL_LLM_URL=https://host.docker.internal:11435
```

Then select the profile explicitly:

```bash
docker compose --profile rank03 up --build
```

Compose uses required-variable interpolation for all ten external runtime
values above. A missing value therefore fails during `docker compose config`
instead of creating a local directory in place of a checkpoint, injecting an
empty model digest, or starting a misleading partial service. Validate the
fully resolved configuration before the first build:

```bash
docker compose --profile rank03 config --quiet
```

The artifact root is mounted read-only at `/models`, the user-generated rank03
contract at `/run/tbx/rank03_runtime.json`, and D-FINE at `/opt/dfine`. The
generated llama.cpp contract, complete native bundle, bundle manifest and API
key are separate read-only mounts; application state uses a Docker volume. This
is deliberate: `/readyz` hashes the local GGUF, executable and every manifest
entry before it trusts the external inference endpoint. Mounting only a GGUF and
URL cannot pass readiness. No model, dataset, key or uploaded image enters the
image build context. The profile does not download TBX11K or rank03 weights and
cannot start until the installed inference-bundle contract and exact registered bytes are
present.

`TBX_ARTIFACT_ROOT=/models` is exclusively the immutable model cache. Mutable
uploads and reports use `TBX_AGENT_CASE_ARTIFACT_ROOT=/data/cases`; this
separation prevents an assessment from trying to write into a read-only model
mount and prevents case data from being mixed with downloadable weights.

The image runs as UID/GID `10001`. On Linux, grant that identity read/traverse
access to the model, D-FINE and llama bundle paths, and read access to each
mounted contract/key file (for example with a narrowly scoped POSIX ACL). Do
not make the API key world-readable to work around a permissions error. A
permission failure is expected to keep `/readyz` closed.

Run exactly one API container against each `/data` volume/SQLite database. The
optional anatomy worker is an in-process thread pool: on startup it atomically
marks prior `pending`/`running` jobs as `technical_failure` with
`worker_interrupted`; it does not resume them. Horizontal replicas require a
leased shared queue and an external transactional state service before they can
be treated as highly available.

Both containers use a read-only root filesystem, drop Linux capabilities, set
`no-new-privileges`, and receive a bounded `noexec` temporary filesystem. Only
the API's named `/data` volume is writable for SQLite and governed runtime
state. The default image installs a wheel rather than an editable checkout and
runs as the fixed unprivileged identity.

`TBX_LLAMA_SERVER_BINARY_NAME` is the direct filename recorded in the generated
runtime contract (`llama-server` on a native Linux build; it may be
`llama-server.exe` when the reviewed endpoint is hosted by Windows). The whole
directory named by `TBX_LLAMA_BUNDLE_DIR` must match the generated bundle
manifest. The container only reads these bytes for attestation; it does not
execute a host-platform binary.

API and UI ports are published on host loopback only. Do not remove the
`127.0.0.1` bindings unless a reviewed TLS/authenticating reverse proxy is in
front of both services; the research profile is not a LAN-facing deployment.

The default base image is deliberately CPU-only. Real GPU inference requires a
project-reviewed CUDA/PyTorch base and matching install source, supplied through
`TBX_CUDA_BASE_IMAGE`; the `gpus: all` reservation must then be enabled and
validated on the target host. The repository does not label an untested CUDA
combination as production-ready.

A custom base must already provide compatible Python 3.11–3.13 plus pip and
either Debian `adduser`/`addgroup` or shadow-utils `useradd`/`groupadd`. Supplying
an arbitrary CUDA image is not sufficient. Pin the selected image by digest in
the deployment environment and record its driver/PyTorch compatibility test.

The real image installs the optional `dicom` extra by default. Its decoder is
portable Python plus NumPy/pydicom and intentionally accepts only uncompressed,
single-frame monochrome CR/DX objects. Compressed transfer syntaxes, multiframe
objects and other modalities fail closed; installing additional pixel-data
plugins does not expand that reviewed input contract.

The external endpoint is not an arbitrary chat API. The current application
contract verifies a local copy of the GGUF plus the served alias and llama.cpp
build fingerprint. A non-loopback endpoint is accepted only over HTTPS, so the
host service or a private gateway must present a certificate trusted by the API
container. Plain `http://host.docker.internal` is intentionally rejected.
`TBX_AGENT_LLAMA_CPP_ALLOW_REMOTE=true` permits this explicitly configured TLS
hop; it does not authorize exposing llama.cpp to a LAN or the Internet. Keep it
behind the host/container boundary and a strong independent credential.

The API container healthcheck calls `/readyz`, so `healthy` means the configured
rank03 and MedGemma contracts completed startup verification, not merely that an
HTTP process exists. Before accepting traffic, also inspect the endpoint from
the host:

```bash
curl --fail http://127.0.0.1:8000/readyz
```

A successful readiness check is engineering evidence for the configured model
bytes only; it is not clinical validation.
