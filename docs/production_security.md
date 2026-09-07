# TBX-Agent production security boundary

TBX-Agent has three explicit deployment profiles: `research` (default),
`development`, and `production`. Research/development preserve the existing
local-demo contract: callers provide `owner_scope` and `user_id` in the
request. Exact comparisons reduce accidental cross-record access, but are not
authentication and must not be internet exposed.

`production` is deliberately fail-closed. `create_app` refuses to construct a
production application unless signed trusted-proxy authentication is enabled
and its environment-only secret is at least 32 bytes. A production request to
any case, assessment, agent, screening, review, or report endpoint is rejected
before routing unless the trusted identity signature is valid and fresh.

## Required production settings

Set these values in the process environment, preferably through the platform's
secret manager:

```text
TBX_AGENT_DEPLOYMENT_PROFILE=production
TBX_AGENT_TRUSTED_PROXY_AUTH_ENABLED=true
TBX_AGENT_TRUSTED_PROXY_HMAC_SECRET=<random secret of at least 32 bytes>
```

Do not add the HMAC secret or metrics token to `configs/app.yaml`. Generate a
secret with a cryptographically secure secret-management facility. Production
startup errors and API responses expose stable blocker codes only, never the
configured value.

The following optional controls are configured in `configs/app.yaml` and can
be overridden by environment variables:

| Control | Default | Environment override |
|---|---:|---|
| Replay window | 60 s | `TBX_AGENT_TRUSTED_PROXY_REPLAY_WINDOW_SECONDS` |
| Maximum complete HTTP body | 22 MiB | `TBX_AGENT_MAX_REQUEST_BODY_BYTES` |
| In-flight protected requests | 16 | `TBX_AGENT_MAX_CONCURRENT_REQUESTS` |
| Rate | 120 requests/minute/principal | `TBX_AGENT_RATE_LIMIT_REQUESTS_PER_MINUTE` |
| Burst | 30 requests/principal | `TBX_AGENT_RATE_LIMIT_BURST` |
| HSTS response header | off | `TBX_AGENT_HSTS_ENABLED` |

The complete request-body limit must exceed the image payload limit because a
multipart envelope has additional bytes. Production rejects body-bearing
requests without an unambiguous `Content-Length` and rejects requests carrying
both `Content-Length` and `Transfer-Encoding`. Configure the same or a smaller
body limit at the reverse proxy.

## Trusted reverse-proxy identity contract

The reverse proxy is the authentication boundary. It may validate OIDC, mTLS,
or the organization's existing identity session, but TBX-Agent does not pretend
to implement an OIDC flow. The proxy must:

1. Terminate TLS and authenticate the caller.
2. Remove every inbound `X-TBX-*` header supplied by the caller.
3. Create a unique, cryptographically random nonce for every upstream request.
4. Inject exactly one value for each header below and calculate the signature.
5. Keep the proxy-to-API network private; do not expose the API port directly.

Headers:

```text
X-TBX-Tenant: clinic-a
X-TBX-User: case-subject-id
X-TBX-Actor: authenticated-clinician-id
X-TBX-Timestamp: 1787961600
X-TBX-Nonce: 128-bits-or-more-base64url
X-TBX-Signature: sha256=<64 lowercase hexadecimal characters>
```

Tenant IDs match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`. User and actor IDs also
allow `:`, `@`. The nonce is 22–128 base64url characters (at least 128 bits of
randomness). Keep proxy and API
host clocks synchronized.

The HMAC-SHA256 input is the UTF-8 encoding of these newline-separated fields,
with no trailing newline:

```text
TBX-HMAC-V1
<UPPERCASE METHOD>
<root-path + decoded ASGI path + optional '?' + raw query string>
<decimal unix timestamp>
<nonce>
<tenant id>
<user id>
<actor id>
```

The API uses constant-time signature comparison and a bounded nonce replay
cache. It derives `owner_scope` as `tenant:<tenant id>`. `X-TBX-User` is the
resource subject boundary; `X-TBX-Actor` is the authenticated operator recorded
for actions that support a separate actor. Identity fields in request bodies, forms, or query
strings are compatibility hints only: omission is allowed in production, and a
conflicting value is rejected with 403. Thus a caller cannot select its tenant
scope by editing JSON, multipart fields, or a download URL.

Every case, active-screening session, review read/list operation, and report
download is additionally bound to the signed `X-TBX-User`. The actor identifier
does not grant access to another user's case. API responses and downloadable JSON
reports remove the server-only `image_artifact_ref`; local filesystem paths are
never part of the public case representation.

### RBAC blocker and cross-subject access

The V1 signed identity contract deliberately contains no role or permission
claim. Consequently TBX-Agent does **not** infer reviewer or administrator
privilege from `X-TBX-Actor`, an actor/user mismatch, a request-body field, or a
client-supplied header. Review completion and report creation are restricted to
the signed subject's own case. Pending-review lists are filtered to that same
subject. Cross-subject clinician/reviewer workflows therefore fail closed in
production.

Before enabling such workflows, define a least-privilege RBAC contract at the
identity provider and trusted proxy, include canonical permissions in the signed
message, validate them inside `TrustedIdentity`, and add explicit authorization
tests for every case/review/report operation. Until that work is complete,
changing `allow_cross_subject` to true is prohibited. This means the HMAC profile
is a real fail-closed application boundary, but the current system is still
NO-GO for ordinary clinician/reviewer workflows that require access to another
subject's case.

### State schema V3 migration

State schema V3 stores case identity as
`(owner_scope, user_id, image_sha256)` and isolates thread and preference keys by
tenant and user. Migration backfills a case or screening subject only when the
existing persisted payload already contains an explicit matching identity.
Historical cases without that evidence keep a `NULL` subject and fail closed;
they cannot be claimed by the next caller. Legacy preferences had no tenant
field, so they are retained in `preferences_legacy_unscoped_v2` for controlled
offline reconciliation and are not exposed as current preferences. The former
global thread table is retained as `threads_legacy_global_v2`; only rows whose
payload exactly agrees with their stored tenant/user/thread identity are copied
into the scoped table.

Back up and integrity-check the database before migration. Reconciliation or
deletion of quarantined legacy tables requires a separately audited operational
procedure; application startup never silently assigns those records.

`build_signed_proxy_headers` in `tbx_agent.security.identity` is a reference
implementation for integration tests. It is not a client-side authentication
scheme and the shared secret must never be distributed to browsers or end
users.

### Current scaling boundary

The nonce cache, rate limiter, backpressure counter, and metrics are in-process.
Run one Uvicorn worker per TBX-Agent instance. Before deploying multiple workers
or replicas, enforce nonce uniqueness in a shared trusted-proxy store (or add a
shared replay backend) and aggregate rate limits/metrics at the gateway. Without
that integration, a captured request could be replayed once per independent
process during the replay window. This is a documented production blocker for
horizontal scaling, not a hidden guarantee.

## Health and observability

- `GET /livez` only confirms that the API process can answer.
- `GET /readyz` checks production security blockers and required deterministic
  components, including the configured required model runtimes. Optional and
  required language runtimes must not be treated as the same readiness contract.
- `GET /healthz` remains the backward-compatible aggregate health response.
- Every response carries a validated/generated `X-Request-ID`, `no-store`,
  `nosniff`, clickjacking, referrer, and browser-permission headers.

In-process metrics are disabled by default. Enable with
`TBX_AGENT_METRICS_ENABLED=true`. Access is allowed only from a loopback client
when `TBX_AGENT_METRICS_ALLOW_LOOPBACK=true`, or with
`Authorization: Bearer <TBX_AGENT_METRICS_ADMIN_TOKEN>`. Do not proxy `/metrics`
when relying on the loopback rule: a reverse proxy on the same host also appears
as loopback. For remote collection, disable the loopback allowance and use a
random admin token of at least 32 bytes.

Metrics contain method, route templates, status classes, aggregate latency
buckets, guard rejection counters, and explicit component fallback counters.
They never store raw URLs, case IDs, tenant/user/actor IDs, prompts, image data,
or configured secrets. The metrics and limiter state reset at process restart.

## Operational checklist

- Bind the application port to loopback/private networking and expose only the
  authenticating reverse proxy.
- Strip inbound identity headers before generating trusted values.
- Use TLS externally and enable HSTS only when clients always use HTTPS.
- Keep the HMAC secret and metrics token in a secret manager; coordinate secret
  rotation with a drained API restart because the v1 contract has one active
  secret.
- Configure proxy request-body, request-rate, connection, and timeout limits in
  addition to the in-process guards.
- Run one API worker until shared replay/rate backends are implemented.
- Do not place request/clinical bodies in access logs or external tracing.
- Treat these controls as application security, not clinical validation.

Run the focused regression suite with:

```powershell
python -m pytest tests/test_api_hardening.py tests/test_api.py
python -m ruff check src/tbx_agent/security src/tbx_agent/observability `
  src/tbx_agent/api src/tbx_agent/config.py tests/test_api_hardening.py
```
