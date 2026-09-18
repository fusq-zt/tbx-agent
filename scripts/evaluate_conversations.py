"""Real local LLM + isolated synthetic CV + local retrieval: end-to-end dialog checks.

No model plans or answers are scripted. Expected tools/text are independent assertions.
Results include all failures. Text assertions are smoke checks, not a quality score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from tbx_agent.config import Settings, load_yaml
from tbx_agent.evaluation.conversation_fixtures import (
    ConversationAnatomy,
    ConversationVision,
    synthetic_png,
)
from tbx_agent.narrator import LlamaCppNarrator
from tbx_agent.plan_react import PLAN_REACT_POLICY_ID
from tbx_agent.service import TBXAgentService


def _safe_config(value):
    """Retain effective configuration without storing credentials or secret-file paths."""
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if any(
                term in str(key).lower() for term in ("api_key", "secret", "password", "token")
            ) else _safe_config(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class CountingGenerator:
    """Count public generation API attempts, including failed/cached native attempts.

    These are not HTTP request counts: the provider may reject a cached unsupported
    native call locally. Do not store prompts, responses, exception text or credentials.
    """

    def __init__(self, generator):
        self.generator = generator
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.generator, name)

    def _call(self, method, *args, **kwargs):
        entry = {"method": method, "success": False}
        entry.update({key: kwargs[key] for key in ("schema_name", "seed", "max_tokens")
                      if key in kwargs})
        self.calls.append(entry)
        started = time.monotonic()
        try:
            result = getattr(self.generator, method)(*args, **kwargs)
            entry["success"] = True
            if isinstance(result, tuple) and isinstance(result[-1], dict):
                entry["usage"] = {key: result[-1][key] for key in (
                    "prompt_tokens", "completion_tokens",
                ) if key in result[-1]}
            return result
        except Exception as exc:
            entry["error_type"] = type(exc).__name__
            raise
        finally:
            entry["seconds"] = round(time.monotonic() - started, 3)

    def complete_structured(self, **kwargs):
        return self._call("complete_structured", **kwargs)

    def complete_tool_calls(self, **kwargs):
        return self._call("complete_tool_calls", **kwargs)

    def narrate(self, response):
        return self._call("narrate", response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path,
                        default=Path("evaluation/suites/conversation_v3/dialogues.json"))
    parser.add_argument("--output-label", default="conversations_v3",
                        help="One directory name under the runtime evaluation directory.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.output_label):
        parser.error("--output-label must be a simple directory name, not a path")
    settings = Settings.from_env()
    generator = CountingGenerator(LlamaCppNarrator(
        model_alias=settings.llama_cpp_model_alias, model_path=settings.llama_cpp_model_path,
        expected_model_sha256=settings.llama_cpp_model_sha256,
        expected_server_build=settings.llama_cpp_server_build, base_url=settings.llama_cpp_base_url,
        api_key=settings.resolved_llama_cpp_api_key(),
        timeout_seconds=settings.llama_cpp_timeout_seconds,
        max_response_bytes=settings.llama_cpp_max_response_bytes,
        allow_remote=settings.llama_cpp_allow_remote,
    ))
    root = settings.data_root / "evaluation" / args.output_label / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "hypothesis": (
            "ReAct-first decisions with optional planning resolve current requests, preserve "
            "cached evidence and permissions, and avoid unrelated or explicitly prohibited tools."
        ),
        "scope": "Real local LLM + synthetic visual geometry + real RAG; not CV accuracy.",
        "created_at": datetime.now(UTC).isoformat(), "model": generator.model,
        "policy": PLAN_REACT_POLICY_ID,
        "model_digest": settings.llama_cpp_model_sha256,
        "fixture_path": str(args.scenarios),
        "split": "Frozen synthetic software scenarios; no medical dataset split is used.",
        "seed": 20260901, "temperature": 0, "peak_vram": None,
        "peak_vram_note": "External model process; not instrumented.",
        "fixture_sha256": hashlib.sha256(args.scenarios.read_bytes()).hexdigest(),
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "source_hashes": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in sorted(
                {path.as_posix() for path in Path("src").rglob("*.py")}
                | {"scripts/evaluate_conversations.py", "ui/streamlit_app.py"}
            ) if Path(name).is_file()
        },
        "generation_count_definition": (
            "Public generator API attempts, including locally rejected native attempts. "
            "Not HTTP requests or guaranteed model generations; inspect success and usage."
        ),
        "base_config": _safe_config(asdict(settings)), "episode_configs": {}, "rows": [],
    }
    started = time.monotonic()
    for episode in json.loads(args.scenarios.read_text(encoding="utf-8")):
        episode_root = root / episode["id"]
        config = replace(
            settings, data_root=episode_root, db_path=episode_root / "state.sqlite3",
            artifact_root=episode_root / "cases",
            vision_backend="mock", narrator_backend="none", require_real_inference=False,
            require_llm_inference=False, anatomy_backend="xrv_pspnet", anatomy_required=False,
            # Keep the deployment's step/tool/cost budgets. Larger exploration
            # budgets must not silently improve the acceptance measurement.
            contour_refinement_backend="none",
        )
        report["episode_configs"][episode["id"]] = {
            "settings": _safe_config(asdict(config)),
            "fusion_policy": _safe_config(config.fusion_policy()),
            "rank03_runtime": _safe_config(config.rank03_config()),
            "retrieval": _safe_config(load_yaml(config.retrieval_config_path)),
            "synthetic_fixture": {key: value for key, value in episode.items() if key != "turns"},
        }
        vision = ConversationVision(config.fusion_policy(), config.rank03_config(),
                                    healthy=episode.get("healthy", False),
                                    empty=episode.get("empty", False))
        anatomy = ConversationAnatomy(fail=episode.get("anatomy_fail", False))
        service = TBXAgentService(config, vision_backend=vision, anatomy_backend=anatomy)
        case = None
        thread_id = episode["id"]
        try:
            for index, turn in enumerate(episode["turns"]):
                if turn.get("new_thread"):
                    thread_id = f"{episode['id']}-{index}"
                if turn.get("upload") or (index == 0 and episode.get("image", True)):
                    case = service.assess_cxr(
                        synthetic_png((48 + index, 68, 88)), user_id="conversation-probe",
                        owner_scope="tenant:conversation-probe", consent_to_process=True,
                        attested_chest_radiograph=True,
                    )[0]
                tick = time.monotonic()
                call_start = len(generator.calls)
                row = {"episode": episode["id"], "turn": index, **turn}
                try:
                    result = service.respond_with_controller(
                        message=turn["query"], case_id=case.case_id if case else None,
                        thread_id=thread_id, user_id="conversation-probe",
                        owner_scope="tenant:conversation-probe", generator=generator,
                        narrator_override=generator,
                    )
                    response = result.response.model_dump(mode="json")
                    text = response["summary"] + "\n" + "\n".join(response["visual_evidence_notes"])
                    tools = result.execution_plan["tool_names"]
                    row.update(
                        tools=tools, tools_passed=tools == turn["tools"],
                        text_checks_passed=(
                            all(value in text for value in turn.get("contains", []))
                            and all(value not in text for value in turn.get("excludes", []))
                            and (not turn.get("contains_any") or any(
                                value in text for value in turn["contains_any"]
                            ))
                        ), response=response, execution_plan=result.execution_plan,
                        terminal=result.trace.terminal.reason_code,
                    )
                except Exception as exc:
                    row.update(tools_passed=False, text_checks_passed=False,
                               error_type=type(exc).__name__)
                row["seconds"] = round(time.monotonic() - tick, 3)
                row["generation_calls"] = generator.calls[call_start:]
                row["generation_api_attempts"] = len(row["generation_calls"])
                row["generation_api_successes"] = sum(
                    entry["success"] for entry in row["generation_calls"]
                )
                report["rows"].append(row)
                report["runtime_seconds"] = round(time.monotonic() - started, 3)
                report["generation_api_attempts"] = len(generator.calls)
                report["generation_api_successes"] = sum(
                    entry["success"] for entry in generator.calls
                )
                report["tools_passed"] = sum(item["tools_passed"] for item in report["rows"])
                report["joint_smoke_checks_passed"] = sum(
                    item["tools_passed"] and item["text_checks_passed"] for item in report["rows"]
                )
                report["total"] = len(report["rows"])
                (root / "results.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                printed = {key: row.get(key) for key in (
                    "episode", "turn", "query", "tools", "tools_passed", "text_checks_passed",
                    "error_type", "seconds",
                    "generation_api_attempts", "generation_api_successes",
                )} | {"answer": row.get("response", {}).get("summary")}
                print(json.dumps(printed, ensure_ascii=False), flush=True)
        finally:
            service.close()
    print(str(root / "results.json"), flush=True)


if __name__ == "__main__":
    main()
