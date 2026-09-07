from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentQuery(APIModel):
    thread_id: str = Field(min_length=1, max_length=128)
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    owner_scope: str | None = Field(default=None, min_length=1, max_length=256)
    message: str = Field(min_length=1, max_length=4000)
    case_id: str | None = None
    # ``local_qwen`` is accepted as a wire-compatibility alias for clients
    # released before the MedGemma migration. New clients emit
    # ``local_medgemma``.
    llm_provider: Literal[
        "local_medgemma", "local_qwen", "openai_compatible"
    ] = "local_medgemma"
    llm_connection_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def selected_llm_has_the_required_connection(self):
        if self.llm_provider == "openai_compatible" and self.llm_connection_id is None:
            raise ValueError("OpenAI-compatible provider requires llm_connection_id")
        if self.llm_provider in {"local_medgemma", "local_qwen"} and (
            self.llm_connection_id is not None
        ):
            raise ValueError("local MedGemma provider must not include llm_connection_id")
        return self


class LLMConnectionCreateRequest(APIModel):
    thread_id: str = Field(min_length=1, max_length=128)
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    owner_scope: str | None = Field(default=None, min_length=1, max_length=256)
    base_url: str = Field(min_length=1, max_length=2_048)
    model: str = Field(min_length=1, max_length=256)
    api_key: SecretStr


class ScreeningStartRequest(APIModel):
    thread_id: str = Field(min_length=1, max_length=128)
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    owner_scope: str | None = Field(default=None, min_length=1, max_length=256)
    case_id: str | None = None
    consent: bool


class ScreeningAnswerRequest(APIModel):
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    owner_scope: str | None = Field(default=None, min_length=1, max_length=256)
    question_id: str = Field(min_length=1, max_length=128)
    answer: Any


class ScreeningCancelRequest(APIModel):
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    owner_scope: str | None = Field(default=None, min_length=1, max_length=256)


class ReviewCompleteRequest(APIModel):
    owner_scope: str | None = Field(default=None, min_length=1, max_length=256)
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    reviewer_id: str | None = Field(default=None, min_length=1, max_length=128)
    expected_version: int = Field(ge=1)
    decision: Literal[
        "keep_model_flagged",
        "keep_model_not_flagged",
        "indeterminate",
        "technical_repeat_required",
    ]
    note: str | None = Field(default=None, max_length=2000)


class ReportRequest(APIModel):
    owner_scope: str | None = Field(default=None, min_length=1, max_length=256)
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    actor_id: str | None = Field(default=None, min_length=1, max_length=128)
