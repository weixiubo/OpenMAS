"""Pydantic config models for OpenAGORA runtime."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openagora.types import TaskSpec


class ProviderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "mock"
    model: str = "mock-leader-v1"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 42
    provider_mode: Literal["mock", "live", "local"] = "mock"
    timeout_s: float = Field(default=30.0, gt=0.0, le=600.0)
    max_retries: int = Field(default=2, ge=0, le=10)
    retry_backoff_base: float = Field(default=0.5, gt=0.0, le=30.0)
    max_concurrency: int = Field(default=4, ge=1, le=128)
    enable_idempotency_key: bool = True
    local_backend: Literal["transformers-local", "vllm-local"] = "transformers-local"
    local_model_path: str = ""
    local_model_profile: Literal["small", "medium", "large"] = "small"
    local_endpoint: str = "http://127.0.0.1:8000/v1"
    local_allow_download: bool = False
    max_gpu_mem_gb: float = Field(default=0.0, ge=0.0, le=96.0)
    gpu_devices: str = ""


class PathSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    db_path: str = ".openagora/openagora.db"
    artifacts_dir: str = ".openagora/artifacts"


class ThresholdSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_negotiation_rounds: int = Field(default=5, ge=1)
    max_reconfig_rounds: int = Field(default=3, ge=1)
    max_reconfigs: int = Field(default=2, ge=0)
    cooldown_steps: int = Field(default=3, ge=0)
    cache_hit_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    cache_write_perf_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    cache_write_align_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    cache_invalidate_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    feedback_stable_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    type_weight: float = Field(default=0.6, ge=0.0, le=1.0)


class ToggleSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable_negotiation: bool = True
    enable_runtime_reconfig: bool = True
    enable_cache: bool = True


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = Field(default=4, ge=1)
    worker_count: int = Field(default=4, ge=1)
    data_version: str = "qa-mini-v1"


class TemplateSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    registry_path: str | None = None
    template_id: str = "default"
    version: str = "v1"
    persona_strategy: Literal["fixed", "seeded_random"] = "fixed"
    fixed_personas: list[str] = Field(default_factory=list)
    persona_pool: list[str] = Field(default_factory=list)


class PluginSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_adapter: str = "qa_tools"
    tool_adapter: str = "local_tools"
    discovery_modules: list[str] = Field(default_factory=list)


class AuditSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_schema_version: Literal["v2"] = "v2"
    max_prompt_chars: int = Field(default=240, ge=40, le=2000)


class EvalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_path: str = "../../experiments/data/qa_mini.jsonl"
    output_csv: str = "../../experiments/results/eval_rows.csv"
    output_summary_csv: str = "../../experiments/results/eval_summary.csv"
    output_failure_csv: str = "../../experiments/results/eval_failure_cases.csv"
    output_pairwise_csv: str = "../../experiments/results/eval_pairwise_stats.csv"
    output_judge_audit_jsonl: str = "../../experiments/results/eval_judge_audit.jsonl"
    output_manifest_json: str = "../../experiments/results/eval_manifest.json"
    output_hashes_json: str = "../../experiments/results/eval_hashes.json"
    gaia_manifest_path: str = "../../experiments/data/gaia40_manifest.json"
    sample_limit: int = Field(default=40, ge=1)
    judge_mode: Literal["rule", "hybrid"] = "rule"
    judge_model: str = "gpt-4o-mini"
    borderline_low: float = Field(default=0.4, ge=0.0, le=1.0)
    borderline_high: float = Field(default=0.8, ge=0.0, le=1.0)
    stats_bootstrap_n: int = Field(default=10000, ge=100)
    stats_seed: int = 2026


class ProvenanceSettings(BaseModel):
    """Explicit source identity, independent of the caller's working directory."""

    model_config = ConfigDict(extra="forbid")
    source_revision: str = "unknown"
    dependency_lock_path: str | None = None


class RunConfig(BaseModel):
    """Top-level runtime config for engine and CLI."""

    model_config = ConfigDict(extra="forbid")
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    toggles: ToggleSettings = Field(default_factory=ToggleSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    templates: TemplateSettings = Field(default_factory=TemplateSettings)
    plugins: PluginSettings = Field(default_factory=PluginSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    provenance: ProvenanceSettings = Field(default_factory=ProvenanceSettings)
    task: TaskSpec
    eval: EvalSettings = Field(default_factory=EvalSettings)
    config_path: str | None = None

    @model_validator(mode="after")
    def check_threshold_order(self) -> "RunConfig":
        t = self.thresholds
        if t.cache_invalidate_threshold > t.cache_write_perf_threshold:
            raise ValueError("cache_invalidate_threshold must be <= cache_write_perf_threshold")
        if self.eval.borderline_low >= self.eval.borderline_high:
            raise ValueError("eval.borderline_low must be < eval.borderline_high")
        return self
