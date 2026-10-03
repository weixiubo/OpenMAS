"""Domain models for OpenAGORA."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CoordinationMechanism(str, Enum):
    DIRECT_SUPERVISION = "direct_supervision"
    MUTUAL_ADJUSTMENT = "mutual_adjustment"
    PROCESS_STANDARDIZATION = "process_standardization"
    OUTPUT_STANDARDIZATION = "output_standardization"
    SKILL_STANDARDIZATION = "skill_standardization"


class TopologyType(str, Enum):
    STAR = "star"
    PIPELINE = "pipeline"
    TREE = "tree"
    PARTIAL_MESH = "partial_mesh"
    DENSE = "dense"


class TaskSignature(BaseModel):
    """Task-family signature from organizational theory dimensions."""

    model_config = ConfigDict(extra="forbid")
    primary_type: str = "general_reasoning"
    uncertainty: Literal["low", "medium", "high"] = "medium"
    complexity: Literal["low", "medium", "high"] = "medium"
    interdependence: Literal["pooled", "sequential", "reciprocal"] = "pooled"


class TaskSpec(BaseModel):
    """User task with optional precomputed signature."""

    model_config = ConfigDict(extra="forbid")
    task_id: str
    prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    signature: TaskSignature | None = None


class RoleSpec(BaseModel):
    """Role assignment inside architecture R."""

    model_config = ConfigDict(extra="forbid")
    worker_id: int
    role_name: str
    role_description: str
    capabilities_required: list[str] = Field(default_factory=list)


class CommunicationTopology(BaseModel):
    """Topology G."""

    model_config = ConfigDict(extra="forbid")
    type: TopologyType
    edges: list[tuple[str, str]] = Field(default_factory=list)
    broadcast_channels: list[str] = Field(default_factory=list)


class CoordinationSpec(BaseModel):
    """Coordination mechanism C."""

    model_config = ConfigDict(extra="forbid")
    mechanism: CoordinationMechanism
    execution_order: Literal["sequential", "parallel", "conditional"] = "sequential"
    sync_points: list[str] = Field(default_factory=list)


class ArchitectureSpec(BaseModel):
    """Architecture tuple A=(R,G,C)."""

    model_config = ConfigDict(extra="forbid")
    roles: list[RoleSpec]
    communication_topology: CommunicationTopology
    coordination: CoordinationSpec
    rationale: str = ""


class WorkerFeedback(BaseModel):
    """Structured worker feedback used in negotiation loop."""

    model_config = ConfigDict(extra="forbid")
    worker_id: int
    overall_assessment: Literal["agree", "conditional_agree", "disagree"]
    role_fit: Literal["good_fit", "acceptable", "poor_fit"] = "acceptable"
    concerns: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    suggestion: str = ""


class ProgressEvaluation(BaseModel):
    """Runtime monitoring result."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["on_track", "minor_issue", "architecture_mismatch"]
    diagnosis: dict[str, Any] = Field(default_factory=dict)
    recommended_action: str = ""


class OTAlignmentResult(BaseModel):
    """Theory alignment score for architecture and task signature."""

    model_config = ConfigDict(extra="forbid")
    predicted_config: str
    actual_config: str
    score: float


class CacheEntry(BaseModel):
    """Current cache view for one task type."""

    model_config = ConfigDict(extra="forbid")
    task_type: str
    version_id: int
    signature: TaskSignature
    architecture: ArchitectureSpec
    avg_perf: float
    sample_count: int
    ot_alignment: float
    needs_revalidation: bool = False


class RunEvent(BaseModel):
    """Structured event emitted by engine components."""

    model_config = ConfigDict(extra="forbid")
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "v2"
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RunManifest(BaseModel):
    """Reproducibility metadata for a run."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    run_id: str
    created_at: str
    git_commit: str
    dependency_lock_hash: str
    config_hash: str
    seed: int
    provider_name: str
    provider_mode: Literal["mock", "live", "local"] = "mock"
    model_name: str
    temperature: float
    data_version: str
    template_id: str = "default"
    template_version: str = "v1"
    event_schema_version: str = "v2"


class RunResult(BaseModel):
    """Final run output used by API, CLI and replay."""

    model_config = ConfigDict(extra="forbid")
    run_id: str
    task_id: str
    final_architecture: ArchitectureSpec
    result: dict[str, Any]
    ot_alignment: OTAlignmentResult
    metrics: dict[str, float]
    cache_used: bool
    negotiation_rounds: int
    reconfig_count: int
    override_used: bool
    deadlock_breaker: bool
    manifest_path: str
    events_path: str
