"""OT-Alignment scorer and architecture prototypes."""

from __future__ import annotations

from openagora.types import (
    ArchitectureSpec,
    CommunicationTopology,
    CoordinationMechanism,
    CoordinationSpec,
    OTAlignmentResult,
    RoleSpec,
    TaskSignature,
    TopologyType,
)


class OTAlignmentScorer:
    """Compute task-theory and architecture alignment score."""

    _SIM: dict[str, dict[str, float]] = {
        "SS": {"SS": 1.0, "MB": 0.5, "PB": 0.3, "DF": 0.2, "Adh": 0.4},
        "MB": {"SS": 0.5, "MB": 1.0, "PB": 0.6, "DF": 0.5, "Adh": 0.2},
        "PB": {"SS": 0.3, "MB": 0.6, "PB": 1.0, "DF": 0.4, "Adh": 0.5},
        "DF": {"SS": 0.2, "MB": 0.5, "PB": 0.4, "DF": 1.0, "Adh": 0.3},
        "Adh": {"SS": 0.4, "MB": 0.2, "PB": 0.5, "DF": 0.3, "Adh": 1.0},
    }

    def predict_config(self, signature: TaskSignature) -> str:
        """Predict Mintzberg config from task signature."""
        if signature.uncertainty == "high" and signature.interdependence == "reciprocal":
            return "Adh"
        if signature.interdependence == "sequential":
            return "MB"
        if signature.complexity == "high" and signature.interdependence == "pooled":
            return "PB"
        if signature.primary_type in {"creative_writing", "general_reasoning"}:
            return "PB"
        return "SS"

    def classify_architecture(self, architecture: ArchitectureSpec) -> str:
        """Map architecture to nearest Mintzberg config."""
        mechanism = architecture.coordination.mechanism
        topology = architecture.communication_topology.type
        if mechanism == CoordinationMechanism.DIRECT_SUPERVISION and topology == TopologyType.STAR:
            return "SS"
        if mechanism == CoordinationMechanism.PROCESS_STANDARDIZATION or topology == TopologyType.PIPELINE:
            return "MB"
        if mechanism == CoordinationMechanism.SKILL_STANDARDIZATION:
            return "PB"
        if mechanism == CoordinationMechanism.OUTPUT_STANDARDIZATION:
            return "DF"
        return "Adh"

    def compute(self, signature: TaskSignature, architecture: ArchitectureSpec) -> OTAlignmentResult:
        """Compute OT-Alignment score for one task and architecture."""
        predicted = self.predict_config(signature)
        actual = self.classify_architecture(architecture)
        score = self._SIM[predicted][actual]
        return OTAlignmentResult(predicted_config=predicted, actual_config=actual, score=score)

    def prototype_architecture(self, config_name: str, worker_count: int) -> ArchitectureSpec:
        """Construct a deterministic prototype architecture by config class."""
        roles = [
            RoleSpec(
                worker_id=i + 1,
                role_name=f"worker_{i + 1}",
                role_description="generalist executor",
                capabilities_required=["reasoning", "tools"],
            )
            for i in range(worker_count)
        ]
        if config_name == "MB":
            topology = CommunicationTopology(
                type=TopologyType.PIPELINE,
                edges=[(str(i), str(i + 1)) for i in range(1, worker_count)],
                broadcast_channels=["leader"],
            )
            coordination = CoordinationSpec(
                mechanism=CoordinationMechanism.PROCESS_STANDARDIZATION,
                execution_order="sequential",
            )
            rationale = "Pipeline flow for sequential dependence."
        elif config_name == "PB":
            topology = CommunicationTopology(type=TopologyType.PARTIAL_MESH, edges=[], broadcast_channels=["leader"])
            coordination = CoordinationSpec(
                mechanism=CoordinationMechanism.SKILL_STANDARDIZATION,
                execution_order="parallel",
            )
            rationale = "Expert autonomy with standardized skill profiles."
        elif config_name == "DF":
            topology = CommunicationTopology(type=TopologyType.TREE, edges=[], broadcast_channels=["leader"])
            coordination = CoordinationSpec(
                mechanism=CoordinationMechanism.OUTPUT_STANDARDIZATION,
                execution_order="conditional",
            )
            rationale = "Divisional subgroups coordinated by output targets."
        elif config_name == "Adh":
            topology = CommunicationTopology(type=TopologyType.DENSE, edges=[], broadcast_channels=["leader"])
            coordination = CoordinationSpec(
                mechanism=CoordinationMechanism.MUTUAL_ADJUSTMENT,
                execution_order="parallel",
            )
            rationale = "Dense collaboration for uncertain reciprocal tasks."
        else:
            topology = CommunicationTopology(type=TopologyType.STAR, edges=[], broadcast_channels=["leader"])
            coordination = CoordinationSpec(
                mechanism=CoordinationMechanism.DIRECT_SUPERVISION,
                execution_order="sequential",
            )
            rationale = "Simple structure with direct supervision."
        return ArchitectureSpec(
            roles=roles,
            communication_topology=topology,
            coordination=coordination,
            rationale=rationale,
        )


class FixedPrototypeScorer(OTAlignmentScorer):
    """Internal ablation scorer with a fixed prototype prior.

    This is used only for paper-side ablations such as "NoOT", where we want to
    remove task-theory guidance without changing the public runtime contract.
    """

    def __init__(self, fixed_config: str = "SS") -> None:
        self._fixed_config = fixed_config

    def predict_config(self, signature: TaskSignature) -> str:
        del signature
        return self._fixed_config
