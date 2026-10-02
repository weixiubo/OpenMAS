"""Validate a scheduler action catalog independently of inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openagora.mechanisms.evidence_gap_repair import EvidenceType
from pydantic import BaseModel, ConfigDict, Field


class Action(BaseModel):
    evidence_type: EvidenceType


class Template(BaseModel):
    action_ids: list[str]


class ActionCatalog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    actions: dict[str, Action] = Field(default_factory=dict)
    templates: dict[str, Template] = Field(default_factory=dict)

    def scheduler_records(self) -> list[dict[str, Any]]:
        result = []
        for name, template in sorted(self.templates.items()):
            unknown = set(template.action_ids) - self.actions.keys()
            if unknown:
                raise ValueError(f"Template {name} references unknown actions: {sorted(unknown)}")
            result.append(
                {
                    "template_id": name,
                    "action_ids": template.action_ids,
                    "evidence_types": [self.actions[key].evidence_type.value for key in template.action_ids],
                }
            )
        return result


def load_scheduler_catalog(path: Path) -> list[dict[str, Any]]:
    return ActionCatalog.model_validate(json.loads(path.read_text())).scheduler_records()
