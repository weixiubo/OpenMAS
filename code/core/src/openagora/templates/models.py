"""Template registry models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PromptTemplate(BaseModel):
    """One prompt template pair and schema."""

    model_config = ConfigDict(extra="forbid")
    system_prompt: str
    user_prompt: str
    required_variables: list[str] = Field(default_factory=list)


class PersonaDefaults(BaseModel):
    """Default persona pools for assignment strategies."""

    model_config = ConfigDict(extra="forbid")
    fixed_personas: list[str] = Field(default_factory=list)
    persona_pool: list[str] = Field(default_factory=list)


class TemplateBundle(BaseModel):
    """Versioned template bundle."""

    model_config = ConfigDict(extra="forbid")
    template_id: str
    version: str
    schema_version: str = "v1"
    persona_defaults: PersonaDefaults = Field(default_factory=PersonaDefaults)
    templates: dict[str, PromptTemplate]

    def validate_required_templates(self) -> None:
        required = {
            "leader_proposal",
            "worker_feedback",
            "feedback_summary",
            "override",
            "minority_report",
        }
        missing = sorted(required - set(self.templates.keys()))
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Template bundle missing required templates: {missing_str}")
