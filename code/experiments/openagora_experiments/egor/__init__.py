"""Fresh, staged evidence-gap organization repair experiments."""

from openagora_experiments.egor.parsing import parse_baseline_output, parse_specialist_output, parse_synthesis_output
from openagora_experiments.egor.prompts import build_baseline_prompt, build_specialist_prompt, build_synthesis_prompt
from openagora_experiments.egor.task_model import PilotTask

__all__ = [
    "PilotTask",
    "build_baseline_prompt",
    "build_specialist_prompt",
    "build_synthesis_prompt",
    "parse_baseline_output",
    "parse_specialist_output",
    "parse_synthesis_output",
]
