"""Experiment suites."""

from openagora_experiments.suites.gaia_mini import run_gaia_mini_ablation
from openagora_experiments.suites.qa_mini import run_qa_mini_ablation

__all__ = ["run_qa_mini_ablation", "run_gaia_mini_ablation"]
