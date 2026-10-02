"""Configuration package."""

from openagora.config.loader import load_run_config
from openagora.config.models import RunConfig

__all__ = ["RunConfig", "load_run_config"]
