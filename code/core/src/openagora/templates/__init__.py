"""Template system entrypoints."""

from openagora.templates.loader import load_template_bundle
from openagora.templates.runtime import TemplateRuntime, build_template_runtime

__all__ = ["TemplateRuntime", "build_template_runtime", "load_template_bundle"]
