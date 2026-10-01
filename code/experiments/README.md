# OpenAGORA experiments and EGOR

This installable package contains experiment adapters, EGOR execution,
outcome-blind overwrite policies, and statistical analysis. Install the core
package first: `pip install -e ../core`, then `pip install -e .[dev]`.

From the repository root, `python examples/egor_smoke.py` exercises the v10
two-support rule offline. `python reproduce/verify.py` verifies the sealed
800-task negative confirmation. `python scripts/run_egor_stage.py --help`
from this directory describes explicit inputs for a new experiment.

The QA and GAIA mini suites are functional fixtures. They do not establish
the paper's efficacy claims. See the root README and reproduction guide for
the current scientific status and model/data requirements.
