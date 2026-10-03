# Third-Party Notes

This repository is released under Apache-2.0. Third-party dependencies, models, datasets, and materials remain subject to their own licenses and terms.

## Python Dependencies

- Runtime dependencies are declared in [code/core/pyproject.toml](code/core/pyproject.toml).
- Experiment dependencies are declared in [code/experiments/pyproject.toml](code/experiments/pyproject.toml).
- Downstream users are responsible for reviewing the licenses of installed Python packages.

## Models

- Optional local-model execution may use third-party model weights such as `Qwen2.5-*`.
- Model weights are not redistributed in this repository.
- Users are responsible for obtaining model access and complying with the corresponding model licenses and usage policies.

## Data And Artifacts

- The public code sync does not ship the private paper replay bundle or manuscript artifact snapshots.
- Any benchmark data, evaluation manifests, or replay bundles must be handled under their own terms and are outside the public repository contract unless explicitly released in a later artifact package.

## Private Mother Repo Materials

The local mother repo may also contain:

- `paper/` manuscript sources
- `resource/` collected templates, PDFs, and reference material
- internal rerun summaries and freeze notes

Those assets are not part of the planned public GitHub sync and should not be treated as Apache-2.0-covered public deliverables by default.
