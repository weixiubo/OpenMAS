# Migration

OpenMAS / OpenAGORA / EGOR now uses this directory as its repository root. The former `当前/`
layer has been removed. Run commands from the repository root in its own
virtual environment. Existing import names and documented research arguments
remain supported unless a specific migration note below says otherwise.

Historical Git repositories, staged/unstaged changes and unique research
material are stored in the local sibling `_research_archive` area. They are
not needed to install the public package. The private recovery inventory maps
old paths to their preserved content. Frozen evidence and source-bound
manifests retain their original hashes; this refactor has a new source identity.

`SOURCE_PROVENANCE.json` identifies the baseline. Public reproduction material
is described in `reproduce/README.md`. Missing external data must be provided
explicitly; an absent asset is not substituted with a synthetic success.

Experiment JSON paths now resolve relative to the experiment working directory
(`code/experiments`); historical analysis CLI root flags remain available.
Model examples use upstream model identifiers instead of another user's cache.
Set the model path to a locally downloaded snapshot for an offline run. The
standard Hugging Face cache can be configured with `HF_HOME`; credentials belong
in the process environment, never in this repository. These portable examples
are new configurations, not the old source-bound server launch records.
