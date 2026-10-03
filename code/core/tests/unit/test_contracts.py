from __future__ import annotations

from openagora.utils.contracts import check_public_contract


def test_public_contract_stable() -> None:
    violations = check_public_contract()
    assert violations == []
