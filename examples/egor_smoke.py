"""Exercise the v10 two-support rule without a model or network connection."""
from dataclasses import asdict
import json
from openagora_experiments.egor.verification_v9 import decide_supported_blind_resolver_overwrite

def card(worker):
    return {"candidate_answer": "42", "evidence": [{
        "evidence_id": worker, "claim": "The computed answer is 42",
        "support": "6 * 7 = 42", "provenance": "constructed offline example"
    }]}

def main():
    inputs = dict(baseline_answer="41", resolver={"candidate_answer": "42",
        "transport": {"operational_complete": True}}, metric="exact_match",
        minimum_specialist_corroborators=2)
    one = decide_supported_blind_resolver_overwrite(**inputs, specialists={"a": card("a")})
    two = decide_supported_blind_resolver_overwrite(**inputs, specialists={"a": card("a"), "b": card("b")})
    if one.accept_overwrite or not two.accept_overwrite:
        raise RuntimeError("Two-support authorization regression")
    print(json.dumps({"one_support": asdict(one), "two_support": asdict(two)}, indent=2))

if __name__ == "__main__":
    main()
