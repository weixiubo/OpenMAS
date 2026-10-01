"""Bootstrap a tiny offline GPT-2 style model for local-provider smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CORPUS = [
    "You are a concise assistant.",
    "Summarize the task and provide one clear answer.",
    "Use tools when needed and verify arithmetic carefully.",
    "Given a prompt, reason step by step and output a short response.",
    "Multi-agent coordination improves robustness under uncertainty.",
    "Runtime reconfiguration should be bounded by cooldown and budget.",
]


def _ensure_deps() -> tuple[object, object, object]:
    try:
        from tokenizers import ByteLevelBPETokenizer  # type: ignore
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "Missing dependencies. Install with: pip install torch transformers tokenizers"
        ) from exc
    return ByteLevelBPETokenizer, GPT2TokenizerFast, (GPT2Config, GPT2LMHeadModel)


def build_tiny_model(
    *,
    output_dir: Path,
    vocab_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    n_positions: int,
    corpus_path: Path | None,
) -> dict[str, object]:
    ByteLevelBPETokenizer, GPT2TokenizerFast, model_apis = _ensure_deps()
    GPT2Config, GPT2LMHeadModel = model_apis

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = output_dir / "corpus.txt"
    if corpus_path is not None and corpus_path.exists():
        corpus_file.write_text(corpus_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        corpus_file.write_text("\n".join(DEFAULT_CORPUS) + "\n", encoding="utf-8")

    tokenizer_trainer = ByteLevelBPETokenizer()
    tokenizer_trainer.train(
        files=[str(corpus_file)],
        vocab_size=max(256, vocab_size),
        min_frequency=1,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
    )
    tokenizer_trainer.save_model(str(output_dir))

    tokenizer = GPT2TokenizerFast(
        vocab_file=str(output_dir / "vocab.json"),
        merges_file=str(output_dir / "merges.txt"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(str(output_dir))
    token_count = len(tokenizer)

    config = GPT2Config(
        vocab_size=token_count,
        n_positions=max(256, n_positions),
        n_ctx=max(256, n_positions),
        n_layer=max(1, n_layer),
        n_head=max(1, n_head),
        n_embd=max(32, n_embd),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(str(output_dir))

    metadata = {
        "model_path": str(output_dir.resolve()),
        "vocab_size": token_count,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "n_positions": config.n_positions,
    }
    (output_dir / "bootstrap_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a tiny offline GPT-style model for local runs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-positions", type=int, default=1024)
    parser.add_argument("--corpus-path", default="")
    args = parser.parse_args()

    metadata = build_tiny_model(
        output_dir=Path(args.output_dir).resolve(),
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_positions=args.n_positions,
        corpus_path=Path(args.corpus_path).resolve() if args.corpus_path.strip() else None,
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
