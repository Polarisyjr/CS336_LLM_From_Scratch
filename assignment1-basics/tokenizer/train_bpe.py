"""
CS336 Assignment 1 — Section 2.5
Train byte-level BPE on TinyStories and OpenWebText,
serialize vocab + merges to JSON, and report stats.
"""

import json
import time
import resource
import cProfile
import pstats
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add parent directory to path to import from tests
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.adapters import run_train_bpe

# ============================================================
# JSON serialization utilities
# ============================================================

def save_bpe_json(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    path: str,
    special_tokens: List[str],
):
    """
    Save BPE vocab + merges to a single JSON file.
    bytes are encoded as hex strings.
    """
    obj = {
        "special_tokens": special_tokens,
        "vocab": {str(i): b.hex() for i, b in vocab.items()},
        "merges": [[a.hex(), b.hex()] for (a, b) in merges],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def load_bpe_json(path: str):
    """
    Load BPE vocab + merges from JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    vocab = {int(i): bytes.fromhex(b) for i, b in obj["vocab"].items()}
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in obj["merges"]]
    special_tokens = obj["special_tokens"]

    return vocab, merges, special_tokens


# ============================================================
# Experiment runner
# ============================================================

def run_bpe_experiment(
    *,
    input_path: str,
    vocab_size: int,
    out_json_path: str,
    special_tokens: List[str],
):
    """
    Run BPE training, save to JSON, and print stats.
    """
    print("=" * 72)
    print(f"Training BPE on {input_path}")
    print(f"vocab_size = {vocab_size}")
    print("=" * 72)

    t0 = time.perf_counter()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    t1 = time.perf_counter()

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    save_bpe_json(
        vocab=vocab,
        merges=merges,
        path=out_json_path,
        special_tokens=special_tokens,
    )

    longest = max(vocab.values(), key=len)

    print(f"Elapsed time: {(t1 - t0) / 60:.2f} minutes")
    print(f"Peak memory:  ~{peak_kb / 1024 / 1024:.2f} GB")
    print(f"Longest token length: {len(longest)} bytes")
    print("Longest token preview:")
    print(longest.decode("utf-8", errors="replace")[:200])
    print()


# ============================================================
# Optional profiling (for TinyStories part b)
# ============================================================

def profile_tinystories(input_path: str):
    profiler = cProfile.Profile()
    profiler.enable()

    run_train_bpe(
        input_path=input_path,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(30)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # ----------------------------
    # EDIT THESE PATHS
    # ----------------------------
    MODE = "valid"  # "train" or "valid"
    DO_TINYSTORIES = True
    DO_OWT = False
    DO_PROFILE = False
    TINYSTORIES_PATH = f"../data/TinyStoriesV2-GPT4-{MODE}.txt"
    OWT_PATH = f"../data/owt_{MODE}.txt"

    OUT_TINYSTORIES = f"tinystories_bpe_10k_{MODE}.json"
    OUT_OWT = f"owt_bpe_32k_{MODE}.json"

    SPECIAL_TOKENS = ["<|endoftext|>"]

    # ----------------------------
    # Run experiments
    # ----------------------------
    if DO_TINYSTORIES:
        run_bpe_experiment(
            input_path=TINYSTORIES_PATH,
            vocab_size=10_000,
            out_json_path=OUT_TINYSTORIES,
            special_tokens=SPECIAL_TOKENS,
        )

    if DO_OWT:
        run_bpe_experiment(
            input_path=OWT_PATH,
            vocab_size=32_000,
            out_json_path=OUT_OWT,
            special_tokens=SPECIAL_TOKENS,
        )

    # ----------------------------
    # Uncomment to profile TinyStories
    # ----------------------------
    if DO_PROFILE:
        profile_tinystories(TINYSTORIES_PATH)
