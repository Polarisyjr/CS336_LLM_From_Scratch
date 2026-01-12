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
    vocab_path: str,
    merges_path: str,
):
    """
    Save BPE vocab and merges to separate JSON files.
    bytes are encoded as hex strings.
    """
    # Save vocab: {str(int): hex_string}
    vocab_obj = {str(i): b.hex() for i, b in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_obj, f, ensure_ascii=False)
    
    # Save merges: [[hex_string, hex_string], ...]
    merges_obj = [[a.hex(), b.hex()] for (a, b) in merges]
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_obj, f, ensure_ascii=False)


def load_bpe_json(vocab_path: str, merges_path: str):
    """
    Load BPE vocab and merges from separate JSON files.
    """
    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_obj = json.load(f)
    vocab = {int(i): bytes.fromhex(b) for i, b in vocab_obj.items()}
    
    # Load merges
    with open(merges_path, "r", encoding="utf-8") as f:
        merges_obj = json.load(f)
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_obj]

    return vocab, merges


# ============================================================
# Experiment runner
# ============================================================

def run_bpe_experiment(
    *,
    input_path: str,
    vocab_size: int,
    out_vocab_path: str,
    out_merges_path: str,
    special_tokens: List[str],
    do_profile: bool = False,
):
    """
    Run BPE training, save to JSON, and print stats.
    """
    print("=" * 72)
    print(f"Training BPE on {input_path}")
    print(f"vocab_size = {vocab_size}")
    print("=" * 72)

    profiler = None
    if do_profile:
        profiler = cProfile.Profile()
        profiler.enable()

    t0 = time.perf_counter()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    t1 = time.perf_counter()

    if do_profile:
        profiler.disable()

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    save_bpe_json(
        vocab=vocab,
        merges=merges,
        vocab_path=out_vocab_path,
        merges_path=out_merges_path,
    )

    longest = max(vocab.values(), key=len)

    print("-" * 72)
    print(f"Elapsed time: {(t1 - t0) / 60:.2f} minutes")
    print(f"Peak memory:  ~{peak_kb / 1024 / 1024:.2f} GB")
    print(f"Longest token length: {len(longest)} bytes")
    print("Longest token preview:")
    print(f"  '{longest.decode('utf-8', errors='replace')[:200]}'")
    print("-" * 72)
    print()

    if do_profile:
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(30)


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
    MODE = "train"  # "train" or "valid"
    DO_TINYSTORIES = False
    DO_OWT = True
    DO_PROFILE = True
    TINYSTORIES_PATH = f"../data/TinyStoriesV2-GPT4-{MODE}.txt"
    OWT_PATH = f"../data/owt_{MODE}.txt"

    OUT_TINYSTORIES_VOCAB = f"tinystories_bpe_10k_{MODE}_vocab.json"
    OUT_TINYSTORIES_MERGES = f"tinystories_bpe_10k_{MODE}_merges.json"
    OUT_OWT_VOCAB = f"owt_bpe_32k_{MODE}_vocab.json"
    OUT_OWT_MERGES = f"owt_bpe_32k_{MODE}_merges.json"

    SPECIAL_TOKENS = ["<|endoftext|>"]

    # ----------------------------
    # Run experiments
    # ----------------------------
    if DO_TINYSTORIES:
        run_bpe_experiment(
            input_path=TINYSTORIES_PATH,
            vocab_size=10_000,
            out_vocab_path=OUT_TINYSTORIES_VOCAB,
            out_merges_path=OUT_TINYSTORIES_MERGES,
            special_tokens=SPECIAL_TOKENS,
            do_profile=DO_PROFILE,
        )

    if DO_OWT:
        run_bpe_experiment(
            input_path=OWT_PATH,
            vocab_size=32_000,
            out_vocab_path=OUT_OWT_VOCAB,
            out_merges_path=OUT_OWT_MERGES,
            special_tokens=SPECIAL_TOKENS,
            do_profile=DO_PROFILE,
        )

    # ----------------------------
    # Uncomment to profile TinyStories
    # ----------------------------
    # if DO_PROFILE:
    #     profile_tinystories(TINYSTORIES_PATH)
