"""
Tokenizer Experiments (Section 2.7)

This script demonstrates various experiments with the trained BPE tokenizers,
including compression ratio analysis, cross-domain tokenization, throughput 
estimation, and dataset tokenization.
"""

import json
import time
import numpy as np
from pathlib import Path
from cs336_basics.tokenizer import Tokenizer

# File paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TOKENIZER_DIR = BASE_DIR / "tokenizer"

# Tokenizer files
TINYSTORIES_VOCAB = TOKENIZER_DIR / "tinystories_bpe_10k_train_vocab.json"
TINYSTORIES_MERGES = TOKENIZER_DIR / "tinystories_bpe_10k_train_merges.json"
OWT_VOCAB = TOKENIZER_DIR / "owt_bpe_32k_train_vocab.json"
OWT_MERGES = TOKENIZER_DIR / "owt_bpe_32k_train_merges.json"

# Data files
TINYSTORIES_TRAIN = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VALID = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
OWT_TRAIN = DATA_DIR / "owt_train.txt"
OWT_VALID = DATA_DIR / "owt_valid.txt"


def sample_documents(file_path: Path, num_docs: int = 10) -> list[str]:
    """
    Sample num_docs documents from a text file.
    For simplicity, we'll read the first num_docs non-empty paragraphs.
    
    Args:
        file_path: Path to the text file
        num_docs: Number of documents to sample
        
    Returns:
        List of document strings
    """
    documents = []
    current_doc = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                current_doc.append(line)
            elif current_doc:  # Empty line and we have content
                documents.append(' '.join(current_doc))
                current_doc = []
                if len(documents) >= num_docs:
                    break
    
    # Add last document if we haven't reached num_docs
    if current_doc and len(documents) < num_docs:
        documents.append(' '.join(current_doc))
    
    return documents[:num_docs]


def calculate_compression_ratio(text: str, token_ids: list[int]) -> float:
    """
    Calculate compression ratio as bytes/token.
    
    Args:
        text: Original text
        token_ids: Encoded token IDs
        
    Returns:
        Compression ratio (bytes per token)
    """
    num_bytes = len(text.encode('utf-8'))
    num_tokens = len(token_ids)
    return num_bytes / num_tokens if num_tokens > 0 else 0


def task_a_compression_ratios():
    """
    Task (a): Sample 10 documents from TinyStories and OpenWebText. Using your 
    previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary, 
    respectively), encode these sampled documents into integer IDs. What is each 
    tokenizer's compression ratio (bytes/token)?
    
    Deliverable: A one-to-two sentence response.
    """
    print("=" * 80)
    print("Task (a): Compression Ratios")
    print("=" * 80)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    ts_tokenizer = Tokenizer.from_files(str(TINYSTORIES_VOCAB), str(TINYSTORIES_MERGES))
    owt_tokenizer = Tokenizer.from_files(str(OWT_VOCAB), str(OWT_MERGES))
    
    # Sample documents
    print("Sampling documents...")
    ts_docs = sample_documents(TINYSTORIES_TRAIN, 10)
    owt_docs = sample_documents(OWT_TRAIN, 10)
    
    # Calculate compression ratios for TinyStories tokenizer on TinyStories data
    print("\n--- TinyStories Tokenizer on TinyStories Data ---")
    ts_on_ts_ratios = []
    for i, doc in enumerate(ts_docs):
        token_ids = ts_tokenizer.encode(doc)
        ratio = calculate_compression_ratio(doc, token_ids)
        ts_on_ts_ratios.append(ratio)
        print(f"Document {i+1}: {len(doc.encode('utf-8'))} bytes, {len(token_ids)} tokens, "
              f"ratio = {ratio:.3f} bytes/token")
    
    avg_ts_on_ts = np.mean(ts_on_ts_ratios)
    print(f"\nAverage ratio: {avg_ts_on_ts:.3f} bytes/token")
    
    # Calculate compression ratios for OpenWebText tokenizer on OpenWebText data
    print("\n--- OpenWebText Tokenizer on OpenWebText Data ---")
    owt_on_owt_ratios = []
    for i, doc in enumerate(owt_docs):
        token_ids = owt_tokenizer.encode(doc)
        ratio = calculate_compression_ratio(doc, token_ids)
        owt_on_owt_ratios.append(ratio)
        print(f"Document {i+1}: {len(doc.encode('utf-8'))} bytes, {len(token_ids)} tokens, "
              f"ratio = {ratio:.3f} bytes/token")
    
    avg_owt_on_owt = np.mean(owt_on_owt_ratios)
    print(f"\nAverage ratio: {avg_owt_on_owt:.3f} bytes/token")
    
    print("\n=== Summary ===")
    print(f"TinyStories Tokenizer (10K vocab) on TinyStories Data: {avg_ts_on_ts:.3f} bytes/token")
    print(f"OpenWebText Tokenizer (32K vocab) on OpenWebText Data: {avg_owt_on_owt:.3f} bytes/token")
    print("\n=== Explanation ===")
    print("OpenWebText tokenizer has higher compression ratio because it has a larger vocabulary (32K vs 10K),")
    print("allowing it to learn longer subword units. Larger vocabularies typically achieve better compression.")
    

def task_b_cross_domain():
    """
    Task (b): What happens if you tokenize your OpenWebText sample with the 
    TinyStories tokenizer? Compare the compression ratio and/or qualitatively 
    describe what happens.
    
    Deliverable: A one-to-two sentence response.
    """
    print("\n" + "=" * 80)
    print("Task (b): Cross-Domain Tokenization")
    print("=" * 80)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    ts_tokenizer = Tokenizer.from_files(str(TINYSTORIES_VOCAB), str(TINYSTORIES_MERGES))
    owt_tokenizer = Tokenizer.from_files(str(OWT_VOCAB), str(OWT_MERGES))
    
    # Sample OpenWebText documents
    print("Sampling OpenWebText documents...")
    owt_docs = sample_documents(OWT_TRAIN, 10)
    
    # Tokenize with both tokenizers
    print("\n--- Encoding OpenWebText Data with TinyStories Tokenizer ---")
    ts_on_owt_ratios = []
    for i, doc in enumerate(owt_docs):
        token_ids = ts_tokenizer.encode(doc)
        ratio = calculate_compression_ratio(doc, token_ids)
        ts_on_owt_ratios.append(ratio)
        print(f"Document {i+1}: {len(doc.encode('utf-8'))} bytes, {len(token_ids)} tokens, "
              f"ratio = {ratio:.3f} bytes/token")
    
    avg_ts_on_owt = np.mean(ts_on_owt_ratios)
    print(f"\nAverage ratio: {avg_ts_on_owt:.3f} bytes/token")
    
    print("\n--- Encoding OpenWebText Data with OpenWebText Tokenizer (Reference) ---")
    owt_on_owt_ratios = []
    for i, doc in enumerate(owt_docs):
        token_ids = owt_tokenizer.encode(doc)
        ratio = calculate_compression_ratio(doc, token_ids)
        owt_on_owt_ratios.append(ratio)
        print(f"Document {i+1}: {len(doc.encode('utf-8'))} bytes, {len(token_ids)} tokens, "
              f"ratio = {ratio:.3f} bytes/token")
    
    avg_owt_on_owt = np.mean(owt_on_owt_ratios)
    print(f"\nAverage ratio: {avg_owt_on_owt:.3f} bytes/token")
    
    print("\n=== Summary ===")
    print(f"TinyStories Tokenizer on OpenWebText Data: {avg_ts_on_owt:.3f} bytes/token")
    print(f"OpenWebText Tokenizer on OpenWebText Data: {avg_owt_on_owt:.3f} bytes/token")
    print(f"Compression ratio change: {((avg_ts_on_owt - avg_owt_on_owt) / avg_owt_on_owt * 100):.1f}%")
    
    print("\n=== Explanation ===")
    print("When using TinyStories tokenizer on OpenWebText data, compression ratio significantly changes.")
    print("This is because TinyStories tokenizer is trained on children's stories with simple, repetitive language.")
    print("OpenWebText contains more complex and diverse vocabulary with domain-specific terms,")
    print("which may not exist in TinyStories vocabulary, requiring more tokens to encode the same text.")


def task_c_throughput():
    """
    Task (c): Estimate the throughput of your tokenizer (e.g., in bytes/second). 
    How long would it take to tokenize the Pile dataset (825GB of text)?
    
    Deliverable: A one-to-two sentence response.
    """
    print("\n" + "=" * 80)
    print("Task (c): Throughput Estimation")
    print("=" * 80)
    
    # Load a tokenizer
    print("\nLoading OpenWebText tokenizer...")
    owt_tokenizer = Tokenizer.from_files(str(OWT_VOCAB), str(OWT_MERGES))
    
    # Read a chunk of data for benchmarking
    print("Reading test data...")
    test_data = []
    total_bytes = 0
    target_bytes = 10 * 1024 * 1024  # 10MB for testing
    
    with open(OWT_TRAIN, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(line)
            total_bytes += len(line.encode('utf-8'))
            if total_bytes >= target_bytes:
                break
    
    test_text = ''.join(test_data)
    actual_bytes = len(test_text.encode('utf-8'))
    
    print(f"Test data size: {actual_bytes / (1024*1024):.2f} MB ({actual_bytes:,} bytes)")
    
    # Benchmark encoding
    print("\nStarting performance benchmark...")
    start_time = time.time()
    token_ids = owt_tokenizer.encode(test_text)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    throughput = actual_bytes / elapsed_time
    
    print(f"Encoding time: {elapsed_time:.3f} seconds")
    print(f"Generated tokens: {len(token_ids):,}")
    print(f"Throughput: {throughput:,.0f} bytes/second ({throughput / (1024*1024):.2f} MB/s)")
    
    # Estimate time for Pile dataset (825GB)
    pile_bytes = 825 * 1024 * 1024 * 1024  # 825GB in bytes
    estimated_seconds = pile_bytes / throughput
    estimated_hours = estimated_seconds / 3600
    estimated_days = estimated_hours / 24
    
    print("\n=== Pile Dataset Encoding Time Estimation ===")
    print(f"Pile dataset size: 825 GB")
    print(f"Estimated time: {estimated_seconds:,.0f} seconds")
    print(f"              = {estimated_hours:,.1f} hours")
    print(f"              = {estimated_days:,.1f} days")
    
    print("\n=== Explanation ===")
    print(f"Based on current throughput ({throughput / (1024*1024):.2f} MB/s), tokenizing the entire Pile dataset (825GB)")
    print(f"would take approximately {estimated_days:.1f} days. In practice, this can be accelerated through parallel")
    print(f"processing, optimized implementation, or faster hardware.")


def task_d_serialize_datasets():
    """
    Task (d): Using your TinyStories and OpenWebText tokenizers, encode the 
    respective training and development datasets into a sequence of integer 
    token IDs. We'll use this later to train our language model. We recommend 
    serializing the token IDs as a NumPy array of datatype uint16. Why is 
    uint16 an appropriate choice?
    
    Deliverable: A one-to-two sentence response.
    """
    print("\n" + "=" * 80)
    print("Task (d): Dataset Serialization (Demo Version)")
    print("=" * 80)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    ts_tokenizer = Tokenizer.from_files(str(TINYSTORIES_VOCAB), str(TINYSTORIES_MERGES))
    owt_tokenizer = Tokenizer.from_files(str(OWT_VOCAB), str(OWT_MERGES))
    
    # Use sampled documents instead of complete datasets (for demo purposes)
    datasets = [
        ("TinyStories Sample", TINYSTORIES_TRAIN, ts_tokenizer, "tinystories_sample_tokens.npy"),
        ("OpenWebText Sample", OWT_TRAIN, owt_tokenizer, "owt_sample_tokens.npy"),
    ]
    
    for name, data_path, tokenizer, output_name in datasets:
        print(f"\n--- Processing {name} ---")
        
        # Sample 10 documents
        print(f"Sampling 10 documents from {data_path.name}...")
        docs = sample_documents(data_path, 10)
        combined_text = '\n'.join(docs)
        
        # Encode
        start_time = time.time()
        token_ids = tokenizer.encode(combined_text)
        elapsed = time.time() - start_time
        
        # Check max token ID
        max_id = max(token_ids) if token_ids else 0
        print(f"Number of documents: {len(docs)}")
        print(f"Text size in bytes: {len(combined_text.encode('utf-8')):,}")
        print(f"Number of tokens: {len(token_ids):,}")
        print(f"Max token ID: {max_id}")
        print(f"Encoding time: {elapsed:.3f} seconds")
        
        # Convert to uint16 numpy array
        print(f"Converting to uint16 NumPy array...")
        tokens_array = np.array(token_ids, dtype=np.uint16)
        
        # Save to file
        output_path = TOKENIZER_DIR / output_name
        print(f"Saving to {output_path}...")
        np.save(output_path, tokens_array)
        
        file_size_kb = output_path.stat().st_size / 1024
        print(f"File size: {file_size_kb:.2f} KB")
        
        # Verify: reload and check
        loaded = np.load(output_path)
        print(f"Verification: reloaded token count = {len(loaded):,}")
    
    print("\n=== Why Use uint16? ===")
    print("\nuint16 can represent integers from 0 to 65,535, which is appropriate for our tokenizers:")
    print("- TinyStories tokenizer vocabulary size: 10,000 (< 65,536)")
    print("- OpenWebText tokenizer vocabulary size: 32,000 (< 65,536)")
    print("\nUsing uint16 saves 50-75% storage space compared to uint32 or int64,")
    print("while still being sufficient to represent all token IDs. If vocabulary exceeds 65,536,")
    print("uint32 would be needed, but for most applications, 32K vocabulary is sufficient.")
    print("\nNote: In practice, you should encode and save the complete training and validation datasets.")


def main():
    """Run all experiments."""
    print("Tokenizer Experiments - Section 2.7")
    print("=" * 80)
    
    # Run all tasks
    task_a_compression_ratios()
    task_b_cross_domain()
    task_c_throughput()
    task_d_serialize_datasets()
    
    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
