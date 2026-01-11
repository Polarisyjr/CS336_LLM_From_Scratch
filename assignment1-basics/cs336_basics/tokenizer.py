"""
BPE Tokenizer: Encoding and Decoding

Implements a Tokenizer class that encodes text into integer IDs and decodes integer IDs into text.
The tokenizer uses Byte Pair Encoding (BPE) with a provided vocabulary and list of merges.
"""

from typing import Optional, Iterator
from collections.abc import Iterable
import regex

# GPT-2 pre-tokenization pattern
GPT2_PAT = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class Tokenizer:
    """
    BPE Tokenizer for encoding and decoding text.
    
    Given a vocabulary and a list of merges, encodes text into integer IDs 
    and decodes integer IDs into text. Also supports user-provided special tokens.
    """
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and 
        (optionally) a list of special tokens.
        
        Args:
            vocab: dict[int, bytes]
                The tokenizer vocabulary, a mapping from int (token ID) to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
            special_tokens: list[str] | None = None
                A list of string special tokens for the tokenizer. These strings will never
                be split into multiple tokens, and will always be kept as a single token.
        """
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_set = set(self.special_tokens)
        # Create reverse vocabulary: bytes -> token_id
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        self.merge_priority = {merge: i for i, merge in enumerate(merges)}

        # Add special tokens to vocabulary if they're not already there
        for special_token in self.special_tokens:
            special_bytes = special_token.encode('utf-8')
            if special_bytes not in self.vocab_reverse:
                # Find the next available token ID
                new_id = max(self.vocab.keys()) + 1
                self.vocab[new_id] = special_bytes
                self.vocab_reverse[special_bytes] = new_id
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[list[str]] = None
    ) -> "Tokenizer":
        """
        Class method that constructs and returns a Tokenizer from a serialized 
        vocabulary and list of merges (in the same format that your BPE training 
        code output) and (optionally) a list of special tokens.
        
        Args:
            vocab_filepath: str
                Path to the vocabulary JSON file
            merges_filepath: str
                Path to the merges JSON file
            special_tokens: list[str] | None = None
                Optional list of special tokens
                
        Returns:
            Tokenizer: A new Tokenizer instance
        """
        import json
        
        # Load vocab from JSON: {str(int): hex_string}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_obj = json.load(f)
        vocab = {int(i): bytes.fromhex(b) for i, b in vocab_obj.items()}
        
        # Load merges from JSON: [[hex_string, hex_string], ...]
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_obj = json.load(f)
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_obj]
        
        # Usage: tok = Tokenizer.from_files("vocab.json", "merges.json")
        return cls(vocab, merges, special_tokens)
    
    def _pre_tokenize(self, text: str) -> list[str]:
        """
        Pre-tokenize text into a list of pre-tokens.
        
        Step 1: Split on special tokens (keeping them as separate pre-tokens)
        Step 2: For non-special parts, apply GPT-2 regex pattern to get pre-tokens
        
        Args:
            text: Input text to pre-tokenize
            
        Returns:
            list[str]: List of pre-token strings (including special tokens)
        """
        # Build special token split pattern if needed
        special_pattern = None
        if self.special_tokens:
            escaped = [regex.escape(tok) for tok in self.special_tokens]
            escaped.sort(key=len, reverse=True)
            special_pattern = "(" + "|".join(escaped) + ")"
        
        # Split on special tokens
        if special_pattern:
            parts = regex.split(special_pattern, text)
        else:
            parts = [text]
        
        pretokens = []
        
        # Process each part
        for part in parts:
            if not part:
                continue

            if part in self.special_set:
                pretokens.append(part)
                continue
            
            for match in GPT2_PAT.finditer(part):
                pretokens.append(match.group(0))
        
        return pretokens
    
    def _apply_bpe(self, pretoken_bytes: bytes) -> list[int]:
        """
        Apply BPE merges to a single pre-token (sequence of bytes).
        
        Iteratively finds the highest-priority (earliest created) merge that
        exists in the current sequence and applies it to ALL occurrences,
        until no more merges apply.
        
        Args:
            pretoken_bytes: The UTF-8 encoded bytes of a pre-token
            
        Returns:
            list[int]: Token IDs after applying BPE merges
        """
        if not pretoken_bytes:
            return []
        
        # Start with individual bytes as tokens
        tokens = [bytes([b]) for b in pretoken_bytes]
        
        # Iteratively apply merges
        while len(tokens) > 1:
            # Find the highest priority (earliest) merge among current pairs
            best_pair = None
            best_priority = float('inf')
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair
            
            # If no merge applies, we're done
            if best_pair is None:
                break
            
            # Apply the merge to ALL occurrences of best_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                # Check if current pair matches best_pair
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    # Merge this pair
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
        
        # Convert tokens to IDs
        return [self.vocab_reverse[token] for token in tokens]
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        
        Args:
            text: str
                The input text to encode
                
        Returns:
            list[int]: A list of token IDs
        """
        if not text:
            return []
        
        # Step 1: Pre-tokenize
        pretokens = self._pre_tokenize(text)
        
        # Step 2: Apply BPE to each pre-token
        result_ids = []
        for pretoken in pretokens:
            # If it's a special token, add its ID directly
            if pretoken in self.special_set:
                pretoken_bytes = pretoken.encode('utf-8')
                result_ids.append(self.vocab_reverse[pretoken_bytes])
            else:
                # Apply BPE merges to this pre-token
                pretoken_bytes = pretoken.encode('utf-8')
                token_ids = self._apply_bpe(pretoken_bytes)
                result_ids.extend(token_ids)
        
        return result_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator 
        that lazily yields token IDs. This is required for memory-efficient tokenization 
        of large files that we cannot directly load into memory.
        
        Args:
            iterable: Iterable[str]
                An iterable of strings (e.g., file handle)
                
        Returns:
            Iterator[int]: A generator that yields token IDs
        """
        for chunk in iterable:
            yield from self.encode(chunk)
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        
        Args:
            ids: list[int]
                A list of token IDs to decode
                
        Returns:
            str: The decoded text
        """
        # 1) Look up the bytes for each token ID in the vocabulary
        byte_seq = b"".join(self.vocab[i] for i in ids)

        # 2) Decode UTF-8, replacing invalid bytes with U+FFFD replacement character
        return byte_seq.decode("utf-8", errors="replace")