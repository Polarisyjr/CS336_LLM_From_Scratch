"""
Transformer architecture components for language modeling.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .module import MultiheadSelfAttention, RMSNorm, SwiGLU, Embedding, Linear


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.
    
    A Transformer block consists of two sublayers:
    1. Multi-head self-attention sublayer
    2. Position-wise feed-forward sublayer
    
    Each sublayer follows the pattern:
        output = input + Sublayer(RMSNorm(input))
    
    This is the "pre-norm" architecture, which applies normalization before
    the sublayer rather than after (as in the original "Attention is All You Need").
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a pre-norm Transformer block.
        
        Args:
            d_model: Dimensionality of the Transformer block inputs/outputs
            num_heads: Number of heads to use in multi-head self-attention
            d_ff: Dimensionality of the position-wise feed-forward inner layer
            max_seq_len: Maximum sequence length for RoPE (optional)
            theta: RoPE base frequency parameter (optional)
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        
        # Store configuration
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # First sublayer: RMSNorm + Multi-head Self-Attention
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        
        # Second sublayer: RMSNorm + Feed-Forward Network
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """
        Apply the Transformer block transformation.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Token position indices of shape (..., seq_len)
                           Required if RoPE is enabled, optional otherwise
        
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # First sublayer: x + MultiHeadSelfAttention(RMSNorm(x))
        # Apply RMSNorm
        x_norm = self.ln1(x)
        # Apply multi-head self-attention
        attn_output = self.attn(x_norm, token_positions)
        # Add residual connection
        x = x + attn_output
        
        # Second sublayer: x + FeedForward(RMSNorm(x))
        # Apply RMSNorm
        x_norm = self.ln2(x)
        # Apply feed-forward network
        ff_output = self.ffn(x_norm)
        # Add residual connection
        x = x + ff_output
        
        return x


class TransformerLM(nn.Module):
    """
    Transformer Language Model.
    
    A complete language model consisting of:
    1. Token embedding layer
    2. Stack of Transformer blocks
    3. Final RMSNorm layer
    4. Language modeling head (output projection to vocabulary)
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a Transformer language model.
        
        Args:
            vocab_size: Size of the vocabulary
            context_length: Maximum context length (for position embeddings)
            d_model: Dimensionality of the model embeddings
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads per block
            d_ff: Dimensionality of the feed-forward inner layer
            theta: RoPE base frequency parameter (optional)
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        
        # Token embedding layer
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        
        # Stack of Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        
        # Final RMSNorm layer
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Language modeling head (output projection to vocabulary)
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Apply the Transformer language model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
        
        Returns:
            Logits over vocabulary of shape (batch_size, seq_len, vocab_size)
        """
        # Get token embeddings
        # Shape: (batch_size, seq_len, d_model)
        x = self.token_embeddings(input_ids)
        
        # Pass through each Transformer block
        # token_positions will be auto-generated inside MultiheadSelfAttention if needed
        for layer in self.layers:
            x = layer(x)
        
        # Apply final RMSNorm
        x = self.ln_final(x)
        
        # Apply language modeling head to get logits
        # Shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)
        
        return logits
