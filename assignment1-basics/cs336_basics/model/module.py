"""
Neural network modules for building language models.
"""

import torch
import torch.nn as nn
from torch import Tensor


class Linear(nn.Module):
    """
    Linear transformation module without bias: y = Wx
    
    This follows the pattern of modern LLMs which typically don't use bias terms.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a linear transformation module.
        
        Args:
            in_features: Final dimension of the input
            out_features: Final dimension of the output
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        
        # Store dimensions
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight matrix as W (not W^T) with shape [out_features, in_features]
        # This is stored as a Parameter so it will be registered and optimized
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        # Initialize weights using truncated normal distribution
        # σ² = 2 / (d_in + d_out), truncated at [-3σ, 3σ]
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        
        Args:
            x: Input tensor with last dimension = in_features
            
        Returns:
            Output tensor with last dimension = out_features
        """
        # y = xW^T, which is equivalent to y = Wx when considering the shapes
        # x: [..., in_features], W: [out_features, in_features]
        # result: [..., out_features]
        return torch.matmul(x, self.W.t())


class Embedding(nn.Module):
    """
    Embedding module that maps integer token IDs to dense vectors.
    
    This is the first layer of a Transformer model, converting discrete tokens
    into continuous vector representations of dimension d_model.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct an embedding module.
        
        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors, i.e., d_model
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        
        # Store dimensions
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize embedding matrix as a Parameter with shape (vocab_size, d_model)
        # This will be registered with the module and included in optimization
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        # Initialize weights using truncated normal distribution
        # Use the default settings for truncated normal initialization
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        
        Args:
            token_ids: Input tensor of token IDs with shape (batch_size, sequence_length)
                      Must be of type torch.LongTensor
            
        Returns:
            Embedding vectors with shape (batch_size, sequence_length, embedding_dim)
            Each token ID is replaced with its corresponding embedding vector
        """
        # Use indexing to select rows from the embedding matrix
        # token_ids: (batch_size, sequence_length)
        # weight: (num_embeddings, embedding_dim)
        # result: (batch_size, sequence_length, embedding_dim)
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Normalizes activations using their RMS value and applies a learnable
    affine transformation. This is simpler and more efficient than LayerNorm.
    Used in modern LLMs like LLaMA and PaLM.
    """
    
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct an RMSNorm module.
        
        Args:
            d_model: Hidden dimension of the model (dimensionality to normalize over)
            eps: Epsilon value for numerical stability (added to denominator)
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        
        # Store configuration
        self.d_model = d_model
        self.eps = eps
        
        # Learnable gain parameter (affine transform weights)
        # Shape: (d_model,) - one gain value per feature dimension
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor.
        
        Args:
            x: Input tensor of shape (..., d_model)
               Can have arbitrary leading dimensions (batch, sequence, etc.)
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Save original dtype to restore later
        in_dtype = x.dtype
        
        # Upcast to float32 to prevent overflow when squaring
        x = x.to(torch.float32)
        
        # Compute RMS: sqrt(mean(x^2) + eps)
        # x^2: element-wise square
        # mean(dim=-1, keepdim=True): compute mean over last dimension (d_model)
        # keepdim=True preserves the dimension for broadcasting
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize: x / RMS(x)
        # This scales each activation by its RMS value
        x_normalized = x / rms
        
        # Apply learnable affine transform: multiply by gain parameter
        # self.weight has shape (d_model,) and broadcasts over leading dimensions
        result = x_normalized * self.weight
        
        # Convert back to original dtype
        return result.to(in_dtype)


class SiLU(nn.Module):
    """
    SiLU (Sigmoid Linear Unit) activation function, also known as Swish.
    
    SiLU(x) = x * σ(x) = x * (1 / (1 + e^(-x)))
    
    Used in modern LLMs as part of the SwiGLU feed-forward network.
    """
    
    def __init__(self):
        """Construct a SiLU activation module."""
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SiLU activation function.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Output tensor of same shape as input
        """
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit feed-forward network.
    
    Combines SiLU activation with a gating mechanism (GLU).
    Architecture: FFN(x) = W2(SiLU(W1·x) ⊙ W3·x)
    
    This is used in modern LLMs like LLaMA and PaLM as the feed-forward
    component of Transformer blocks.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a SwiGLU feed-forward network.
        
        Args:
            d_model: Dimensionality of input and output
            d_ff: Dimensionality of the inner feed-forward layer
                  Typically set to (8/3) * d_model, rounded to nearest multiple of 64
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        
        # Store dimensions
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Three linear projections (all without bias, following modern LLM practice)
        # W1: projects input to d_ff for SiLU gate
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
        # W3: projects input to d_ff for element-wise multiplication
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
        # W2: projects back down to d_model
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        
        self.silu = SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
        """
        # FFN(x) = W2(SiLU(W1·x) ⊙ W3·x)
        # where ⊙ is element-wise multiplication
        
        # Gate path: SiLU(W1·x)
        gate = self.silu(self.w1(x))
        
        # Value path: W3·x
        value = self.w3(x)
        
        # Element-wise multiplication (gating)
        gated = gate * value
        
        # Project back to d_model
        output = self.w2(gated)
        
        return output


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Applies rotary positional embeddings to query or key tensors.
    For a given position i, rotates pairs of embedding elements by angle θ_{i,k} = i / (Θ^(2k/d)).
    
    This allows the model to encode relative positional information through rotation matrices.
    Used in models like LLaMA, PaLM, and GPT-NeoX.
    """
    
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """
        Construct the RoPE module and precompute rotation matrices.
        
        Args:
            theta: Θ value for RoPE (controls rotation frequency)
            d_k: Dimension of query and key vectors
            max_seq_len: Maximum sequence length to precompute
            device: Device to store the buffers on
        """
        super().__init__()
        
        # Store configuration
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Precompute rotation angles for all positions and all dimension pairs
        # Shape: (max_seq_len, d_k // 2)
        
        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        # Shape: (max_seq_len, 1)
        position = torch.arange(max_seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Dimension pair indices: k = [0, 1, 2, ..., d_k//2 - 1]
        # Compute frequency: 1 / (theta^(2k/d_k))
        # Shape: (1, d_k // 2)
        k = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta ** (k / d_k))
        freqs = freqs.unsqueeze(0)  # Shape: (1, d_k // 2)
        
        # Compute angles: position * freqs
        # Shape: (max_seq_len, d_k // 2)
        angles = position * freqs
        
        # Precompute cos and sin for all angles
        # Shape: (max_seq_len, d_k // 2)
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)
        
        # Register as buffers (non-learnable, but part of state_dict)
        # persistent=False means they won't be saved in state_dict
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
               Typically query or key vectors
            token_positions: Token position indices of shape (..., seq_len)
                           Used to slice precomputed cos/sin tensors
            
        Returns:
            Rotated tensor of same shape as input
        """
        # Slice cos and sin using token positions
        # token_positions: (..., seq_len)
        # cos_cached: (max_seq_len, d_k // 2)
        # Result: (..., seq_len, d_k // 2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        # 若 x 形如 (*batch, H, T, d)，则需要在 cos/sin 上插入 head 广播维
        # x.ndim = len(batch)+3, token_positions.ndim = len(batch)+1
        if x.ndim == token_positions.ndim + 2:
            cos = cos.unsqueeze(-3)  # (*batch, 1, T, d2)
            sin = sin.unsqueeze(-3)  # (*batch, 1, T, d2)
        
        # Reshape x to separate even and odd indices
        # x: (..., seq_len, d_k) -> (..., seq_len, d_k // 2, 2)
        # -1 表示“这一维我不写，让框架自动推断”
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        
        # Split into even and odd elements
        # x1: elements at positions [0, 2, 4, ...]
        # x2: elements at positions [1, 3, 5, ...]
        x1 = x_reshaped[..., 0]  # Shape: (..., seq_len, d_k // 2)
        x2 = x_reshaped[..., 1]  # Shape: (..., seq_len, d_k // 2)
        
        # Apply rotation matrix:
        # [x1']   [cos  -sin] [x1]
        # [x2'] = [sin   cos] [x2]
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos
        
        # Stack back together and reshape to original shape
        # Shape: (..., seq_len, d_k // 2, 2)
        x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1)
        
        # Reshape back to (..., seq_len, d_k)
        # "*" 表示“把这个列表/元组/张量的所有维度都展开写出来”，如果 x.shape 是 (2, 3, 4)，那么 *x.shape 就相当于写 2, 3, 4
        x_out = x_rotated.reshape(*x.shape)
        
        return x_out

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax operation to a tensor along a specific dimension.
    
    Computes: softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
    
    Uses the numerical stability trick of subtracting the maximum value
    from all elements before computing exponentials to avoid overflow.
    
    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension along which to apply softmax
        
    Returns:
        Output tensor with the same shape as input, with values normalized
        to a probability distribution along the specified dimension.
    """
    # For numerical stability, subtract the maximum value along the dimension
    # This prevents exp(x) from becoming inf for large values
    # softmax(x) = softmax(x - c) for any constant c
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    # torch.max returns (values, indices)
    x_shifted = x - x_max
    
    # Compute exp(x - max(x))
    exp_x = torch.exp(x_shifted)
    
    # Compute the sum of exponentials along the dimension
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    
    # Normalize to get probabilities
    return exp_x / sum_exp


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
    
    Args:
        Q: Query tensor of shape (..., seq_len_q, d_k)
        K: Key tensor of shape (..., seq_len_k, d_k)
        V: Value tensor of shape (..., seq_len_k, d_v)
        mask: Optional boolean mask of shape (seq_len_q, seq_len_k)
              True means attend, False means don't attend (set to -inf)
              
    Returns:
        Output tensor of shape (..., seq_len_q, d_v)
    """
    # Get d_k from the last dimension of Q or K
    d_k = Q.size(-1)
    
    # Compute attention scores: Q·K^T / sqrt(d_k)
    # Q: (..., seq_len_q, d_k)
    # K^T: (..., d_k, seq_len_k)
    # scores: (..., seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Apply mask if provided
    if mask is not None:
        # Where mask is False, set scores to -inf
        # This ensures these positions get 0 probability after softmax
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Apply softmax to get attention probabilities
    # Shape: (..., seq_len_q, seq_len_k)
    attn_probs = softmax(scores, dim=-1)
    
    # Apply attention to values
    # attn_probs: (..., seq_len_q, seq_len_k)
    # V: (..., seq_len_k, d_v)
    # output: (..., seq_len_q, d_v)
    output = torch.matmul(attn_probs, V)
    
    return output


class MultiheadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention.
    
    Implements multi-head self-attention with causal masking to prevent
    attention to future positions. This is a core component of decoder-only
    Transformer models like GPT.
    
    Architecture:
        1. Project input to Q, K, V using learned weight matrices
        2. Split into multiple heads
        3. Apply scaled dot-product attention with causal mask for each head
        4. Concatenate heads and project to output
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a multi-head self-attention module.
        
        Args:
            d_model: Dimensionality of model embeddings (input and output)
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length for RoPE (optional, enables RoPE if provided)
            theta: RoPE base frequency parameter (optional, default: 10000.0)
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        
        # Store configuration
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Following Vaswani et al. 2017: d_k = d_v = d_model / num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        # Total dimensions for all heads combined
        # q_proj and k_proj: project to h * d_k dimensions
        # v_proj: project to h * d_v dimensions
        total_d_k = num_heads * self.d_k
        total_d_v = num_heads * self.d_v
        
        # Query, Key, Value projection matrices
        # These project from d_model to (num_heads * d_k) or (num_heads * d_v)
        # Using a single matrix multiply for efficiency
        self.q_proj = Linear(d_model, total_d_k, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, total_d_k, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, total_d_v, device=device, dtype=dtype)
        
        # Output projection: projects concatenated heads back to d_model
        self.o_proj = Linear(total_d_v, d_model, device=device, dtype=dtype)
        
        # Optional RoPE support
        if max_seq_len is not None:
            if theta is None:
                theta = 10000.0
            self.rope = RoPE(
                theta=theta,
                d_k=self.d_k,  # RoPE dimension is per-head dimension
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.rope = None
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply causal multi-head self-attention.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
               Can have arbitrary batch dimensions
            token_positions: Token position indices of shape (..., seq_len)
                           Required if RoPE is enabled, optional otherwise
            
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # Get shape information
        # x: (..., seq_len, d_model)
        *batch_dims, seq_len, d_model = x.shape
        
        # If RoPE is enabled and token_positions not provided, use default sequential positions
        if self.rope is not None and token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
            # Add batch dimensions if needed
            for _ in batch_dims:
                token_positions = token_positions.unsqueeze(0)
            # Expand to match batch dimensions
            token_positions = token_positions.expand(*batch_dims, seq_len)
        
        # Project to Q, K, V
        # Shape: (..., seq_len, num_heads * d_k) or (..., seq_len, num_heads * d_v)
        Q = self.q_proj(x)  # (..., seq_len, num_heads * d_k)
        K = self.k_proj(x)  # (..., seq_len, num_heads * d_k)
        V = self.v_proj(x)  # (..., seq_len, num_heads * d_v)
        
        # Reshape to separate heads
        # (..., seq_len, num_heads * d_k) -> (..., seq_len, num_heads, d_k)
        # Then transpose to (..., num_heads, seq_len, d_k) for parallel attention
        Q = Q.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch_dims, seq_len, self.num_heads, self.d_v).transpose(-3, -2)
        
        # Now shapes are: (..., num_heads, seq_len, d_k) or (..., num_heads, seq_len, d_v)
        
        # Apply RoPE to Q and K if enabled (not V!)
        # The head dimension is treated as a batch dimension for RoPE
        # PyTorch will automatically broadcast cos/sin across the num_heads dimension
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        # Create causal mask: position i can only attend to positions j <= i
        # This is a lower triangular matrix (including diagonal)
        # Shape: (seq_len, seq_len)
        # Example for seq_len=4:
        # [[T, F, F, F],   <- position 0 can only see itself
        #  [T, T, F, F],   <- position 1 can see 0,1
        #  [T, T, T, F],   <- position 2 can see 0,1,2
        #  [T, T, T, T]]   <- position 3 can see all
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        
        # Apply scaled dot-product attention with causal mask
        # Q, K, V: (..., num_heads, seq_len, d_k/d_v)
        # mask: (seq_len, seq_len) - broadcasts over batch and head dimensions
        # output: (..., num_heads, seq_len, d_v)
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        
        # Reshape: (..., num_heads, seq_len, d_v) -> (..., seq_len, num_heads, d_v)
        attn_output = attn_output.transpose(-3, -2)
        
        # Concatenate all heads: (..., seq_len, num_heads, d_v) -> (..., seq_len, num_heads * d_v)
        attn_output = attn_output.contiguous().view(*batch_dims, seq_len, self.num_heads * self.d_v)
        
        # Final output projection
        # (..., seq_len, num_heads * d_v) -> (..., seq_len, d_model)
        output = self.o_proj(attn_output)
        
        return output