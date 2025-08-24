import torch  # Import torch module for deep learning operations
import re  # Import re module for regular expressions
from torch import nn  # Import nn module from torch for neural network components
from torch.nn import functional as F  # Import functional from nn for functional operations
from config import block_size, n_embd, n_head, n_layer, dropout  # Import config parameters

class Head(nn.Module):
    """Class for a single attention head."""
    def __init__(self, head_size):
        super().__init__()  # Call super class initializer
        self.key = nn.Linear(n_embd, head_size, bias=False)  # Linear layer for key
        self.query = nn.Linear(n_embd, head_size, bias=False)  # Linear layer for query
        self.value = nn.Linear(n_embd, head_size, bias=False)  # Linear layer for value
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # Register tril buffer for masking
        self.dropout = nn.Dropout(dropout)  # Dropout layer

    def forward(self, x):
        """Forward pass for attention head."""
        B, T, C = x.shape  # Get batch, time, channel dimensions
        k = self.key(x)  # Compute key
        q = self.query(x)  # Compute query
        wei = q @ k.transpose(-2, -1) * C**-0.5  # Compute attention weights
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Mask future tokens
        wei = torch.softmax(wei, dim=-1)  # Softmax for probabilities
        wei = self.dropout(wei)  # Apply dropout
        v = self.value(x)  # Compute value
        out = wei @ v  # Weighted sum
        return out  # Return output

class MultiHeadAttention(nn.Module):
    """Class for multi-head attention."""
    def __init__(self, num_heads, head_size):
        super().__init__()  # Call super class initializer
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # List of attention heads
        self.proj = nn.Linear(n_embd, n_embd)  # Projection layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer

    def forward(self, x):
        """Forward pass for multi-head attention."""
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate head outputs
        out = self.dropout(self.proj(out))  # Project and apply dropout
        return out  # Return output

class FeedForward(nn.Module):
    """Class for feed-forward network."""
    def __init__(self, n_embd):
        super().__init__()  # Call super class initializer
        self.net = nn.Sequential(  # Sequential network
            nn.Linear(n_embd, 4 * n_embd),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(4 * n_embd, n_embd),  # Linear layer
            nn.Dropout(dropout),  # Dropout layer
        )

    def forward(self, x):
        """Forward pass for feed-forward network."""
        return self.net(x)  # Pass input through network

class Block(nn.Module):
    """Class for transformer block."""
    def __init__(self, n_embd, n_head):
        super().__init__()  # Call super class initializer
        head_size = n_embd // n_head  # Compute head size
        self.sa = MultiHeadAttention(n_head, head_size)  # Self-attention layer
        self.ffwd = FeedForward(n_embd)  # Feed-forward layer
        self.ln1 = nn.LayerNorm(n_embd)  # Layer norm 1
        self.ln2 = nn.LayerNorm(n_embd)  # Layer norm 2

    def forward(self, x):
        """Forward pass for transformer block."""
        x = x + self.sa(self.ln1(x))  # Self-attention residual
        x = x + self.ffwd(self.ln2(x))  # Feed-forward residual
        return x  # Return output

class GPTLanguageModel(nn.Module):
    """Class for GPT language model."""
    def __init__(self, vocab_size):
        super().__init__()  # Call super class initializer
        self.vocab_size = vocab_size  # Set vocabulary size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)  # Token embedding layer
        self.position_embedding = nn.Embedding(block_size, n_embd)  # Position embedding layer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])  # Transformer blocks
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer norm
        self.head = nn.Linear(n_embd, vocab_size)  # Output head
        self.apply(self._init_weights)  # Apply weight initialization

    def _init_weights(self, module):
        """Function to initialize weights."""
        if isinstance(module, nn.Linear):  # Check if linear layer
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Initialize weights
            if module.bias is not None:  # Check if bias exists
                torch.nn.init.zeros_(module.bias)  # Initialize bias to zero
        elif isinstance(module, nn.Embedding):  # Check if embedding layer
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Initialize weights

    def forward(self, idx, targets=None):
        """Forward pass for GPT language model."""
        B, T = idx.shape  # Get batch and time dimensions
        tok_emb = self.token_embedding(idx)  # Token embeddings
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # Position embeddings
        x = tok_emb + pos_emb  # Combine embeddings
        x = self.blocks(x)  # Pass through transformer blocks
        x = self.ln_f(x)  # Final layer norm
        logits = self.head(x)  # Output logits
        if targets is None:  # If no targets
            loss = None  # No loss
        else:
            B, T, C = logits.shape  # Get dimensions
            logits = logits.view(B * T, C)  # Reshape logits
            targets = targets.view(B * T)  # Reshape targets
            loss = torch.nn.functional.cross_entropy(logits, targets)  # Compute loss
        return logits, loss  # Return logits and loss

def tokenize(sentences, vocab, block_size=256):
    """Function to tokenize sentences."""
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}  # Create word to index dictionary
    sequences = []  # Initialize list for sequences
    for sentence in sentences:  # Loop through sentences
        sentence = re.sub(r'\s+', ' ', sentence).strip().lower()  # Clean sentence
        words = re.findall(r'[a-z]+|[0-9]+|[.,]', sentence)  # Extract words
        tokens = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]  # Get tokens
        tokens = [word_to_idx['<SOS>']] + tokens + [word_to_idx['<EOS>']]  # Add SOS and EOS
        if len(tokens) > block_size:  # Check length
            tokens = tokens[:block_size]  # Truncate if too long
        else:
            tokens += [word_to_idx['<PAD>']] * (block_size - len(tokens))  # Pad if too short
        sequences.append(tokens)  # Append sequence
    return sequences  # Return sequences
