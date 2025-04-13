import torch  # Core library for tensor computations and neural networks
import torch.nn as nn  # Provides neural network layers and utilities
from torch.nn import functional as F  # Functional operations like softmax and padding
import pickle  # Serialization tool to save/load the trained model
import pandas as pd  # Data handling for reading Parquet files
import re  # Regular expressions for text cleaning and tokenization
from torch.amp import GradScaler, autocast  # Mixed precision training tools
from collections import Counter  # Utility to count token frequencies
import gc  # Garbage collection to manage memory usage
import os  # Operating system interface for file operations

# Clean up system resources before starting training
def cleanup():
    if torch.cuda.is_available():  # If GPU is available
        torch.cuda.empty_cache()  # Release unused GPU memory
        torch.cuda.ipc_collect()  # Collect inter-process memory fragments
    gc.collect()  # Trigger Python garbage collection
    model_file = 'model.pkl'  # File where model will be saved
    if os.path.exists(model_file):  # Check if old model exists
        os.remove(model_file)  # Delete it to start fresh
        print(f"Removed previous {model_file}")

print("Performing cleanup...")
cleanup()

# Set computation device (GPU if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

# Define model and training hyperparameters
block_size = 128         # Maximum sequence length for inputs/outputs
batch_size = 8           # Number of sequences per training batch
max_iters = 50000        # Total training iterations
learning_rate = 3e-4     # Initial learning rate for optimizer
eval_iters = 100         # Number of batches for loss evaluation (increased for consistency)
n_embd = 384             # Embedding dimension for token representations
n_head = 6               # Number of attention heads (n_embd / n_head = 64)
n_layer = 10             # Number of transformer layers
dropout = 0.2            # Dropout probability to prevent overfitting
accumulation_steps = 4   # Steps to accumulate gradients for memory efficiency

# Load training data from Parquet file (optional: align column names with second training)
def load_data(file_path, prompt_col='question', response_col='response'):
    df = pd.read_parquet(file_path)  # Load Parquet into a DataFrame
    prompts = df[prompt_col].values.tolist()  # Convert prompt column to list
    responses = df[response_col].values.tolist()  # Convert response column to list
    return prompts, responses

file_path = "C:/Users/Owner/PycharmProjects/LLMProject8/1M-GPT4-Augmented.parquet"
prompts, responses = load_data(file_path)  # Fetch prompts and responses

# Build vocabulary from dataset (aligned with second training and interface)
def build_vocabulary(prompts, responses):
    combined_data = prompts + responses  # Merge prompts and responses
    text = " ".join(str(t) for t in combined_data if t is not None)  # Concatenate into one string
    text = re.sub(r'[\[\]"\']', ' ', text)  # Remove brackets and quotes
    text = re.sub(r'-{2,}', ' ', text)  # Collapse multiple hyphens
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    words = re.findall(r'[a-zA-Z]+[-a-zA-Z]*|[0-9]+|[.,$]', text)  # Tokenize words, numbers, punctuation
    word_counts = Counter(words)  # Count token frequencies
    filtered_words = set(word.lower() for word in words if
                         len(word) > 1 or word.isdigit() or word in '.,$')  # Filter valid tokens
    vocab = [word for word, count in word_counts.most_common() if word.lower() in filtered_words]  # Sort by frequency
    return vocab

words = build_vocabulary(prompts, responses)  # Generate vocabulary
vocabulary_size = len(words)
special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']  # Special tokens (consistent across scripts)
if vocabulary_size > 50000:  # Cap vocabulary at 50,000
    words = words[:50000 - len(special_tokens)]
words = special_tokens + sorted(set(words) - set(special_tokens))  # Add special tokens, ensure uniqueness
string_to_int = {word: i for i, word in enumerate(words)}  # Word-to-index mapping
int_to_string = {i: word for i, word in enumerate(words)}  # Index-to-word mapping
vocabulary_size = len(words)
print(f"Final vocabulary size: {vocabulary_size}")
print("Sample vocabulary words:", words[:10])

# Encode text into integer tokens
def encode(text):
    text = re.sub(r'-{2,}', ' ', text)  # Collapse hyphens
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    tokens = re.findall(r'[a-zA-Z]+[-a-zA-Z]*|[0-9]+|[.,$]', text)  # Tokenize
    return [string_to_int.get(token.lower(), string_to_int['<UNK>']) for token in tokens]  # Map to integers

# Decode integer tokens back to text
def decode(tokens):
    return ' '.join([int_to_string[int(i)] for i in tokens if int(i) in int_to_string])  # Join tokens into string

# Prepare training batches
def get_batch(prompts, responses):
    ix = torch.randint(len(prompts), (batch_size,))  # Random indices for batch
    batch_prompts = [prompts[i] for i in ix]  # Select prompts
    batch_responses = [responses[i] for i in ix]  # Select responses
    full_sequences = [encode(f"<SOS> {p} {r} <EOS>") for p, r in zip(batch_prompts, batch_responses)]  # Combine with tokens
    x_sequences, y_sequences = [], []
    for seq in full_sequences:
        x_seq = seq[:-1][:block_size]  # Input: all but last token
        y_seq = seq[1:][:block_size]  # Target: all but first token
        x_seq = x_seq + [string_to_int['<PAD>']] * (block_size - len(x_seq))  # Pad input
        y_seq = y_seq + [-1] * (block_size - len(y_seq))  # Pad target
        x_sequences.append(torch.tensor(x_seq, dtype=torch.long, device=device))  # Convert to tensor
        y_sequences.append(torch.tensor(y_seq, dtype=torch.long, device=device))
    return torch.stack(x_sequences), torch.stack(y_sequences)  # Stack into batch tensors

# Single attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)  # Key projection
        self.query = nn.Linear(n_embd, head_size, bias=False)  # Query projection
        self.value = nn.Linear(n_embd, head_size, bias=False)  # Value projection
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # Causal mask
        self.dropout = nn.Dropout(dropout)  # Regularization

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # Compute keys
        q = self.query(x)  # Compute queries
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # Scaled dot-product
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Mask future
        wei = F.softmax(wei, dim=-1)  # Normalize
        wei = self.dropout(wei)  # Apply dropout
        v = self.value(x)  # Compute values
        return wei @ v  # Attention output

# Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # Multiple heads
        self.proj = nn.Linear(head_size * num_heads, n_embd)  # Projection layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate head outputs
        return self.dropout(self.proj(out))  # Project and dropout

# Feed-forward network
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand
            nn.ReLU(),  # Activation
            nn.Linear(4 * n_embd, n_embd),  # Contract
            nn.Dropout(dropout),  # Regularization
        )

    def forward(self, x):
        return self.net(x)

# Transformer block
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head  # Size per head
        self.sa = MultiHeadAttention(n_head, head_size)  # Self-attention
        self.ffwd = FeedForward(n_embd)  # Feed-forward
        self.ln1 = nn.LayerNorm(n_embd)  # Layer norm 1
        self.ln2 = nn.LayerNorm(n_embd)  # Layer norm 2

    def forward(self, x):
        y = self.sa(x)  # Self-attention
        x = self.ln1(x + y)  # Residual + norm
        y = self.ffwd(x)  # Feed-forward
        return self.ln2(x + y)  # Residual + norm

# GPT language model
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # Token embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # Position embeddings
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])  # Transformer blocks
        self.ln_f = nn.LayerNorm(n_embd)  # Final norm
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Output layer
        self.apply(self._init_weights)  # Initialize weights

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Linear weights
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # Linear biases
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Embedding weights

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # Token embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # Position embeddings
        x = tok_emb + pos_emb  # Combine
        x = self.blocks(x)  # Transformer blocks
        x = self.ln_f(x)  # Normalize
        logits = self.lm_head(x)  # Predict logits
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # Flatten
            targets = targets.view(B * T)  # Flatten
            loss = F.cross_entropy(logits, targets, ignore_index=-1)  # Compute loss
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.7):  # Default temperature (not used in interface)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Last block_size tokens
            logits, _ = self.forward(idx_cond)  # Predict
            logits = logits[:, -1, :] / temperature  # Scale logits
            probs = F.softmax(logits, dim=-1)  # Probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample
            idx = torch.cat((idx, idx_next), dim=1)  # Append
            if idx_next.item() == string_to_int['<EOS>']:  # Stop at EOS
                break
        return idx

# Initialize model, optimizer, and scaler
model = GPTLanguageModel(vocabulary_size).to(device)  # Create model and move to device
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # AdamW optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=3e-5)  # Added LR scheduler
scaler = GradScaler('cuda')  # Mixed precision scaler

# Estimate validation loss
@torch.no_grad()
def estimate_loss():
    model.eval()  # Evaluation mode
    losses = torch.zeros(eval_iters)  # Store loss values
    for k in range(eval_iters):
        X, Y = get_batch(prompts, responses)  # Get batch
        with autocast('cuda'):  # Mixed precision
            logits, loss = model(X, Y)  # Compute loss
        losses[k] = loss.item()
    model.train()  # Back to training mode
    return losses.mean()  # Average loss

# Training loop
print("Training...")
for iter in range(max_iters):
    if iter % (eval_iters) == 0 or iter == 0:  # Evaluate every 100 steps or at start
        loss = estimate_loss()  # Compute validation loss
        print(f"Step {iter}, loss: {loss:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")
    optimizer.zero_grad(set_to_none=True)  # Clear gradients
    for _ in range(accumulation_steps):  # Accumulate gradients
        xb, yb = get_batch(prompts, responses)  # Get batch
        with autocast('cuda'):  # Mixed precision
            logits, loss = model(xb, yb)  # Compute loss
        loss = loss / accumulation_steps  # Average loss
        scaler.scale(loss).backward()  # Scaled backprop
    scaler.step(optimizer)  # Update weights
    scaler.update()  # Update scaler
    scheduler.step()  # Adjust learning rate

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)  # Save trained model
print('Model saved.')
