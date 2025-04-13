import torch  # Library for tensor operations and neural network computations
import torch.nn as nn  # Neural network components and layers
from torch.nn import functional as F  # Functional utilities like softmax and padding
import pickle  # For loading the updated trained model
import pandas as pd  # For handling Parquet data files
import re  # Regular expressions for text preprocessing
from collections import Counter  # For counting tokens to build vocabulary
import gc  # Garbage collection for memory management
import os  # Operating system utilities for file operations

# Clean up system resources to optimize performance
def cleanup():
    if torch.cuda.is_available():  # Check if GPU is available
        torch.cuda.empty_cache()  # Release unused GPU memory
        torch.cuda.ipc_collect()  # Collect GPU memory fragments
    gc.collect()  # Free up Python memory

print("Performing cleanup...")
cleanup()

# Set the computation device (GPU preferred, CPU as fallback)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

# Define hyperparameters consistent with the updated model
block_size = 128    # Maximum sequence length for input/output (matches training)
n_embd = 384        # Embedding dimension for token representations (matches training)
n_head = 6          # Number of attention heads (n_embd / n_head = 64, matches training)
n_layer = 10        # Number of transformer layers (matches training)
dropout = 0.2       # Dropout rate to prevent overfitting (matches training)
temperature = 0.9   # Temperature for sampling (matches updated training for diversity)
repetition_penalty = 1.5  # Penalty to discourage token repetition (matches updated training)

# Load data from the new dataset to rebuild vocabulary
def load_data(file_path, text_col='text'):
    df = pd.read_parquet(file_path)  # Read the new Parquet file into a DataFrame
    texts = df[text_col].values.tolist()  # Extract text column as a list
    return texts

# Specify the path to the new dataset used for further training
file_path = "C:/Users/Owner/PycharmProjects/LLMProject9/000_00000.parquet"  # Updated to new dataset path
texts = load_data(file_path)  # Load texts from new dataset

# Reconstruct vocabulary from the new dataset to match the updated model
def build_vocabulary(texts):
    combined_data = texts  # Use the text list directly
    text = " ".join(str(t) for t in combined_data if t is not None)  # Join into a single string
    text = re.sub(r'[\[\]"\']', ' ', text)  # Remove brackets and quotes
    text = re.sub(r'-{2,}', ' ', text)  # Replace multiple hyphens with a space
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    words = re.findall(r'[a-zA-Z]+[-a-zA-Z]*|[0-9]+|[.,$]', text)  # Tokenize into words, numbers, punctuation
    word_counts = Counter(words)  # Count frequency of each token
    filtered_words = set(word.lower() for word in words if
                         len(word) > 1 or word.isdigit() or word in '.,$')  # Filter valid tokens
    vocab = [word for word, count in word_counts.most_common() if word.lower() in filtered_words]  # Sort by frequency
    return vocab

words = build_vocabulary(texts)  # Generate vocabulary from new dataset
special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']  # Define special tokens (consistent with training)
if len(words) > 50000:  # Cap vocabulary at 50,000 tokens
    words = words[:50000 - len(special_tokens)]
words = special_tokens + sorted(set(words) - set(special_tokens))  # Add special tokens and ensure uniqueness
string_to_int = {word: i for i, word in enumerate(words)}  # Create word-to-index mapping
int_to_string = {i: word for i, word in enumerate(words)}  # Create index-to-word mapping
vocabulary_size = len(words)  # Set vocabulary size
print(f"Final vocabulary size: {vocabulary_size}")
print("Sample vocabulary words:", words[:10])

# Encode text into integer tokens using the new vocabulary
def encode(text):
    text = re.sub(r'-{2,}', ' ', text)  # Collapse multiple hyphens
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    tokens = re.findall(r'[a-zA-Z]+[-a-zA-Z]*|[0-9]+|[.,$]', text)  # Split into tokens
    return [string_to_int.get(token.lower(), string_to_int['<UNK>']) for token in tokens]  # Map to integers

# Decode integer tokens back to readable text
def decode(tokens):
    filtered_tokens = []  # List to store valid tokens
    for token in tokens:
        if token == string_to_int['<EOS>']:  # Stop decoding at end-of-sequence token
            break
        if token not in [string_to_int['<PAD>'], string_to_int['<UNK>']]:  # Skip padding and unknown tokens
            filtered_tokens.append(token)
    return ' '.join([int_to_string[int(i)] for i in filtered_tokens if int(i) in int_to_string])  # Join into string

# Define a single attention head for the transformer model
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)  # Linear layer for key projection
        self.query = nn.Linear(n_embd, head_size, bias=False)  # Linear layer for query projection
        self.value = nn.Linear(n_embd, head_size, bias=False)  # Linear layer for value projection
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # Causal mask for autoregression
        self.dropout = nn.Dropout(dropout)  # Dropout to regularize attention weights

    def forward(self, x):
        B, T, C = x.shape  # Batch size, sequence length, embedding size
        k = self.key(x)  # Compute keys from input
        q = self.query(x)  # Compute queries from input
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # Calculate scaled dot-product attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Mask future tokens
        wei = F.softmax(wei, dim=-1)  # Normalize attention weights
        wei = self.dropout(wei)  # Apply dropout to weights
        v = self.value(x)  # Compute values from input
        return wei @ v  # Return weighted sum of values

# Define multi-head attention combining multiple heads
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # List of attention heads
        self.proj = nn.Linear(head_size * num_heads, n_embd)  # Projection layer to combine head outputs
        self.dropout = nn.Dropout(dropout)  # Dropout after projection

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate outputs from all heads
        return self.dropout(self.proj(out))  # Project and apply dropout

# Define feed-forward network for post-attention processing
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand embedding size by 4x
            nn.ReLU(),  # Apply ReLU activation for non-linearity
            nn.Linear(4 * n_embd, n_embd),  # Contract back to original size
            nn.Dropout(dropout),  # Apply dropout for regularization
        )

    def forward(self, x):
        return self.net(x)  # Process input through the network

# Define a transformer block combining attention and feed-forward
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head  # Calculate size per attention head
        self.sa = MultiHeadAttention(n_head, head_size)  # Multi-head self-attention layer
        self.ffwd = FeedForward(n_embd)  # Feed-forward network layer
        self.ln1 = nn.LayerNorm(n_embd)  # Layer normalization before attention
        self.ln2 = nn.LayerNorm(n_embd)  # Layer normalization before feed-forward

    def forward(self, x):
        y = self.sa(x)  # Apply self-attention
        x = self.ln1(x + y)  # Add residual connection and normalize
        y = self.ffwd(x)  # Apply feed-forward
        return self.ln2(x + y)  # Add residual connection and normalize

# Define the GPT language model architecture
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # Embedding layer for tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # Embedding layer for positions
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])  # Stack of transformer blocks
        self.ln_f = nn.LayerNorm(n_embd)  # Final normalization layer
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Output layer for token prediction
        self.apply(self._init_weights)  # Apply custom weight initialization

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):  # For linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Normal initialization
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # Zero initialization for biases
        elif isinstance(module, nn.Embedding):  # For embedding layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Normal initialization

    def forward(self, idx, targets=None):
        B, T = idx.shape  # Batch size and sequence length
        tok_emb = self.token_embedding_table(idx)  # Convert token indices to embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # Generate position embeddings
        x = tok_emb + pos_emb  # Combine token and position embeddings
        x = self.blocks(x)  # Process through transformer blocks
        x = self.ln_f(x)  # Apply final normalization
        logits = self.lm_head(x)  # Predict next token logits
        if targets is None:  # Inference mode (no loss computation)
            loss = None
        else:  # Training mode (not used here)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # Flatten for loss computation
            targets = targets.view(B * T)  # Flatten targets
            loss = F.cross_entropy(logits, targets, ignore_index=-1)  # Compute loss
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.9, repetition_penalty=1.5):
        recent_tokens = set()  # Set to track recently generated tokens for penalty
        for _ in range(max_new_tokens):  # Generate tokens iteratively
            idx_cond = idx[:, -block_size:]  # Use last block_size tokens as context
            logits, _ = self.forward(idx_cond)  # Predict next token logits
            logits = logits[:, -1, :] / temperature  # Scale logits by temperature
            for token in recent_tokens:  # Apply repetition penalty to recent tokens
                logits[0, token] /= repetition_penalty
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample next token
            idx = torch.cat((idx, idx_next), dim=1)  # Append to sequence
            recent_tokens.add(idx_next.item())  # Add to recent tokens set
            if len(recent_tokens) > 10:  # Limit penalty window to last 10 tokens
                recent_tokens.pop()
            if idx_next.item() == string_to_int['<EOS>']:  # Stop at end-of-sequence token
                break
        return idx

# Load the updated model with error handling
model = GPTLanguageModel(vocabulary_size)  # Initialize model structure
model_file = 'model_updated.pkl'  # Path to the updated model
print('Loading updated model parameters...')
try:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)  # Load the updated model weights
    print('Updated model loaded successfully.')
except FileNotFoundError:
    print(f"Error: {model_file} not found. Please ensure the updated model has been trained and saved.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

m = model.to(device)  # Move model to the computation device
m.eval()  # Set model to evaluation mode for inference

# Interactive loop for user prompts
while True:
    prompt = input("Prompt:\n")  # Get user input
    if prompt.lower() == 'goodbye':  # Exit condition
        break
    encoded_prompt = encode(f"<SOS> {prompt}")  # Encode prompt with start token
    context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)  # Convert to tensor
    context = F.pad(context, (0, block_size - context.shape[1]), value=string_to_int['<PAD>'])  # Pad to block_size
    max_new_tokens = min(500, len(encoded_prompt) * 5)  # Dynamically set max generation length
    generated = m.generate(context, max_new_tokens=max_new_tokens)  # Generate response
    decoded_output = decode(generated[0].tolist())  # Decode generated tokens to text
    print(f'Completion: \n{decoded_output}')  # Display the generated completion
