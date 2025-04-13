import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import pandas as pd
import re
from torch.amp import GradScaler, autocast
from collections import Counter
import gc
import os

# Clean up system resources
def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    print("Cleanup completed.")

print("Performing cleanup...")
cleanup()

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

# Hyperparameters
block_size = 128
batch_size = 8
max_iters = 50000
learning_rate = 1e-4
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 10
dropout = 0.2
accumulation_steps = 4

# Load dataset
def load_data(file_path, text_col='text'):
    df = pd.read_parquet(file_path)
    texts = df[text_col].values.tolist()
    return texts

file_path = "C:/Users/Owner/PycharmProjects/LLMProject9/000_00000.parquet"
texts = load_data(file_path)
print(f"Loaded {len(texts)} text entries.")

# Build vocabulary incrementally
def build_vocabulary(texts, max_vocab_size=50000):
    word_counts = Counter()
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    print("Building vocabulary incrementally...")
    for i, text in enumerate(texts):
        if i % 100000 == 0:  # Progress update
            print(f"Processed {i}/{len(texts)} entries...")
        if text is None:
            continue
        # Clean and tokenize each text individually
        text = re.sub(r'[\[\]"\']', ' ', str(text))
        text = re.sub(r'-{2,}', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = re.findall(r'[a-zA-Z]+[-a-zA-Z]*|[0-9]+|[.,$]', text)
        # Update counts for valid tokens
        for token in tokens:
            if len(token) > 1 or token.isdigit() or token in '.,$':
                word_counts[token.lower()] += 1
    # Select top tokens
    vocab = [word for word, _ in word_counts.most_common(max_vocab_size - len(special_tokens))]
    vocab = special_tokens + sorted(set(vocab) - set(special_tokens))
    print(f"Vocabulary built with {len(vocab)} tokens.")
    return vocab

new_words = build_vocabulary(texts)
string_to_int = {word: i for i, word in enumerate(new_words)}
int_to_string = {i: word for i, word in enumerate(new_words)}
vocabulary_size = len(new_words)
print(f"New vocabulary size: {vocabulary_size}")
print("Sample vocabulary words:", new_words[:10])

# Encode and decode functions
def encode(text):
    text = re.sub(r'-{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = re.findall(r'[a-zA-Z]+[-a-zA-Z]*|[0-9]+|[.,$]', text)
    return [string_to_int.get(token.lower(), string_to_int['<UNK>']) for token in tokens]

def decode(tokens):
    return ' '.join([int_to_string[int(i)] for i in tokens if int(i) in int_to_string])

# Batch generation
def get_batch(texts):
    ix = torch.randint(len(texts), (batch_size,))
    batch_texts = [texts[i] for i in ix]
    full_sequences = [encode(f"<SOS> {t} <EOS>") for t in batch_texts]
    x_sequences, y_sequences = [], []
    for seq in full_sequences:
        if len(seq) < 2:
            continue
        x_seq = seq[:-1][:block_size]
        y_seq = seq[1:][:block_size]
        x_seq = x_seq + [string_to_int['<PAD>']] * (block_size - len(x_seq))
        y_seq = y_seq + [-1] * (block_size - len(y_seq))
        x_sequences.append(torch.tensor(x_seq, dtype=torch.long, device=device))
        y_sequences.append(torch.tensor(y_seq, dtype=torch.long, device=device))
    if not x_sequences:
        return get_batch(texts)
    return torch.stack(x_sequences), torch.stack(y_sequences)

# Model classes (unchanged)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        return self.ln2(x + y)

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.9):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == string_to_int['<EOS>']:
                break
        return idx

# Load and adjust pre-trained model
print('Loading pre-trained model...')
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print('Pre-trained model loaded successfully.')
except FileNotFoundError:
    print("Error: 'model.pkl' not found. Please run the original training script first.")
    exit(1)

if model.token_embedding_table.num_embeddings != vocabulary_size:
    print(f"Adjusting model for new vocabulary size: {vocabulary_size}")
    old_embedding = model.token_embedding_table.weight.data
    old_lm_head = model.lm_head.weight.data
    model.token_embedding_table = nn.Embedding(vocabulary_size, n_embd).to(device)
    model.lm_head = nn.Linear(n_embd, vocabulary_size).to(device)
    old_vocab_size = min(old_embedding.shape[0], vocabulary_size)
    model.token_embedding_table.weight.data[:old_vocab_size] = old_embedding[:old_vocab_size]
    model.lm_head.weight.data[:old_vocab_size] = old_lm_head[:old_vocab_size]
    if vocabulary_size > old_vocab_size:
        torch.nn.init.normal_(model.token_embedding_table.weight.data[old_vocab_size:], mean=0.0, std=0.02)
        torch.nn.init.normal_(model.lm_head.weight.data[old_vocab_size:], mean=0.0, std=0.02)

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)
scaler = GradScaler('cuda')

# Estimate loss
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(texts)
        with autocast('cuda'):
            logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

# Training loop
print("Further training on new dataset...")
for iter in range(max_iters):
    if iter % eval_iters == 0 or iter == 0:
        loss = estimate_loss()
        print(f"Step {iter}, loss: {loss:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")
    optimizer.zero_grad(set_to_none=True)
    for _ in range(accumulation_steps):
        xb, yb = get_batch(texts)
        with autocast('cuda'):
            logits, loss = model(xb, yb)
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

# Save updated model
with open('model_updated.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Updated model saved as model_updated.pkl.')
