import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import pandas as pd
import re
from torch.amp import GradScaler, autocast
from collections import Counter

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

# Hyperparameters
block_size = 32
batch_size = 16
max_iters = 5000
learning_rate = 3e-4
eval_iters = 100
n_embd = 192
n_head = 4
n_layer = 6
dropout = 0.2
accumulation_steps = 4


# Step 1: Load data from Parquet file
def load_data(file_path, prompt_col='question', response_col='response'):
    df = pd.read_parquet(file_path)
    print(f"Available columns: {df.columns.tolist()}")
    if prompt_col not in df.columns or response_col not in df.columns:
        raise ValueError(f"Columns '{prompt_col}' or '{response_col}' not found!")
    prompts = df[prompt_col].values.tolist()
    responses = df[response_col].values.tolist()
    print(f"Sample prompts (first 5): {prompts[:5]}")
    print(f"Sample responses (first 5): {responses[:5]}")
    return prompts, responses


file_path = "C:/Users/Owner/PycharmProjects/LLMProject8/1M-GPT4-Augmented.parquet"
prompts, responses = load_data(file_path, prompt_col='question', response_col='response')


# Step 2: Preprocess data and build vocabulary
def build_vocabulary(prompts, responses):
    combined_data = prompts + responses
    text = " ".join(str(t) for t in combined_data if t is not None)
    print(f"Raw text sample (first 200 chars): {text[:200]}")

    # Clean text
    text = re.sub(r'[\[\]"\']', ' ', text)
    text = re.sub(r'-{2,}', ' ', text)
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    print(f"Cleaned text sample (first 200 chars): {text[:200]}")

    # Split into words and count frequencies
    words = text.split()
    word_counts = Counter(words)

    # Filter words: no digits, ASCII only, reasonable hyphen use
    filtered_words = set(word.lower() for word in words if
                         len(word) > 1 and
                         all(c.isalpha() for c in word.replace('-', '')) and  # No digits
                         word.count('-') <= 2 and
                         not word.startswith('-') and
                         all(ord(c) < 128 for c in word))

    # Sort by frequency (most common first) and take top N
    vocab = [word for word, count in word_counts.most_common() if word.lower() in filtered_words]
    print(f"Sample raw words (first 10): {vocab[:10]}")
    return vocab


words = build_vocabulary(prompts, responses)
vocabulary_size = len(words)
print(f"Initial vocabulary size: {vocabulary_size}")

# Limit vocabulary size and add special tokens
special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
if vocabulary_size > 50000:
    words = words[:50000 - len(special_tokens)]
    print("Vocabulary trimmed to 50,000 words (minus special tokens)")
words = special_tokens + sorted(set(words) - set(special_tokens))
string_to_int = {word: i for i, word in enumerate(words)}
int_to_string = {i: word for i, word in enumerate(words)}
vocabulary_size = len(words)
print(f"Final vocabulary size: {vocabulary_size}")
print("Sample vocabulary words:", words[:10])
print("'hello' in vocabulary:", "hello" in string_to_int)


# Tokenization functions
def encode(text):
    text = re.sub(r'-{2,}', ' ', text)
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return [string_to_int.get(word.lower(), string_to_int['<UNK>']) for word in text.split() if word]


def decode(tokens):
    return ' '.join([int_to_string[int(i)] for i in tokens if int(i) in int_to_string])


# Step 3: Prepare training data
def get_batch(prompts, responses):
    ix = torch.randint(len(prompts), (batch_size,))
    batch_prompts = [prompts[i] for i in ix]
    batch_responses = [responses[i] for i in ix]

    x_sequences = [encode(f"<SOS> {p} <EOS>") for p in batch_prompts]
    y_sequences = [encode(f"{r} <EOS>") for r in batch_responses]

    x = torch.stack([torch.nn.functional.pad(
        torch.tensor(seq[:block_size], dtype=torch.long, device=device),
        (0, block_size - len(seq[:block_size])),
        value=string_to_int['<PAD>']
    ) for seq in x_sequences])
    y = torch.stack([torch.nn.functional.pad(
        torch.tensor(seq[:block_size], dtype=torch.long, device=device),
        (0, block_size - len(seq[:block_size])),
        value=-1
    ) for seq in y_sequences])
    return x, y


# Step 4: Define the model (unchanged)
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
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


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
        x = self.ln2(x + y)
        return x


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


# Step 5: Train the model
model = GPTLanguageModel(vocabulary_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = GradScaler('cuda')


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(prompts, responses)
        with autocast('cuda'):
            logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()


print("Training...")
for iter in range(max_iters):
    if iter % eval_iters == 0:
        loss = estimate_loss()
        print(f"Step {iter}, loss: {loss:.4f}")
    optimizer.zero_grad(set_to_none=True)
    for _ in range(accumulation_steps):
        xb, yb = get_batch(prompts, responses)
        with autocast('cuda'):
            logits, loss = model(xb, yb)
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if loss < 0.0001:
        break

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model saved.')
