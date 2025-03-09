import mmap
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

# Hyperparameters
block_size = 64
batch_size = 128
max_iters_pretrain = 10000  # Pretraining iterations
max_iters_finetune = 2000  # Fine-tuning iterations
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 8
n_layer = 8
dropout = 0.2

# Load vocabulary and add special tokens
with open("C:/Users/Owner/PycharmProjects/LLMProject8/vocab.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text))
special_tokens = ['[PROMPT]', '[RESPONSE]']
chars.extend(special_tokens)
vocabulary_size = len(chars)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s if c in string_to_int]
decode = lambda l: ''.join([int_to_string[int(i)] for i in l if int(i) in int_to_string])

# Pretraining data loading
def get_random_chunk(split):
    filename = "C:/Users/Owner/PycharmProjects/LLMProject8/output_train.txt" if split == 'train' else "C:/Users/Owner/PycharmProjects/LLMProject8/output_val.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, max(0, file_size - block_size * batch_size))
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data

def get_batch_pretrain(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Fine-tuning data loading
def load_finetune_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return [line.split('|||') for line in lines if '|||' in line]

finetune_data = load_finetune_data("C:/Users/Owner/PycharmProjects/LLMProject8/dialogue_data.txt")

def get_batch_finetune():
    ix = torch.randint(len(finetune_data), (batch_size,))
    batch = [finetune_data[i] for i in ix]
    x = [encode(f"[PROMPT]{prompt}[RESPONSE]{response}") for prompt, response in batch]
    x = [torch.tensor(xi[:block_size], dtype=torch.long, device=device) for xi in x]
    y = [torch.tensor(xi[1:block_size+1], dtype=torch.long, device=device) for xi in x]
    x = torch.stack([torch.nn.functional.pad(xi, (0, block_size - len(xi))) for xi in x])
    y = torch.stack([torch.nn.functional.pad(yi, (0, block_size - len(yi)), value=-1) for yi in y])
    return x, y

# Model definition (unchanged except for generation)
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
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
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

class FeedFoward(nn.Module):
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
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocabulary_size)
        self.apply(self._int_weights)

    def _int_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
        return logits, loss

# Training loop
model = GPTLanguageModel(vocabulary_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss(get_batch_func, name):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch_func('train' if name == 'pretrain' else None)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

# Pretraining
print("Pretraining...")
for iter in range(max_iters_pretrain):
    if iter % eval_iters == 0:
        loss = estimate_loss(get_batch_pretrain, 'pretrain')
        print(f"Pretrain step {iter}, loss: {loss:.4f}")
    xb, yb = get_batch_pretrain('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Fine-tuning
print("Fine-tuning...")
for iter in range(max_iters_finetune):
    if iter % eval_iters == 0:
        loss = estimate_loss(get_batch_finetune, 'finetune')
        print(f"Finetune step {iter}, loss: {loss:.4f}")
    xb, yb = get_batch_finetune()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save model
with open('C:/Users/Owner/PycharmProjects/LLMProject8/model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model saved.')
