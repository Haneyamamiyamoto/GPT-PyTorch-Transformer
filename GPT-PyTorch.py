import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast, get_linear_schedule_with_warmup
from tqdm import tqdm

# ---------------- Configuration ----------------
parser = argparse.ArgumentParser(description="Improved GPT Training Script")
parser.add_argument('--train_file', type=str, required=True, help='Path to training text file')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--block_size', type=int, default=128)
parser.add_argument('--max_iters', type=int, default=3000)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--warmup_steps', type=int, default=200)
parser.add_argument('--eval_interval', type=int, default=50)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
args = parser.parse_args()

device = args.device

# --------------- Tokenization ----------------
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')  # BPE tokenizer

# --------------- Dataset ----------------------
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        tokens = tokenizer(texts, return_tensors='pt', add_special_tokens=False).input_ids[0]
        self.data = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y

# Load text and split
with open(args.train_file, 'r', encoding='utf-8') as f:
    text = f.read()
train_text = text[:int(0.8*len(text))]
val_text = text[int(0.8*len(text)):]

train_ds = TextDataset(train_text, tokenizer, args.block_size)
val_ds = TextDataset(val_text, tokenizer, args.block_size)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size)

# -------------- Model -------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1,2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1,2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) * (self.head_size ** -0.5)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.dropout(nn.Linear(C, C)(y))

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # pre-LN
        x = x + self.ff(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        token_embeddings = self.token_emb(idx)
        positions = torch.arange(T, device=idx.device)
        pos_embeddings = self.pos_emb(positions).unsqueeze(0)
        x = token_embeddings + pos_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -------------- Training ----------------------
model = GPTModel(
    vocab_size=tokenizer.vocab_size,
    n_embd=384,
    n_head=6,
    n_layer=6,
    block_size=args.block_size,
    dropout=args.dropout
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

total_steps = args.max_iters
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=total_steps
)

scaler = torch.cuda.amp.GradScaler()

for step in tqdm(range(total_steps)):
    model.train()
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)

    with torch.cuda.amp.autocast():
        logits, loss = model(xb, yb)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()

    if step % args.eval_interval == 0:
        model.eval()
        losses = []
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                _, l = model(xv, yv)
                losses.append(l.item())
        print(f"Step {step} | Val Loss: {sum(losses)/len(losses):.4f} | PPL: {torch.exp(torch.tensor(sum(losses)/len(losses))):.2f}")

# -------------- Generation ----------------------
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None):
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature
            # top-k
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_val = values[:, -1].unsqueeze(1)
                logits = torch.where(logits < min_val, torch.full_like(logits, -float('Inf')), logits)
            # top-p
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                cutoff = cumulative_probs > top_p
                sorted_logits[cutoff] = -float('Inf')
                logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
    return tokenizer.decode(tokens[0].tolist())

# Example generation
if __name__ == '__main__':
    prompt = "Hello! Can you see me?"
    print(generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9))
