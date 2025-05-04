
import torch, math
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.pe = pe.unsqueeze(0).to(device)

    def forward(self, x): return x + self.pe[:, :x.size(1)]

# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(device)
        self.heads = heads

    def forward(self, x):
        B, L, D = x.size()
        qkv = self.qkv(x).view(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        scores = (q @ k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, D)
        return self.out(out)

# Feedforward Layer
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x): return self.net(x)

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads)
        self.ffn = FeedForward(d_model, hidden_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        return self.norm2(x + self.drop(self.ffn(x)))

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, heads, layers, hidden_dim, vocab_size, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(d_model, heads, hidden_dim) for _ in range(layers)])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.pos_enc(self.embed(x))
        for layer in self.enc_layers: x = layer(x)
        return self.out(x)

# Hyperparameters & Training
d_model, heads, layers, hidden_dim, vocab_size, max_len = 128, 8, 6, 512, 10000, 100
model = TransformerEncoder(d_model, heads, layers, hidden_dim, vocab_size, max_len).to(device)
criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=1e-3)

# Dummy training loop
def train_model(epochs=10):
    for epoch in range(epochs):
        x = torch.randint(0, vocab_size, (32, max_len)).to(device)
        y = torch.randint(0, vocab_size, (32, max_len)).to(device)
        optimizer.zero_grad()
        loss = criterion(model(x).view(-1, vocab_size), y.view(-1))
        loss.backward(); optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

train_model()
