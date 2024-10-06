import os
import requests
import sys
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 32
block_size = 8
C = 32  # Embedding dimensions
learning_rate = 1e-4
eval_interval = 100
eval_iters = 100
num_train_steps = 10000

# Setting random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

####################################################################################################
# Download the Dataset

datasets = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "gadsby": "https://www.gutenberg.org/files/47367/47367-0.txt"
}

if len(sys.argv) > 1:
    dataset_name = sys.argv[1].lower()  
else:
    dataset_name = "shakespeare"  

if dataset_name not in datasets:
    raise ValueError(f"Dataset '{dataset_name}' is not available. Choose from {list(datasets.keys())}")

url = datasets[dataset_name]

data_dir = os.path.join("data", "local", "text")
data_file = os.path.join(data_dir, f"{dataset_name}.txt")

os.makedirs(data_dir, exist_ok=True)

if not os.path.isfile(data_file):
    print(f"Downloading {dataset_name} dataset from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()  
        with open(data_file, 'w') as f:
            f.write(response.text)
        print(f"Dataset saved to {data_file}.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
else:
    print(f"Dataset already exists at {data_file}.")

####################################################################################################
# Load the Dataset and Create Vocabulary

with open(data_file, 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

####################################################################################################
# Encode and Decode Text

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

####################################################################################################
# Preparing the Validation Set

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

####################################################################################################
# The Self-Attention Head
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(C, head_size, bias=False)
        self.key = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)

        # Register triangular mask as a buffer
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape

        # Compute queries, keys, and values
        Q = self.query(x)  # Shape (B, T, head_size)
        K = self.key(x)    # Shape (B, T, head_size)
        V = self.value(x)  # Shape (B, T, head_size)

        # Compute affinities (Q dot K^T) and apply scaling
        affinities = Q @ K.transpose(-2, -1) / (self.head_size ** 0.5)

        # Apply triangular mask to prevent future tokens from communicating
        affinities = affinities.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(affinities, dim=-1)  # Shape (B, T, T)

        # Aggregate values based on attention weights
        out = attention_weights @ V  # Shape (B, T, head_size)
        return out

####################################################################################################
# The Bigram Language Model with Positional Embeddings

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, C)  # Embedding into C dimensions
        self.position_embedding_table = nn.Embedding(block_size, C)  # Positional Embeddings
        self.attention_head = SelfAttentionHead(C)  # Self-attention head with the same size as embeddings
        self.LM_head = nn.Linear(C, vocab_size)  # Language Modeling Head
    
    def forward(self, idx, targets=None):
        B, T = idx.shape  # B: Batch size, T: Time steps
        
        # Token embeddings (B x T -> B x T x C)
        token_embeddings = self.token_embedding_table(idx)  # Shape (B, T, C)
        
        # Positional embeddings (T -> T x C)
        position_idx = torch.arange(T, device=idx.device) % block_size  # Use modulo to wrap around
        positional_embeddings = self.position_embedding_table(position_idx)  # Shape (T, C)
        
        # Combine token and positional embeddings
        x = token_embeddings + positional_embeddings  # Shape (B, T, C)

        # Pass through self-attention head
        x = self.attention_head(x)  # Shape (B, T, C)

        # Pass the embeddings through the LM head to get logits
        logits = self.LM_head(x)  # Shape (B, T, vocab_size)
        
        return logits
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

            logits = self(idx_cond)  
            logits = logits[:, -1, :]  
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


model = BigramLanguageModel()

####################################################################################################
# Evaluate the Loss Function

def compute_loss(logits, targets):
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = targets.view(B * T)
    loss = F.cross_entropy(logits, targets)
    return loss

####################################################################################################
# Optimizing for GPU

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

def get_batch(split):
    data = train_data if split == 'train' else val_data

    ix = [random.randint(0, len(data) - block_size - 1) for _ in range(batch_size)]
    x_batch = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y_batch = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)

    return x_batch, y_batch

####################################################################################################
# Estimate Loss

def estimate_loss():
    model.eval()
    losses = {'train': [], 'val': []}
    
    with torch.no_grad():
        for split in ['train', 'val']:
            for _ in range(eval_iters):
                xb, yb = get_batch(split)
                logits = model(xb)
                loss = compute_loss(logits, yb)
                losses[split].append(loss.item())

    model.train()
    return {split: sum(losses[split]) / eval_iters for split in losses}

####################################################################################################
# Training Loop

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_model():
    for step in range(num_train_steps):
        xb, yb = get_batch('train')
        logits = model(xb)
        loss = compute_loss(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            losses = estimate_loss()
            print(f"Step {step}, Train Loss: {losses['train']}, Val Loss: {losses['val']}")

train_model()

####################################################################################################
# Generating Text

idx = torch.zeros((1, 1), dtype=torch.long)
generated_sequence = model.generate(idx, max_new_tokens=100)
generated_text = decode(generated_sequence[0].tolist())
print(generated_text)
