import os
import requests
import sys
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

# Updated Hyperparameters
batch_size = 64  # Increased batch size
block_size = 256  # Increased block size for more context
C = 384  # Increased embedding dimension
learning_rate = 3e-4  # Lowered learning rate for stable training
n_layer = 6  # Increased number of layers
num_heads = 6  # Increased number of attention heads
dropout_rate = 0.2  # Dropout rate for regularization
eval_interval = 100
eval_iters = 100
num_train_steps = 10000
head_size = C // num_heads  # Size of each head (C is the total embedding dimension)

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
# The Self-Attention Head with Dropout

class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, dropout_rate):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(C, head_size, bias=False)
        self.key = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout after attention weights

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        affinities = Q @ K.transpose(-2, -1) / (self.head_size ** 0.5)
        affinities = affinities.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        attention_weights = F.softmax(affinities, dim=-1)
        attention_weights = self.dropout(attention_weights)  # Apply dropout to attention weights

        out = attention_weights @ V
        return out

####################################################################################################
# Multi-Head Self-Attention with Dropout

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, dropout_rate):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size, dropout_rate) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, C)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout after projection

    def forward(self, x):
        # Concatenate the outputs of all heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # Apply dropout before projecting back to original size
        return self.dropout(self.projection(out))

####################################################################################################
# Feedforward Layer with Dropout
class FeedForwardLayer(nn.Module):
    def __init__(self, C, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(C, 4 * C)  # Expand to 4x the embedding size
        self.fc2 = nn.Linear(4 * C, C)  # Project back down to embedding size
        self.relu = nn.ReLU()  # ReLU non-linearity
        self.dropout = nn.Dropout(dropout_rate)  # Dropout for regularization

    def forward(self, x):
        x = self.fc1(x)  # First linear transformation
        x = self.relu(x)  # Apply ReLU
        x = self.dropout(x)  # Apply dropout after ReLU
        return self.fc2(x)  # Project back to original dimension

####################################################################################################
# Transformer Block with Dropout

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, head_size, dropout_rate):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(num_heads, head_size, dropout_rate)
        self.feed_forward = FeedForwardLayer(C, dropout_rate)
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        self.dropout = nn.Dropout(dropout_rate)  # Apply dropout with the given rate

    def forward(self, x):
        # Multi-head attention with dropout
        x = x + self.dropout(self.multi_head_attention(self.ln1(x)))
        # Feedforward layer with dropout
        x = x + self.dropout(self.feed_forward(self.ln2(x)))
        return x

####################################################################################################
# The Bigram Language Model with Dynamic Hyperparameters

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, C, n_layer, num_heads, dropout_rate):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, C)
        self.position_embedding_table = nn.Embedding(block_size, C)

        head_size = C // num_heads  # Calculate head size based on embedding dimension and number of heads
        
        # Stack Transformer Blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(num_heads, head_size, dropout_rate) for _ in range(n_layer)]
        )

        self.final_layer_norm = nn.LayerNorm(C)  # Final LayerNorm before output
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

        # Pass through stacked Transformer Blocks
        x = self.transformer_blocks(x)  # Shape (B, T, C)

        # Final LayerNorm before the output projection
        x = self.final_layer_norm(x)  # Normalize before output
        
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

# Create the BigramLanguageModel with the required arguments
model = BigramLanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    C=C,
    n_layer=n_layer,
    num_heads=num_heads,
    dropout_rate=dropout_rate
)


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
# Save the Model

def save_checkpoint(model, optimizer, epoch, checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

####################################################################################################
# Load the Model
def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch

####################################################################################################
# Training Loop

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_model(start_step=0):
    for step in range(start_step, num_train_steps):
        xb, yb = get_batch('train')
        logits = model(xb)
        loss = compute_loss(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            losses = estimate_loss()
            print(f"Step {step}, Train Loss: {losses['train']}, Val Loss: {losses['val']}")
            # Save checkpoint at each evaluation interval
            save_checkpoint(model, optimizer, step)

# Attempt to load from a checkpoint
start_epoch = 0
checkpoint_path = 'checkpoints/model_epoch_9999.pth'  # Replace with your checkpoint path
if os.path.exists(checkpoint_path):
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

# Start or resume training
train_model(start_step=start_epoch)

####################################################################################################
# Generating Text

idx = torch.zeros((1, 1), dtype=torch.long)
generated_sequence = model.generate(idx, max_new_tokens=100)
generated_text = decode(generated_sequence[0].tolist())
print(generated_text)
