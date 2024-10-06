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
learning_rate = 1e-3
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
# The Bigram Language Model

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        # Embed the tokens (B x T -> B x T x C)
        x = self.token_embedding_table(idx)  # Shape (B, T, C)

        # Initialize the bag-of-words tensor (same shape as input data: B x T x C)
        B, T, C = x.shape
        X_bag_of_words = torch.zeros_like(x)

        # Calculate feature vector by averaging the previous tokens
        for b in range(B):
            for t in range(T):
                # Get all previous tokens (including the current token)
                X_prev = x[b, :t + 1]  # (T_prev, C), where T_prev = t + 1
                
                # Compute the average along the time dimension
                feature_vector = X_prev.mean(dim=0)  # Shape (C)
                
                # Store the result in X_bag_of_words
                X_bag_of_words[b, t] = feature_vector

        # (Optionally: proceed with other operations like generating logits)
        logits = self.token_embedding_table(idx)  # Placeholder for next steps
        return logits

    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)  
            logits = logits[:, -1, :]  
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = BigramLanguageModel(vocab_size)

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
