import os
import requests
import sys
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

# Import the ByteLevelBPETokenizer from Hugging Face's tokenizers library
try:
    from tokenizers import ByteLevelBPETokenizer
except ImportError:
    raise ImportError("Please install the tokenizers library with: pip install tokenizers")

# Updated Hyperparameters
batch_size = 64          # Increased batch size
block_size = 256         # Increased block size for more context (block_size now counts tokens)
C = 384                  # Increased embedding dimension
learning_rate = 3e-4     # Lowered learning rate for stable training
n_layer = 6              # Increased number of layers
num_heads = 6            # Increased number of attention heads
dropout_rate = 0.2       # Dropout rate for regularization
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
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
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
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Dataset saved to {data_file}.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
else:
    print(f"Dataset already exists at {data_file}.")

####################################################################################################
# GPT-2 Style BPE Tokenization Setup

# Define a directory to store (or load) the BPE tokenizer files
bpe_dir = os.path.join(data_dir, "bpe")
vocab_file = os.path.join(bpe_dir, "vocab.json")
merges_file = os.path.join(bpe_dir, "merges.txt")

# If the BPE model files do not exist, train a new tokenizer on the dataset.
if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
    os.makedirs(bpe_dir, exist_ok=True)
    print("Training new BPE tokenizer...")
    tokenizer = ByteLevelBPETokenizer()
    # You can adjust vocab_size as needed; GPT-2 used 50,257 tokens by default.
    tokenizer.train(files=[data_file], vocab_size=50257, min_frequency=2, special_tokens=["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"])
    tokenizer.save_model(bpe_dir)
    print(f"Tokenizer saved to {bpe_dir}.")
else:
    print("Loading existing BPE tokenizer...")
    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)

# Get the vocabulary size from the tokenizer
vocab_size = tokenizer.get_vocab_size()

# Define encode and decode functions that work with tokens rather than characters
def encode(s):
    # Returns a list of token IDs
    return tokenizer.encode(s).ids

def decode(token_ids):
    # Converts a list of token IDs back into a string
    return tokenizer.decode(token_ids)

# Read the dataset and encode it into token IDs
with open(data_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Encode the entire text; note that this produces a long list of token IDs.
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
        B, T, C_ = x.shape

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
        self.relu = nn.ReLU()           # ReLU non-linearity
        self.dropout = nn.Dropout(dropout_rate)  # Dropout for regularization

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

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
        # The embedding layer now uses token IDs (from BPE) rather than characters.
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
        token_embeddings = self.token_embedding_table(idx)
        
        # Positional embeddings (T -> T x C)
        position_idx = torch.arange(T, device=idx.device) % block_size  # Use modulo to wrap around
        positional_embeddings = self.position_embedding_table(position_idx)
        
        # Combine token and positional embeddings
        x = token_embeddings + positional_embeddings

        # Pass through stacked Transformer Blocks
        x = self.transformer_blocks(x)

        # Final LayerNorm before the output projection
        x = self.final_layer_norm(x)
        
        # LM head: project back to vocabulary size
        logits = self.LM_head(x)
        
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # Focus on the last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# Create the BigramLanguageModel with the updated vocabulary size and hyperparameters
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
    B, T, C_ = logits.shape
    logits = logits.view(B * T, C_)
    targets = targets.view(B * T)
    loss = F.cross_entropy(logits, targets)
    return loss

####################################################################################################
# Optimizing for GPU

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

def get_batch(split):
    data_subset = train_data if split == 'train' else val_data
    ix = [random.randint(0, len(data_subset) - block_size - 1) for _ in range(batch_size)]
    x_batch = torch.stack([data_subset[i:i + block_size] for i in ix]).to(device)
    y_batch = torch.stack([data_subset[i + 1:i + block_size + 1] for i in ix]).to(device)
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
    os.makedirs(checkpoint_dir, exist_ok=True)
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Epoch {start_epoch} Loaded")
    return start_epoch

####################################################################################################
# Generate Text

def generate_text():
    """Generates text starting from predefined prompts."""
    scenarios = {
        "Start with '<|bos|>'": "<|bos|> ",
        "Start with 'The'": "The ",
        "Start with 'To be, or not to be, that is the '": "To be, or not to be, that is the "
    }
    for description, start_text in scenarios.items():
        print(f"\n--- {description} ---")
        idx = torch.tensor([encode(start_text)], dtype=torch.long).to(device)
        generated_sequence = model.generate(idx, max_new_tokens=256)
        generated_text = decode(generated_sequence[0].tolist())
        print(generated_text)

####################################################################################################
# Training Loop

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_model(start_step=0):
    print(f"\n[+] Starting training from step {start_step}")
    print(f"[+] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[+] Training configuration:")
    print(f"    - Batch size: {batch_size}")
    print(f"    - Learning rate: {learning_rate}")
    print(f"    - Number of layers: {n_layer}")
    print(f"    - Number of heads: {num_heads}")
    print(f"    - Embedding dimension: {C}")
    
    for step in range(start_step, num_train_steps):
        xb, yb = get_batch('train')
        logits = model(xb)
        loss = compute_loss(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            losses = estimate_loss()
            print(f"\n[+] Step {step}/{num_train_steps} ({step/num_train_steps*100:.1f}%)")
            print(f"    - Train Loss: {losses['train']:.4f}")
            print(f"    - Val Loss: {losses['val']:.4f}")
            print(f"    - Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint and generate sample
            save_checkpoint(model, optimizer, step)
            print("\n[+] Generating sample text...")
            generate_text()

def get_path_to_latest_checkpoint():
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoint_files = os.listdir(checkpoint_dir)
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    latest_checkpoint = checkpoint_files[-1]
    return os.path.join(checkpoint_dir, latest_checkpoint)

####################################################################################################
# Interaction Loop

def help():
    print("Available commands:")
    print("- load <epoch>: Load a model checkpoint from a specific epoch")
    print("- train: Train the model")
    print("- generate <prompt>: Generate text starting from the prompt")
    print("- help: Show available commands")
    print("- exit: Exit the program")

def run():
    print("Welcome to the GPT Text Model Trainer!")
    help()
    while True:
        command = input("Enter a command: ")
        if command == "exit":
            break
        elif command.startswith("load"):
            epoch = int(command.split()[1])
            load(epoch)
        elif command == "train":
            train_model()
        elif command.startswith("generate"):
            # Everything after the command is taken as the prompt.
            prompt = command[len("generate "):]
            generate(prompt)
        else:
            print("Invalid command. Please try again.")

def load(epoch=0):
    print(f"Loading model from epoch {epoch}")
    checkpoint_path = f'checkpoints/model_epoch_{epoch}.pth'
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, optimizer, checkpoint_path)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")

def generate(prompt="The", max_tokens=256):
    print(f"Generating text from prompt: {prompt}")
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    generated_sequence = model.generate(idx, max_new_tokens=max_tokens)
    generated_text = decode(generated_sequence[0].tolist())
    print(generated_text)

def main():
    # Attempt to load from the latest checkpoint if one exists
    start_epoch = 0
    checkpoint_path = get_path_to_latest_checkpoint()
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    run()

if __name__ == '__main__':
    main()
