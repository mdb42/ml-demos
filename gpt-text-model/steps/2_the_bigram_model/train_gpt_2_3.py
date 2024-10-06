import os
import requests
import sys

import torch
import random

import torch.nn as nn

####################################################################################################
# Download the Dataset

# URLs for different datasets
datasets = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "gadsby": "https://www.gutenberg.org/files/47367/47367-0.txt"
}

# Allow dataset selection via command-line argument
if len(sys.argv) > 1:
    dataset_name = sys.argv[1].lower()  # Get dataset name from command line
else:
    dataset_name = "shakespeare"  # Default dataset

# Check if the chosen dataset exists in the dictionary
if dataset_name not in datasets:
    raise ValueError(f"Dataset '{dataset_name}' is not available. Choose from {list(datasets.keys())}")

url = datasets[dataset_name]

# Use os.path.join to ensure cross-platform compatibility
data_dir = os.path.join("data", "local", "text")
data_file = os.path.join(data_dir, f"{dataset_name}.txt")

# Make sure data directory exists
os.makedirs(data_dir, exist_ok=True)

# Download the dataset if it's not already present
if not os.path.isfile(data_file):
    print(f"Downloading {dataset_name} dataset from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for successful request
        with open(data_file, 'w') as f:
            f.write(response.text)
        print(f"Dataset saved to {data_file}.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
else:
    print(f"Dataset already exists at {data_file}.")

####################################################################################################
# Load the Dataset and Create Vocabulary

# Read the dataset into a string
with open(data_file, 'r') as f:
    text = f.read()

# Create a set of all unique characters in the text (vocabulary)
chars = sorted(list(set(text)))
vocab_size = len(chars)

####################################################################################################
# Encode and Decode Text

# Create encoder and decoder mappings
stoi = {ch: i for i, ch in enumerate(chars)}  # String to integer
itos = {i: ch for i, ch in enumerate(chars)}  # Integer to string

# Encoder: Convert string to list of integers
def encode(s):
    return [stoi[c] for c in s]

# Decoder: Convert list of integers back to string
def decode(l):
    return ''.join([itos[i] for i in l])

# Encode the entire dataset into a sequence of integers
data = torch.tensor(encode(text), dtype=torch.long)

####################################################################################################
# Preparing the Validation Set

# Split the dataset into training and validation sets
n = int(0.9 * len(data))  # 90% of the data for training
train_data = data[:n]
val_data = data[n:]

# Print the sizes of the training and validation sets
print(f"Training data size: {len(train_data)} characters")
print(f"Validation data size: {len(val_data)} characters")

####################################################################################################
# Chunking the Data (Block Size)

# Set block size (context length)
block_size = 8

# Get a sample chunk of the training set
x = train_data[:block_size]  # Input chunk of size block_size
y = train_data[1:block_size + 1]  # Target chunk, offset by one character

####################################################################################################
# Creating Multiple Examples from one Chunk

for t in range(block_size):
    context = x[:t + 1]  # Context is all characters up to and including position t
    target = y[t]  # Target is the t-th character in the target chunk

####################################################################################################
# Batching the Data

batch_size = 4  # Number of chunks processed in parallel
x_batch = torch.stack([train_data[i:i + block_size] for i in range(0, batch_size * block_size, block_size)])
y_batch = torch.stack([train_data[i + 1:i + block_size + 1] for i in range(0, batch_size * block_size, block_size)])

print(f"Input batch (x_batch): {x_batch}")
print(f"Target batch (y_batch): {y_batch}")

####################################################################################################
# Get a Batch of Data

# Set random seed for reproducibility
random.seed(42)

# Function to get a batch of data
def get_batch(split, block_size=8, batch_size=4):
    """
    Generate a batch of input-output pairs for training or validation.

    Args:
        split (str): 'train' or 'val', determines which dataset to use.
        block_size (int): Length of each sequence in the batch.
        batch_size (int): Number of sequences processed in parallel.
    
    Returns:
        x_batch (torch.Tensor): Input batch.
        y_batch (torch.Tensor): Target batch.
    """
    data = train_data if split == 'train' else val_data
    
    # Generate random starting positions for the batch
    ix = [random.randint(0, len(data) - block_size - 1) for _ in range(batch_size)]
    
    # Stack the input and target sequences
    x_batch = torch.stack([data[i:i + block_size] for i in ix])
    y_batch = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    
    return x_batch, y_batch

# Get a batch of training data
xb, yb = get_batch('train')

# Print the input and target batches
print(f"Input batch (xb): {xb}")
print(f"Target batch (yb): {yb}")

####################################################################################################
# Define the Model

# Define the Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Token embedding table (vocab_size x vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx)  # (B, T, C), where C is vocab_size
        
        # Return the logits (predictions for the next character)
        return logits

# Instantiate the model
model = BigramLanguageModel(vocab_size)

# Test the model with a batch of input data
logits = model(xb)

print(f"Logits shape: {logits.shape}")  # Should be (batch_size, block_size, vocab_size)
