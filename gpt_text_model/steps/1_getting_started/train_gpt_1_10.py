import os
import requests
import sys

import torch


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
        
    # Print vocabulary size and characters
    print(f"Vocabulary size: {vocab_size}")
    print(f"Vocabulary: {chars}")

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

# Test the encoding and decoding
test_string = "hi there"
encoded = encode(test_string)
decoded = decode(encoded)

print(f"Original: {test_string}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")

# Encode the entire dataset into a sequence of integers
data = torch.tensor(encode(text), dtype=torch.long)

# Print the first 1,000 elements of the tensor
print(data[:1000])

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

print(f"Input chunk (x): {x}")
print(f"Target chunk (y): {y}")

####################################################################################################
# Creating Multiple Examples from one Chunk

for t in range(block_size):
    context = x[:t + 1]  # Context is all characters up to and including position t
    target = y[t]  # Target is the t-th character in the target chunk
    print(f"Context: {context} -> Target: {target}")

