import os
import requests
import sys

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