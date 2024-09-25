"""
Author: Matthew Del Branson
Date: 2024-09-24

This script implements an optimized LSTM-based recurrent neural network for character-level text generation.
Key features:
- **LSTM Implementation**: Utilizes LSTM layers for better handling of long-term dependencies.
- **Batch Processing**: Employs PyTorch's DataLoader and Dataset for efficient batch training.
- **Data Splitting**: Splits data into training and validation sets to monitor performance and prevent overfitting.
- **GPU Acceleration**: Automatically uses GPU if available to speed up training.
- **Optimized Training Loop**: Streamlined for clarity and performance with proper hidden state management.
- **Regularization Techniques**: Incorporates weight decay, dropout, gradient clipping, and learning rate scheduling to mitigate overfitting.
- **Early Stopping**: Implements early stopping to prevent overfitting when validation loss increases.
- **Enhanced Text Generation**: Generates coherent and diverse text by sampling from predicted character probabilities.
- **Logging and Visualization**: Provides detailed logging and plots training and validation losses for monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import unidecode
import string
import re
import os
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# ---------------------- Configuration and Setup ----------------------

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)
session_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# Set up directories for saving models and figures
figures_directory = "improved_lstm/figures"
models_directory = "improved_lstm/models"

os.makedirs(figures_directory, exist_ok=True)
os.makedirs(models_directory, exist_ok=True)

# Device configuration: Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# ---------------------- Data Preprocessing ----------------------

def clean_text(file_path):
    """
    Read and clean the text from the given file path.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Cleaned text content.
    """
    with open(file_path, 'rb') as f:
        content_bytes = f.read()

    # Try decoding with UTF-8, fallback to Latin-1
    try:
        content = content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        content = content_bytes.decode('latin-1')

    # Normalize Unicode to ASCII and remove non-printable characters
    content = unidecode.unidecode(content)
    content = re.sub(f'[^{re.escape(string.printable)}]', '', content)

    return content

# Load and preprocess the text data
text_file_path = 'data/local/text/gadsby.txt'  # Update this path as needed
text = clean_text(text_file_path)
logging.info(f"Text length after cleaning: {len(text)} characters")

# Create character mappings
all_characters = sorted(list(set(text)))
num_characters = len(all_characters)
char2int = {ch: i for i, ch in enumerate(all_characters)}
int2char = {i: ch for i, ch in enumerate(all_characters)}

# Encode the entire text
encoded_text = np.array([char2int[ch] for ch in text])
logging.info(f"Number of unique characters: {num_characters}")

# Split data into training and validation sets
train_ratio = 0.9  # 90% for training, 10% for validation
train_data_len = int(len(encoded_text) * train_ratio)
train_data = encoded_text[:train_data_len]
val_data = encoded_text[train_data_len:]
logging.info(f"Training data length: {len(train_data)}")
logging.info(f"Validation data length: {len(val_data)}")

# Create sequences and corresponding targets
seq_length = 100  # Length of input sequences
batch_size = 64    # Number of samples per batch

def create_sequences(data, seq_length):
    """
    Create input sequences and their corresponding targets.

    Args:
        data (np.array): Array of encoded characters.
        seq_length (int): Length of each input sequence.

    Returns:
        Tuple[List[List[int]], List[List[int]]]: Sequences and targets.
    """
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + 1:i + seq_length + 1])  # Shifted by one
    return sequences, targets

train_sequences, train_targets = create_sequences(train_data, seq_length)
val_sequences, val_targets = create_sequences(val_data, seq_length)
logging.info(f"Number of training sequences: {len(train_sequences)}")
logging.info(f"Number of validation sequences: {len(val_sequences)}")

# Define custom Dataset
class TextDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )

# Create Datasets and DataLoaders
train_dataset = TextDataset(train_sequences, train_targets)
val_dataset = TextDataset(val_sequences, val_targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
logging.info("DataLoaders created successfully.")

# ---------------------- Model Definition ----------------------

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.5):
        """
        Initialize the LSTM-based neural network.

        Args:
            vocab_size (int): Number of unique characters.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout probability between LSTM layers.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer to convert character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer to map LSTM outputs to character logits
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length).
            hidden (Tuple[Tensor, Tensor]): Tuple of (hidden_state, cell_state).

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: Output logits and new hidden states.
        """
        x = self.embedding(x)  # (batch_size, seq_length, hidden_size)
        out, hidden = self.lstm(x, hidden)  # out: (batch_size, seq_length, hidden_size)
        out = out.contiguous().view(-1, self.hidden_size)  # (batch_size * seq_length, hidden_size)
        out = self.fc(out)  # (batch_size * seq_length, vocab_size)
        return out, hidden

    def init_hidden(self, batch_size):
        """
        Initialize hidden state and cell state to zeros.

        Args:
            batch_size (int): Batch size.

        Returns:
            Tuple[Tensor, Tensor]: Initialized hidden and cell states.
        """
        weight = next(self.parameters()).data
        hidden = (
            weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        )
        return hidden

# ---------------------- Hyperparameters and Initialization ----------------------

# Hyperparameters
vocab_size = num_characters
hidden_size = 256
num_layers = 2
dropout = 0.5
learning_rate = 0.002
num_epochs = 20
patience = 5  # For early stopping

# Initialize the model, optimizer, and loss function
model = LSTMModel(vocab_size, hidden_size, num_layers, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay for regularization
criterion = nn.CrossEntropyLoss()
logging.info("Model, optimizer, and loss function initialized.")

# Initialize the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

# ---------------------- Training and Evaluation Functions ----------------------

def detach(hidden):
    """
    Detach hidden states from their history to prevent backpropagating through entire training history.

    Args:
        hidden (Tuple[Tensor, Tensor]): Hidden and cell states.

    Returns:
        Tuple[Tensor, Tensor]: Detached hidden and cell states.
    """
    return tuple([h.detach() for h in hidden])

def train_epoch(model, data_loader, criterion, optimizer, epoch):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        data_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epoch (int): Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    hidden = model.init_hidden(batch_size)

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        hidden = detach(hidden)
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output, target.view(-1))
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            logging.info(f'Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}')

    average_loss = total_loss / len(data_loader)
    return average_loss

def evaluate(model, data_loader, criterion):
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        data_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            hidden = detach(hidden)
            output, hidden = model(data, hidden)
            loss = criterion(output, target.view(-1))
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    return average_loss

# ---------------------- Early Stopping Implementation ----------------------

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                logging.info(f'Initial validation loss: {val_loss:.4f}')
            return

        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                logging.info(f'Validation loss decreased to {val_loss:.4f}. Resetting early stopping counter.')
        else:
            self.counter += 1
            logging.info(f'Validation loss did not improve. Early stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

# ---------------------- Training Loop with Early Stopping ----------------------

train_losses = []
val_losses = []
early_stopping = EarlyStopping(patience=patience, verbose=True)

logging.info("Starting training...")

for epoch in range(1, num_epochs + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
    val_loss = evaluate(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    logging.info(f'Epoch {epoch} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}')

    # Check early stopping condition
    early_stopping(val_loss)
    scheduler.step(val_loss)  # Update the learning rate based on validation loss
    if early_stopping.early_stop:
        logging.info("Early stopping triggered. Stopping training.")
        break

# ---------------------- Saving the Model ----------------------

# Define the filename and path to save the model
model_filename = f"lstm_model_{session_start_time}.pth"
model_path = os.path.join(models_directory, model_filename)

# Save the trained model's state dictionary
torch.save(model.state_dict(), model_path)
logging.info(f"Trained model saved to {model_path}")

# ---------------------- Plotting Losses ----------------------

def plot_losses(train_losses, val_losses):
    """
    Plot training and validation losses over epochs.

    Args:
        train_losses (List[float]): List of training losses.
        val_losses (List[float]): List of validation losses.
    """
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # Define the path to save the loss plot
    loss_fig_path = os.path.join(
        figures_directory,
        f"lstm_loss_plot_{session_start_time}.png"
    )
    plt.savefig(loss_fig_path)
    logging.info(f"Loss plot saved to {loss_fig_path}")
    plt.show()

# Plot the training and validation losses
plot_losses(train_losses, val_losses)

logging.info("Training complete.")

# ---------------------- Text Generation Functions ----------------------

def predict(model, char, hidden, temperature=1.0):
    """
    Given a character, predict the next character.

    Args:
        model (nn.Module): The trained model.
        char (str): The current character.
        hidden (Tuple[Tensor, Tensor]): The hidden and cell states.
        temperature (float): Controls the randomness of predictions.

    Returns:
        Tuple[str, Tuple[Tensor, Tensor]]: Predicted next character and new hidden state.
    """
    model.eval()
    char_tensor = torch.tensor([[char2int[char]]], dtype=torch.long).to(device)
    with torch.no_grad():
        output, hidden = model(char_tensor, hidden)
    output_dist = output.data.view(-1).div(temperature).exp()
    top_char = torch.multinomial(output_dist, 1)[0]
    predicted_char = int2char[top_char.item()]
    return predicted_char, hidden

def sample(model, starting_str='The', len_generated=200, temperature=1.0):
    """
    Generate text by sampling from the model.

    Args:
        model (nn.Module): The trained model.
        starting_str (str): Starting string for the text generation.
        len_generated (int): Desired length of the generated text.
        temperature (float): Controls the randomness of predictions.

    Returns:
        str: Generated text.
    """
    model.eval()
    hidden = model.init_hidden(1)
    generated_str = starting_str

    # Use the starting string to warm up the hidden state
    for char in starting_str[:-1]:
        char, hidden = predict(model, char, hidden, temperature)

    char = starting_str[-1]
    for _ in range(len_generated):
        char, hidden = predict(model, char, hidden, temperature)
        generated_str += char

    return generated_str

# ---------------------- Generate and Log Sample Text ----------------------

generated_text = sample(model, starting_str='The', len_generated=1000, temperature=0.8)
logging.info("Generated Text:")
logging.info(generated_text)
