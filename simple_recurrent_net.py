"""
Author: Matthew Del Branson
Date: 2024-01-02

Simple Recurrent Network is a recurrent neural network with one recurrent layer and is trained
on the text contained in the input.txt file and generates text by predicting the next character.
    
"""

import torch
import torch.nn as nn
import numpy as np
import unidecode
import string
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
session_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# Check for the existence of the directory to save figures and models
figures_directory = "figures"
models_directory = "models"
os.makedirs(figures_directory, exist_ok=True)
os.makedirs(models_directory, exist_ok=True)

# Initialize lists to store loss and accuracy values
train_losses = []
val_losses = []

# Clean the text
def clean_text(file_path):
    with open(file_path, 'rb') as f:
        content_bytes = f.read()

    try:
        content = content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        content = content_bytes.decode('latin-1')

    content = unidecode.unidecode(content)
    content = re.sub(f'[^{re.escape(string.printable)}]', '', content)
    
    return content

text_file_path = 'data/local/input.txt'
text = clean_text(text_file_path)

# Create a dictionary of characters and indices
chars = string.printable
int2char = dict(enumerate(chars))
char2int = {char: ind for ind, char in int2char.items()}

# Encode the text
encoded = np.array([char2int[ch] for ch in text if ch in char2int])

# Hyperparameters
input_size = len(char2int)
hidden_size = 100
output_size = len(char2int)
n_layers = 1
batch_size = 1
seq_length = 50  # sequence length
n_epochs = 100
learning_rate = 0.01

# Define the model
class SimpleRecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(SimpleRecurrentNet, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

# Initialize the model
model = SimpleRecurrentNet(input_size, hidden_size, output_size, n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Saving the model with versioning
model_filename = f"simple_rnn_model_{session_start_time}.pth"
model_path = os.path.join(models_directory, model_filename)
torch.save(model.state_dict(), model_path)
logging.info(f"Model saved to {model_path}")

# Training and validation
def train_and_validate(model, criterion, optimizer, epochs=10):
    """Trains and validates the model.

    Args:
        model (torch.nn.Module): Model to train and validate.
        criterion (torch.nn.modules.loss): Loss function.
        optimizer (torch.optim): Optimizer.
        epochs (int, optional): Number of epochs. Defaults to 10.
    """
    logging.info("Training and validating the model...")
    try:
        # Training Loop
        for epoch in range(1, epochs + 1):
            model.train()
            hidden = model.init_hidden(batch_size)
            optimizer.zero_grad()
            loss = 0

            for char in range(0, len(encoded) - seq_length, seq_length):
                # Get batch data
                input_seq = torch.LongTensor(encoded[char:char + seq_length]).unsqueeze(0)
                target_seq = torch.LongTensor(encoded[char + 1:char + seq_length + 1]).unsqueeze(0)

                # Forward pass
                output, hidden = model(input_seq, hidden)
                loss += criterion(output.squeeze(), target_seq.squeeze())

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item() / (len(encoded) // seq_length))
        
            # Validation Loop
            val_loss = 0
            hidden = model.init_hidden(batch_size)
            model.eval()
            with torch.no_grad():
                for char in range(0, len(encoded) - seq_length, seq_length):
                    input_seq = torch.LongTensor(encoded[char:char + seq_length]).unsqueeze(0)
                    target_seq = torch.LongTensor(encoded[char + 1:char + seq_length + 1]).unsqueeze(0)

                    output, hidden = model(input_seq, hidden)
                    val_loss += criterion(output.squeeze(), target_seq.squeeze()).item()

            average_val_loss = val_loss / (len(encoded) // seq_length)
            val_losses.append(average_val_loss)

            logging.info(f"Epoch {epoch}, Training Loss: {loss.item() / (len(encoded) // seq_length)}, Validation Loss: {average_val_loss}")

    except KeyboardInterrupt:
        logging.info("Interrupted")
        return
    except Exception as e:
        logging.error(e)
        return

# Clean the text
def clean_text(file_path):
    with open(file_path, 'rb') as f:
        content_bytes = f.read()

    try:
        content = content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        content = content_bytes.decode('latin-1')

    content = unidecode.unidecode(content)
    content = re.sub(f'[^{re.escape(string.printable)}]', '', content)
    
    return content

# Plot the training and validation loss
def plot_training_and_validation_loss(train_losses, val_losses):
    """Plots the training and validation loss.

    Args:
        train_losses (list): Training loss.
        val_losses (list): Validation loss.
    """
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    loss_fig_path = os.path.join(figures_directory, f"rnn_loss_plot_{session_start_time}.png")
    plt.savefig(loss_fig_path)
    logging.info(f"Loss plot saved to {loss_fig_path}")
    plt.show()

train_and_validate(model, criterion, optimizer, n_epochs)
plot_training_and_validation_loss(train_losses, val_losses)

logging.info("Training complete.")


# Generate text
def predict(model, char, hidden=None, k=1):
    x = np.array([[char2int[char]]])
    x = torch.LongTensor(x)

    if hidden is not None:
        hidden = hidden.detach()

    out, hidden = model(x, hidden)

    prob = nn.functional.softmax(out, dim=2).data
    prob, top_char = prob.topk(k)
    top_char = top_char.numpy().squeeze()

    char = np.random.choice(top_char.flatten(), 1)[0]

    return int2char[char], hidden

def sample(model, out_len, start='the'):
    model.eval()
    start = start.lower()
    chars = [ch for ch in start]
    size = out_len - len(chars)

    hidden = model.init_hidden(1)
    for ch in start:
        char, hidden = predict(model, ch, hidden, k=1)

    chars.append(char)

    for ii in range(size):
        char, hidden = predict(model, chars[-1], hidden, k=3)
        chars.append(char)

    return ''.join(chars)

logging.info(sample(model, 1000, 'the'))


