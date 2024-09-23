"""
Author: Matthew Del Branson
Date: 2024-01-02

Simple Recurrent Network is a recurrent neural network with one recurrent layer and is trained
on the text contained in the input.txt file and generates text by predicting the next character.
    
"""

# Import necessary libraries for building and training the neural network

import torch  # PyTorch library for tensor computations and deep learning
import torch.nn as nn  # Neural network modules, including layers and loss functions
import numpy as np  # Library for numerical computations
import unidecode  # For normalizing Unicode characters to ASCII
import string  # String processing, provides constants like printable characters
import matplotlib.pyplot as plt  # Library for plotting graphs and visualizations
import logging  # Logging module to track events during execution
import os  # Operating system interface for file operations
from datetime import datetime  # For handling date and time operations
import re  # Regular expressions for string manipulation

# Set up the logging configuration to display the time, log level, and message
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO to capture all INFO and above messages
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the format of the log messages
)
session_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # e.g., '20240102_123456'

# Define the directory paths where figures and models will be saved
figures_directory = "simple_demos/figures"  # Directory to save plots and figures
models_directory = "simple_demos/models"  # Directory to save trained model files

# Create the directories if they do not exist
os.makedirs(figures_directory, exist_ok=True)  # Create figures directory if it doesn't exist
os.makedirs(models_directory, exist_ok=True)  # Create models directory if it doesn't exist

# Initialize empty lists to keep track of training and validation losses over epochs
train_losses = []  # List to store training loss for each epoch
val_losses = []  # List to store validation loss for each epoch

# Define a function to clean and preprocess the text data
def clean_text(file_path):
    """
    Read and clean the text from the given file path.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Cleaned text content.
    """
    # Open the file in binary read mode
    with open(file_path, 'rb') as f:
        content_bytes = f.read()  # Read the content as bytes

    # Try decoding the bytes using UTF-8 encoding
    try:
        content = content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        # If UTF-8 decoding fails, decode using Latin-1 encoding
        content = content_bytes.decode('latin-1')

    # Normalize Unicode characters to ASCII equivalents
    content = unidecode.unidecode(content)

    # Remove any characters that are not printable (non-ASCII)
    content = re.sub(f'[^{re.escape(string.printable)}]', '', content)
    
    return content

# Specify the path to the text file to be used as the corpus
text_file_path = 'data/local/text/gadsby.txt'  # Path to the text file (novel "Gadsby")

# Clean and preprocess the text
text = clean_text(text_file_path)

# Create dictionaries to map characters to integers and vice versa
chars = string.printable  # All printable ASCII characters
int2char = dict(enumerate(chars))  # Map integers to characters
char2int = {char: ind for ind, char in int2char.items()}  # Map characters to integers

# Encode the text into integers using the character to integer mapping
encoded = np.array([char2int[ch] for ch in text if ch in char2int])

# Define hyperparameters for the neural network and training process
input_size = len(char2int)  # Size of the input layer (number of unique characters)
hidden_size = 100  # Number of neurons in the hidden layer
output_size = len(char2int)  # Size of the output layer (same as input_size)
n_layers = 1  # Number of recurrent layers
batch_size = 1  # Number of samples per batch (since we're processing sequences one at a time)
seq_length = 50  # Length of the input sequences
n_epochs = 10  # Number of epochs to train the model
learning_rate = 0.01  # Learning rate for the optimizer

# Define the recurrent neural network model
class SimpleRecurrentNet(nn.Module):
    """
    A simple recurrent neural network for character-level text generation.

    This network predicts the next character in a sequence given the previous characters.

    **Explanation of the Model:**

    - **Embedding Layer**: Converts input characters (represented as integers) into dense vector representations.
    - **Recurrent Layer (RNN)**: Processes the sequence of embeddings and captures temporal dependencies.
    - **Fully Connected Layer**: Maps the output of the RNN to the desired output size (number of unique characters).
    
    Recurrent neural networks are particularly suited for sequence data because they maintain a hidden state that can capture information about previous inputs in the sequence.
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        """
        Initialize the neural network layers.

        Args:
            input_size (int): Size of the input layer (number of unique characters).
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Size of the output layer (number of unique characters).
            n_layers (int): Number of recurrent layers.
        """
        super(SimpleRecurrentNet, self).__init__()
        self.hidden_size = hidden_size  # Store hidden size for later use
        self.n_layers = n_layers  # Store number of layers for later use

        # Embedding layer to convert input indices to dense vectors
        self.embed = nn.Embedding(
            num_embeddings=input_size,  # Size of the dictionary of embeddings (number of unique characters)
            embedding_dim=hidden_size   # The size of each embedding vector
        )

        # Recurrent Neural Network layer
        self.rnn = nn.RNN(
            input_size=hidden_size,  # Input size to the RNN is the embedding size
            hidden_size=hidden_size,  # Hidden state size
            num_layers=n_layers,  # Number of RNN layers
            batch_first=True  # Input and output tensors are provided as (batch, seq, feature)
        )

        # Fully connected layer to map RNN outputs to character logits
        self.fc = nn.Linear(
            in_features=hidden_size,  # Input size from RNN hidden state
            out_features=output_size  # Output size (number of unique characters)
        )

    def forward(self, x, hidden):
        """
        Define the forward pass of the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length).
            hidden (Tensor): Hidden state tensor.

        Returns:
            Tuple[Tensor, Tensor]: Output logits and new hidden state.
        """
        # Pass input through embedding layer
        x = self.embed(x)  # Shape: (batch_size, seq_length, hidden_size)

        # Pass embeddings and hidden state through the RNN layer
        out, hidden = self.rnn(x, hidden)  # out shape: (batch_size, seq_length, hidden_size)

        # Pass the RNN outputs through the fully connected layer
        out = self.fc(out)  # Shape: (batch_size, seq_length, output_size)

        return out, hidden

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state to zeros.

        Args:
            batch_size (int): Batch size.

        Returns:
            Tensor: Initialized hidden state tensor.
        """
        # Create a tensor of zeros for the hidden state
        return torch.zeros(
            self.n_layers,  # Number of layers
            batch_size,     # Batch size
            self.hidden_size  # Hidden size
        )

# Initialize the model with the specified hyperparameters
model = SimpleRecurrentNet(input_size, hidden_size, output_size, n_layers)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(
    model.parameters(),  # Parameters to optimize
    lr=learning_rate     # Learning rate
)  # Using Adam optimizer for efficient training
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification tasks

# Define the filename and path to save the model
model_filename = f"simple_rnn_model_{session_start_time}.pth"  # Filename includes timestamp
model_path = os.path.join(models_directory, model_filename)  # Full path to save the model

# Save the model's initial state dictionary (parameters)
torch.save(model.state_dict(), model_path)  # Save the model's parameters to the specified path

# Log that the model has been saved
logging.info(f"Model saved to {model_path}")

# Define the training and validation function
def train_and_validate(model, criterion, optimizer, epochs=10):
    """
    Train and validate the model over a number of epochs.

    Args:
        model (nn.Module): The neural network model to train.
        criterion (nn.Module): The loss function to optimize.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        epochs (int): Number of epochs to train the model.
    """
    logging.info("Training and validating the model...")
    try:
        # Loop over the number of epochs
        for epoch in range(1, epochs + 1):
            model.train()  # Set the model to training mode
            hidden = model.init_hidden(batch_size)  # Initialize the hidden state
            optimizer.zero_grad()  # Reset the gradients of the optimizer
            loss = 0  # Initialize cumulative loss

            # Loop over the text data in steps of seq_length
            for char in range(0, len(encoded) - seq_length, seq_length):
                # Prepare input and target sequences
                input_seq = torch.LongTensor(encoded[char:char + seq_length]).unsqueeze(0)  # Shape: (1, seq_length)
                target_seq = torch.LongTensor(encoded[char + 1:char + seq_length + 1]).unsqueeze(0)  # Next chars

                # Forward pass
                output, hidden = model(input_seq, hidden)
                # Calculate loss (flatten outputs and targets to compute CrossEntropyLoss)
                loss += criterion(
                    output.view(-1, output_size),  # Reshape outputs to (batch_size * seq_length, output_size)
                    target_seq.view(-1)            # Reshape targets to (batch_size * seq_length)
                )

            # Backward pass and optimization
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

            # Record the average training loss for this epoch
            avg_train_loss = loss.item() / (len(encoded) // seq_length)
            train_losses.append(avg_train_loss)

            # Validation phase (using the same data for simplicity)
            val_loss = 0  # Initialize validation loss
            hidden = model.init_hidden(batch_size)  # Reset hidden state
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Disable gradient calculation
                for char in range(0, len(encoded) - seq_length, seq_length):
                    input_seq = torch.LongTensor(encoded[char:char + seq_length]).unsqueeze(0)
                    target_seq = torch.LongTensor(encoded[char + 1:char + seq_length + 1]).unsqueeze(0)

                    output, hidden = model(input_seq, hidden)
                    val_loss += criterion(
                        output.view(-1, output_size),
                        target_seq.view(-1)
                    ).item()

            # Record the average validation loss for this epoch
            average_val_loss = val_loss / (len(encoded) // seq_length)
            val_losses.append(average_val_loss)

            # Log the training and validation losses
            logging.info(f"Epoch {epoch}, Training Loss: {avg_train_loss}, Validation Loss: {average_val_loss}")

    except KeyboardInterrupt:
        # Handle interruption gracefully
        logging.info("Training interrupted by user.")
        return
    except Exception as e:
        # Log any exceptions that occur
        logging.error(e)
        return

# Define a function to plot the training and validation losses
def plot_training_and_validation_loss(train_losses, val_losses):
    """
    Plot the training and validation loss over epochs.

    Args:
        train_losses (list): List of average training losses per epoch.
        val_losses (list): List of average validation losses per epoch.
    """
    plt.figure(figsize=(10, 5))  # Create a new figure with specified size
    plt.title("Training and Validation Loss")  # Set the title of the plot
    plt.plot(train_losses, label="Train")  # Plot training loss
    plt.plot(val_losses, label="Validation")  # Plot validation loss
    plt.xlabel("Epoch")  # Label for x-axis
    plt.ylabel("Loss")  # Label for y-axis
    plt.legend()  # Display legend to differentiate between training and validation lines
    # Define the path to save the loss plot
    loss_fig_path = os.path.join(
        figures_directory,
        f"simple_rnn_loss_plot_{session_start_time}.png"
    )
    plt.savefig(loss_fig_path)  # Save the plot to the specified path
    logging.info(f"Loss plot saved to {loss_fig_path}")  # Log the save event
    plt.show()  # Display the plot

# Train and validate the model
train_and_validate(model, criterion, optimizer, n_epochs)

# Plot the training and validation losses
plot_training_and_validation_loss(train_losses, val_losses)

logging.info("Training complete.")

# Define functions for generating text using the trained model
def predict(model, char, hidden=None, k=1):
    """
    Given a character, predict the next character.

    Args:
        model (nn.Module): The trained model.
        char (str): The current character.
        hidden (Tensor): The hidden state.
        k (int): The number of top characters to consider.

    Returns:
        Tuple[str, Tensor]: Predicted next character and new hidden state.
    """
    # Convert the input character to a tensor
    x = np.array([[char2int[char]]])  # Shape: (1, 1)
    x = torch.LongTensor(x)

    # Detach the hidden state to prevent backpropagating through the entire training history
    if hidden is not None:
        hidden = hidden.detach()

    # Forward pass through the model
    out, hidden = model(x, hidden)

    # Apply softmax to get probabilities of the next character
    prob = nn.functional.softmax(out, dim=2).data  # Shape: (1, 1, output_size)

    # Get top k characters and their probabilities
    prob, top_char = prob.topk(k)  # prob and top_char shapes: (1, 1, k)
    prob = prob.cpu().numpy().squeeze()  # Convert probabilities to numpy array
    top_char = top_char.cpu().numpy().squeeze()  # Convert top characters to numpy array

    # Ensure that prob and top_char are arrays
    if k == 1:
        prob = np.array([prob])
        top_char = np.array([top_char])

    # Randomly choose one of the top characters as the next character, based on probabilities
    char = np.random.choice(top_char, p=prob / prob.sum())

    # Return the predicted character and hidden state
    return int2char[char], hidden

def sample(model, out_len, start='the'):
    """
    Generate text by sampling from the model.

    Args:
        model (nn.Module): The trained model.
        out_len (int): Desired length of the generated text.
        start (str): Starting string for the text generation.

    Returns:
        str: Generated text.
    """
    model.eval()  # Set the model to evaluation mode
    start = start.lower()  # Convert the starting string to lowercase
    chars = [ch for ch in start]  # Initialize the list of generated characters with the starting string
    size = out_len - len(chars)  # Calculate the remaining number of characters to generate

    hidden = model.init_hidden(1)  # Initialize hidden state for a batch size of 1
    for ch in start:
        # For each character in the starting string, predict the next character
        char, hidden = predict(model, ch, hidden, k=1)

    chars.append(char)  # Append the first predicted character to the list

    for _ in range(size):
        # Predict the next character based on the last character generated
        char, hidden = predict(model, chars[-1], hidden, k=3)  # Consider top 3 characters
        chars.append(char)  # Append the predicted character to the list

    return ''.join(chars)  # Combine the list of characters into a single string

# Generate and log a sample of generated text
generated_text = sample(model, 1000, 'the')  # Generate 1000 characters starting with 'the'

# Log the generated text
logging.info(generated_text)
