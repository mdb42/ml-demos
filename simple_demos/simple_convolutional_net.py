"""
Author: Matthew Del Branson
Date: 2024-09-22

Simple Convolutional Network is a convolutional neural network with one convolutional layer, 
one pooling layer, and one fully connected layer and is trained on the MNIST dataset.

"""

# Import necessary libraries for building and training the neural network

import torch  # PyTorch library for tensor computations and deep learning
import torch.nn as nn  # Neural network modules, including layers and loss functions
import torch.optim as optim  # Optimization algorithms like SGD, Adam, etc.
import torchvision  # Library for computer vision datasets and models
import torchvision.transforms as transforms  # Transformations for image preprocessing
from torch.utils.data import DataLoader, random_split  # Utilities for data loading and splitting
import matplotlib.pyplot as plt  # Library for plotting graphs and visualizations
import seaborn as sns  # Statistical data visualization library
from sklearn.metrics import confusion_matrix  # For computing confusion matrix
import logging  # Logging module to track events during execution
import os  # Operating system interface for file operations
from datetime import datetime  # For handling date and time operations

# Set up the logging configuration to display the time, log level, and message
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO to capture all INFO and above messages
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the format of the log messages
)

# Get the current date and time, formatted as a string, to use in filenames
session_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # e.g., '20240101_123456'

# Define the directory paths where figures and models will be saved
figures_directory = "simple_demos/figures"  # Directory to save plots and figures
models_directory = "simple_demos/models"  # Directory to save trained model files

# Create the directories if they do not exist
os.makedirs(figures_directory, exist_ok=True)  # Create figures directory if it doesn't exist
os.makedirs(models_directory, exist_ok=True)  # Create models directory if it doesn't exist

# Initialize empty lists to keep track of training and validation losses and validation accuracies over epochs
train_losses = []  # List to store training loss for each epoch
val_losses = []  # List to store validation loss for each epoch
val_accuracies = []  # List to store validation accuracy for each epoch

# Define hyperparameters for the neural network and training process
number_of_output_classes = 10  # Number of classes for classification (digits 0-9)
learning_rate = 0.001  # Learning rate for the optimizer
batch_size = 64  # Number of samples processed before updating the model parameters
num_epochs = 10  # Number of times the entire dataset is passed through the network

# Define a sequence of image transformations to apply to the dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to a PyTorch tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the tensor with mean and standard deviation
    # Mean and std are given for the MNIST dataset
])

# Load the MNIST dataset for training
dataset = torchvision.datasets.MNIST(
    root='./data',  # Directory to store the dataset
    train=True,  # Specify to load the training set
    download=True,  # Download the dataset if it's not already present
    transform=transform  # Apply the defined transformations to the data
)

# Calculate the number of samples for training and validation datasets
num_train = int(len(dataset) * 0.8)  # 80% of the total dataset for training
num_val = len(dataset) - num_train   # Remaining 20% for validation

# Randomly split the dataset into training and validation subsets
train_dataset, val_dataset = random_split(
    dataset,  # The original dataset to split
    [num_train, num_val]  # Sizes of the splits
)

# Create data loaders to efficiently load data in batches during training and validation
train_loader = DataLoader(
    train_dataset,  # The training dataset
    batch_size=batch_size,  # Number of samples per batch
    shuffle=True  # Shuffle the data at every epoch
)

val_loader = DataLoader(
    val_dataset,  # The validation dataset
    batch_size=batch_size,  # Number of samples per batch
    shuffle=True  # Shuffle the data
)

# Define a simple convolutional neural network
class SimpleConvolutionalNet(nn.Module):
    """
    A simple convolutional neural network with one convolutional layer, one pooling layer,
    and one fully connected layer.

    This class defines the architecture of the convolutional neural network, including its layers and the forward pass.
    It inherits from torch.nn.Module, which is the base class for all neural network modules in PyTorch.

    **Difference from the Feedforward Network:**

    - Instead of flattening the input image and using fully connected layers, this network uses convolutional and pooling layers to exploit the spatial structure of the images.
    - Convolutional layers are better suited for image data as they can capture local patterns and hierarchical features.
    - This architecture often leads to improved performance and faster convergence compared to simple feedforward networks on image tasks.

    As a result, this model demonstrates better accuracy in fewer epochs compared to the previous feedforward model.
    """
    def __init__(self):
        """
        Initialize the neural network layers.

        This method sets up the layers of the network.
        """
        # Call the superclass constructor
        super(SimpleConvolutionalNet, self).__init__()

        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=1,  # Number of input channels (grayscale images have 1 channel)
            out_channels=32,  # Number of filters (kernels) to apply
            kernel_size=5,  # Size of the filter (5x5)
            stride=1,  # Stride of the convolution
            padding=2  # Zero-padding added to both sides of the input
        )

        # Note: Using a convolutional layer allows the network to learn spatial hierarchies of features.
        # Unlike the feedforward network which flattens the image, the convolutional layer preserves spatial information.

        # Define the activation function (Rectified Linear Unit)
        self.relu = nn.ReLU()

        # Define the max pooling layer
        self.maxpool = nn.MaxPool2d(
            kernel_size=2,  # Size of the window to take a max over (2x2)
            stride=2  # Stride of the window
        )

        # Note: Max pooling reduces the spatial dimensions (height and width) by taking the maximum value over a window.
        # This helps to reduce computational complexity and extract dominant features.

        # Define the fully connected layer
        # After convolution and pooling, the image size is reduced
        # Original image: 28x28
        # After conv layer (with padding=2 and stride=1): still 28x28
        # After pooling layer: 14x14
        # With 32 output channels from the conv layer, the input to the fc layer is 14*14*32
        self.fc1 = nn.Linear(
            in_features=14*14*32,  # Number of input features
            out_features=number_of_output_classes  # Number of output classes (10)
        )

    def forward(self, x):
        """
        Define the forward pass of the network.

        This method defines how the input data flows through the network layers during forward propagation.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor: Output logits from the network
        """
        # Pass the input through the convolutional layer
        x = self.conv1(x)  # Output shape: (batch_size, 32, 28, 28)
        x = self.relu(x)   # Apply activation function

        # Pass the result through the max pooling layer
        x = self.maxpool(x)  # Output shape: (batch_size, 32, 14, 14)

        # Flatten the tensor to feed into the fully connected layer
        x = x.view(-1, 14*14*32)  # Reshape to (batch_size, 14*14*32)

        # Pass through the fully connected layer
        x = self.fc1(x)  # Output shape: (batch_size, 10)

        return x

    def predict(self, x):
        """
        Make predictions on input data without computing gradients.

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Predicted class indices
        """
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Perform the forward pass
            x = self.forward(x)

            # Get the index of the class with the highest score (logit)
            predictions = torch.argmax(x, dim=1)

            return predictions

def train_and_validate(model, loss_function, optimizer, epochs=10):
    """
    Train and validate the neural network model.

    This function handles the training loop over the specified number of epochs,
    computes the training and validation losses, and calculates validation accuracy.

    Args:
        model (nn.Module): The neural network model to train
        loss_function (nn.Module): The loss function to optimize
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters
        epochs (int): Number of epochs to train the model
    """
    try:
        # Loop over the number of epochs
        for epoch in range(epochs):
            running_loss = 0.0  # Cumulative loss for the current epoch
            model.train()  # Set the model to training mode (enables dropout, batchnorm, etc.)

            # Iterate over batches of training data
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data  # Get the inputs and labels from the data loader

                optimizer.zero_grad()  # Reset the gradients of the optimizer

                outputs = model(inputs)  # Forward pass: compute the model output
                loss = loss_function(outputs, labels)  # Compute the loss between outputs and true labels
                loss.backward()  # Backward pass: compute gradients
                optimizer.step()  # Update model parameters based on computed gradients

                running_loss += loss.item()  # Accumulate the loss for reporting

            # Compute and log the average training loss for the epoch
            avg_train_loss = running_loss / len(train_loader)
            logging.info(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

            # Validation phase
            val_loss = 0.0  # Cumulative validation loss
            model.eval()  # Set the model to evaluation mode (disables dropout, batchnorm, etc.)
            with torch.no_grad():  # Disable gradient calculation for validation
                for data in val_loader:
                    inputs, labels = data
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    val_loss += loss.item()

            # Compute and log the average validation loss for the epoch
            avg_val_loss = val_loss / len(val_loader)
            logging.info(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

            # Validation accuracy calculation
            correct_predictions = 0.0  # Count of correct predictions
            total_samples = 0  # Total number of samples
            model.eval()
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data
                    outputs = model.predict(inputs)
                    correct_predictions += torch.sum(outputs == labels).item()
                    total_samples += labels.size(0)

            # Compute and log the validation accuracy for the epoch
            val_accuracy = correct_predictions / total_samples
            logging.info(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}")

            # Append losses and accuracy to the lists for visualization later
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

    except KeyboardInterrupt:
        # Handle the case where the user interrupts the training process
        logging.info("Training interrupted by user.")
        pass
    except Exception as e:
        # Handle any other exceptions that may occur
        logging.error(e)
        pass

def plot_training_and_validation_loss(train_losses, val_losses):
    """
    Plot the training and validation loss over epochs.

    This function generates a line plot showing how the training and validation losses change over time.

    Args:
        train_losses (list): List of average training losses per epoch
        val_losses (list): List of average validation losses per epoch
    """
    plt.figure(figsize=(10, 5))  # Create a new figure with specified size
    plt.title("Training and Validation Loss")  # Set the title of the plot
    plt.plot(train_losses, label="Training")  # Plot training loss
    plt.plot(val_losses, label="Validation")  # Plot validation loss
    plt.xlabel("Epochs")  # Label for x-axis
    plt.ylabel("Loss")  # Label for y-axis
    plt.legend()  # Display legend to differentiate between training and validation lines

    # Define the path to save the loss plot
    loss_fig_path = os.path.join(
        figures_directory,
        f"simple_cnn_loss_plot_{session_start_time}.png"
    )

    plt.savefig(loss_fig_path)  # Save the plot to the specified path
    logging.info(f"Loss plot saved to {loss_fig_path}")  # Log the save event
    plt.show()  # Display the plot

def plot_validation_accuracy(val_accuracies):
    """
    Plot the validation accuracy over epochs.

    This function generates a line plot showing how the validation accuracy changes over time.

    Args:
        val_accuracies (list): List of validation accuracies per epoch
    """
    plt.figure(figsize=(10, 5))  # Create a new figure with specified size
    plt.title("Validation Accuracy")  # Set the title of the plot
    plt.plot(val_accuracies, label="Validation Accuracy")  # Plot validation accuracy
    plt.xlabel("Epochs")  # Label for x-axis
    plt.ylabel("Accuracy")  # Label for y-axis
    plt.legend()  # Display legend

    # Define the path to save the accuracy plot
    acc_fig_path = os.path.join(
        figures_directory,
        f"simple_cnn_accuracy_plot_{session_start_time}.png"
    )

    plt.savefig(acc_fig_path)  # Save the plot to the specified path
    logging.info(f"Accuracy plot saved to {acc_fig_path}")  # Log the save event
    plt.show()  # Display the plot

def plot_confusion_matrix(cm):
    """
    Plot the confusion matrix as a heatmap.

    This function visualizes the confusion matrix to analyze the performance of the classifier.

    Args:
        cm (numpy.ndarray): Confusion matrix array
    """
    # Create a heatmap using seaborn
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")  # fmt="d" formats the annotations as integers
    plt.xlabel("Predicted")  # Label for x-axis
    plt.ylabel("Actual")  # Label for y-axis
    plt.title("Confusion Matrix")  # Title of the plot

    # Define the path to save the confusion matrix plot
    cm_fig_path = os.path.join(
        figures_directory,
        f"simple_cnn_confusion_matrix_{session_start_time}.png"
    )

    plt.savefig(cm_fig_path)  # Save the plot to the specified path
    logging.info(f"Confusion matrix saved to {cm_fig_path}")  # Log the save event
    plt.show()  # Display the plot

# Create an instance of the convolutional neural network model
model = SimpleConvolutionalNet()  # Initializes the model with defined architecture

# Define the loss function and optimizer for training
criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multi-class classification
optimizer = optim.Adam(
    model.parameters(),  # Parameters (weights and biases) of the model to optimize
    lr=learning_rate  # Learning rate for the optimizer
)

# Define the filename and path to save the model
model_filename = f"simple_cnn_model_{session_start_time}.pth"  # Filename includes timestamp
model_path = os.path.join(models_directory, model_filename)  # Full path to save the model

# Save the model's state dictionary (parameters)
torch.save(model.state_dict(), model_path)  # Save the model's parameters to the specified path

# Log that the model has been saved
logging.info(f"Model saved to {model_path}")

# Call the function to train and validate the model
train_and_validate(
    model,  # The neural network model to train
    criterion,  # The loss function
    optimizer,  # The optimizer
    epochs=num_epochs  # Number of epochs to train
)

# Plot the training and validation losses over epochs
plot_training_and_validation_loss(
    train_losses,  # List of training losses per epoch
    val_losses  # List of validation losses per epoch
)

# Plot the validation accuracy over epochs
plot_validation_accuracy(
    val_accuracies  # List of validation accuracies per epoch
)

# Compute and plot the confusion matrix using the validation set

# Initialize empty lists to store all predictions and labels
all_preds = []  # List to store all predicted labels
all_labels = []  # List to store all actual labels

# Disable gradient calculations during inference
with torch.no_grad():
    # Iterate over the validation data loader
    for inputs, labels in val_loader:
        outputs = model.predict(inputs)  # Get the predicted class indices
        all_preds.extend(outputs)  # Add predictions to the list
        all_labels.extend(labels)  # Add actual labels to the list

# Compute the confusion matrix
cm = confusion_matrix(all_labels, all_preds)  # Rows: actual labels, Columns: predicted labels

# Plot the confusion matrix
plot_confusion_matrix(cm)

# Indicate that training is complete
logging.info("Training complete.")
