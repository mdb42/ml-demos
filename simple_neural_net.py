"""
Author: Matthew Del Branson
Date: 2024-01-01
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for the existence of the directory to save figures and models
figures_directory = "figures"
models_directory = "models"
os.makedirs(figures_directory, exist_ok=True)
os.makedirs(models_directory, exist_ok=True)

# Define hyperparameters
number_of_input_features = 784 # 28 x 28 = 784
hidden_layer_size = 100 # Number of neurons in the hidden layer
number_of_output_classes = 10 # Number of classes in the output layer
learning_rate = 0.001 # Learning rate
batch_size = 64 # Number of samples per batch
num_epochs = 10 # Number of epochs

# Initialize lists to store loss and accuracy values
train_losses = []
val_losses = []
val_accuracies = []

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(), # Convert image to tensor
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize the image for illumination differences
])

# Define the dataset
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Define sizes for training and validation sets
num_train = int(len(dataset) * 0.8)  # e.g., 80% of the dataset
num_val = len(dataset) - num_train   # e.g., 20% of the dataset

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

# Define data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


class SimpleNeuralNet(nn.Module):
    """Defines the SimpleNeuralNet Class:
        Creates a class that inherits from nn.Module.
        Defines the layers in the __init__ method.
        Implements the forward pass in the forward method.
        
        Args:
            nn (Module): Inherits from nn.Module.

        Returns:
            SimpleNeuralNet: A neural network with one hidden layer.
        
        """
    
    def __init__(self):
        """Initializes the neural network.

        Args:
            self (SimpleNeuralNet): An instance of SimpleNeuralNet.
        
        """
        super(SimpleNeuralNet, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(in_features=number_of_input_features, out_features=hidden_layer_size) # First layer
        self.relu = nn.ReLU() # Non-linearity
        self.fc2 = nn.Linear(in_features=hidden_layer_size, out_features=number_of_output_classes) # Second layer

    def forward(self, x):
        """Implements the forward pass.

        Args:
            x (Tensor): Input data.

        Returns:
            Tensor: The output of the network.
        
        """
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)  # First layer
        x = self.relu(x)  # Non-linearity
        x = self.fc2(x)  # Second layer
        return x

    def predict(self, x):
        """Predicts the class of the input data.
        
        Args:
            x (Tensor): Input data.
        
        Returns:
            Tensor: The predicted class of the input data.
        
        """
        with torch.no_grad():
            x = self.forward(x) # Forward pass
            return torch.argmax(x, dim=1) # Return the class with the highest probability

def train_and_validate(model, loss_function, optimizer, epochs=10):
    try:
        # Training loop
        for epoch in range(epochs):
            running_loss = 0.0  # Initialize the loss for training
            model.train()  # Set the model to training mode

            # Training loop
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                optimizer.zero_grad()  # zero the parameter gradients

                outputs = model(inputs)  # forward pass
                loss = loss_function(outputs, labels)  # compute the loss
                loss.backward()  # backward pass
                optimizer.step()  # update weights

                running_loss += loss.item()  # accumulate the loss
            logging.info(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}")

            # Validation loop
            val_loss = 0.0  # Initialize the loss for validation
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Turn off gradients for validation
                for data in val_loader:
                    inputs, labels = data
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    val_loss += loss.item()

            logging.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}")

            # Validation accuracy
            acc = 0.0
            model.eval()
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data
                    outputs = model.predict(inputs)
                    acc += torch.sum(outputs == labels).item()

            logging.info(f"Epoch {epoch + 1}, Validation Accuracy: {acc / len(val_dataset)}")
            # Append losses for visualization
            train_losses.append(running_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(acc / len(val_dataset))
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
        pass
    except Exception as e:
        logging.error(e)
        pass

def plot_training_and_validation_loss(train_losses, val_losses):
    """Plots the training and validation loss.

    Args:
        train_losses (list): Training loss.
        val_losses (list): Validation loss.
    
    """
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="Training")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    loss_fig_path = os.path.join(figures_directory, f"loss_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(loss_fig_path)
    logging.info(f"Loss plot saved to {loss_fig_path}")
    plt.show()

def plot_validation_accuracy(val_accuracies):
    """Plots the validation accuracy.

    Args:
        val_accuracies (list): Validation accuracy.
    
    """
    plt.figure(figsize=(10, 5))
    plt.title("Validation Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    acc_fig_path = os.path.join(figures_directory, f"accuracy_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(acc_fig_path)
    logging.info(f"Accuracy plot saved to {acc_fig_path}")
    plt.show()

def plot_confusion_matrix(cm):
    """Plots the confusion matrix.

    Args:
        cm (numpy.ndarray): Confusion matrix.
    
    """
    # Compute and plot the confusion matrix for the validation set
    
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_fig_path = os.path.join(figures_directory, f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(cm_fig_path)
    logging.info(f"Confusion matrix saved to {cm_fig_path}")
    plt.show()

# Instantiate the network
model = SimpleNeuralNet()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For a classification problem
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer

# Saving the model with versioning
model_filename = f"simple_nn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
model_path = os.path.join(models_directory, model_filename)
torch.save(model.state_dict(), model_path)
logging.info(f"Model saved to {model_path}")

# Train and validate the model
train_and_validate(model, criterion, optimizer, epochs=num_epochs)

# Plot the training and validation loss
plot_training_and_validation_loss(train_losses, val_losses)

# Plot the validation accuracy
plot_validation_accuracy(val_accuracies)

# Compute and plot the confusion matrix for the validation set
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model.predict(inputs)
        all_preds.extend(outputs)
        all_labels.extend(labels)
cm = confusion_matrix(all_labels, all_preds)
plot_confusion_matrix(cm)

print('Finished Training')
torch.save(model.state_dict(), 'model.pth')


