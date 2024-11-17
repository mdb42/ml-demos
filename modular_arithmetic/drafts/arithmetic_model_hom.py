"""
-----------------------------------------------------------------------------------------
Modular Arithmetic Neural Network with Homomorphism Error (HomError) Loss
-----------------------------------------------------------------------------------------

Purpose:
This implementation explores whether neural networks can be guided to learn and preserve
algebraic structures by incorporating a "homomorphism error" term in the loss function.
We aim to measure and minimize how much the neural network deviates from being a homomorphism
between groups, specifically in the context of modular arithmetic.

Premise:
- Given a mapping between groups (G, •) and (H, *), a perfect homomorphism satisfies:
  f(a • b) = f(a) * f(b) for all a, b in G.
- For an imperfect mapping like our neural network Ψ, we can quantify the deviation
  from homomorphic behavior using a distance metric:
  HomError(Ψ) = Σ d(Ψ(a • b), Ψ(a) * Ψ(b)) / |G|²
- By integrating the HomError metric into the training loss function, neural networks
  can be guided to learn functions that better preserve underlying algebraic structures.

Key Components:
1. Data Representation:
   - Input: Numbers mod n (normalized to [-1, 1])
   - Output: Complex numbers on the unit circle (e^(2πix/n)), represented as 2D real vectors
     (cos(2πx/n), sin(2πx/n))

2. Network Architecture:
   - Feed-forward neural network with multiple hidden layers
   - Tanh activation functions for smooth gradients and outputs in [-1, 1]
   - Output normalized to lie on the unit circle to match target representation

3. Loss Function Components:
   - Primary Loss (L_primary): Measures the difference between the network's output and
     the target roots of unity using cosine similarity loss
   - Homomorphism Error (L_hom): Measures the deviation from homomorphic behavior by comparing
     Ψ(a + b mod n) and Ψ(a) * Ψ(b)
   - Combined Loss: L_total = L_primary + λ_hom * L_hom

Proposed Improvements:
1. Loss Tracking:
   - Separate logging for primary loss and HomError components
   - Track running averages to monitor training progress

2. Visualization:
   - Plot outputs on the unit circle to visualize how well the network learns the mapping
   - Plot loss curves over time for both primary loss and HomError
   - Visualize phase alignment and error distributions

3. Performance Enhancements:
   - Implement learning rate scheduling to adjust the learning rate during training
   - Introduce early stopping based on validation loss to prevent overfitting
   - Experiment with batch size adjustments to improve convergence
   - Tune λ_hom to find the optimal balance between primary loss and HomError

4. Validation and Testing:
   - Test the model with different moduli n to assess generalization
   - Compare performance with and without the HomError term to evaluate its impact
   - Measure generalization to unseen data or operations

Date: November 4, 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # For visualization

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set the modulus n
n = 113  # You can choose any prime number or positive integer

# Define the dataset: integers from 0 to n-1
G = np.arange(n)

# Convert integers to PyTorch tensors
inputs = torch.tensor(G, dtype=torch.float32).unsqueeze(1)
# Scale inputs to range [-1, 1]
inputs = (inputs - n / 2) / (n / 2)

# Define the target outputs: n-th roots of unity
angles = (2 * np.pi * G) / n
targets = torch.tensor(
    np.column_stack((np.cos(angles), np.sin(angles))), dtype=torch.float32
)

# Neural Network Model
class ModularAdditionNet(nn.Module):
    def __init__(self):
        super(ModularAdditionNet, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)  # Activation function
        x = self.fc2(x)
        x = torch.tanh(x)  # Activation function
        x = self.fc3(x)
        x = torch.tanh(x)  # Activation function
        x = self.fc4(x)
        # Normalize output to unit circle
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x

# Initialize the model, optimizer, and hyperparameters
model = ModularAdditionNet()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Hyperparameters
epochs = 2000
batch_size = 32
lambda_hom = 0.5  # Weight for the HomError term

# For loss tracking
losses_primary = []
losses_hom = []
losses_total = []

# Training loop
for epoch in range(epochs):
    permutation = torch.randperm(inputs.size()[0])
    total_loss = 0.0
    total_loss_primary = 0.0
    total_loss_hom = 0.0

    for i in range(0, inputs.size()[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[i : i + batch_size]
        batch_inputs = inputs[indices]
        batch_targets = targets[indices]

        # Forward pass
        outputs = model(batch_inputs)

        # Compute the primary loss (cosine similarity loss)
        cos_sim = torch.sum(outputs * batch_targets, dim=1)
        loss_primary = 1 - torch.mean(cos_sim)

        # Compute the HomError loss
        # Sample m pairs (a_i, b_i)
        m = batch_size
        idx_a = torch.randint(0, n, (m,))
        idx_b = torch.randint(0, n, (m,))
        a = inputs[idx_a]
        b = inputs[idx_b]
        # Compute (a + b) mod n, scaled to [-1, 1]
        sum_ab = ((a + b) * (n / 2) + n / 2) % n
        a_plus_b_mod_n = (sum_ab - n / 2) / (n / 2)

        # Compute Ψ(a), Ψ(b), and Ψ(a + b mod n)
        Psi_a = model(a)
        Psi_b = model(b)
        Psi_a_plus_b = model(a_plus_b_mod_n)

        # Compute Ψ(a) * Ψ(b) using complex multiplication
        real_part = Psi_a[:, 0] * Psi_b[:, 0] - Psi_a[:, 1] * Psi_b[:, 1]
        imag_part = Psi_a[:, 0] * Psi_b[:, 1] + Psi_a[:, 1] * Psi_b[:, 0]
        Psi_a_mul_b = torch.stack((real_part, imag_part), dim=1)
        # Normalize to unit circle
        Psi_a_mul_b = Psi_a_mul_b / torch.norm(Psi_a_mul_b, dim=1, keepdim=True)

        # Compute the HomError loss (cosine similarity between Ψ(a + b) and Ψ(a) * Ψ(b))
        cos_sim_hom = torch.sum(Psi_a_plus_b * Psi_a_mul_b, dim=1)
        loss_hom = 1 - torch.mean(cos_sim_hom)

        # Total loss
        loss = loss_primary + lambda_hom * loss_hom

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_primary += loss_primary.item()
        total_loss_hom += loss_hom.item()

    # Compute average losses for the epoch
    avg_loss = total_loss / (inputs.size()[0] / batch_size)
    avg_loss_primary = total_loss_primary / (inputs.size()[0] / batch_size)
    avg_loss_hom = total_loss_hom / (inputs.size()[0] / batch_size)

    # Store losses for plotting
    losses_total.append(avg_loss)
    losses_primary.append(avg_loss_primary)
    losses_hom.append(avg_loss_hom)

    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Total Loss: {avg_loss:.6f}, "
            f"Primary Loss: {avg_loss_primary:.6f}, "
            f"HomError Loss: {avg_loss_hom:.6f}"
        )

# Test the model on the entire dataset
with torch.no_grad():
    test_inputs = torch.tensor(np.arange(n), dtype=torch.float32).unsqueeze(1)
    test_inputs = (test_inputs - n / 2) / (n / 2)
    test_outputs = model(test_inputs)
    test_angles = torch.atan2(test_outputs[:, 1], test_outputs[:, 0])

    # Compare the predicted angles to the true angles
    true_angles = torch.tensor(angles, dtype=torch.float32)
    angle_diff = torch.abs(test_angles - true_angles)
    # Adjust angle differences to be between 0 and π
    angle_diff = torch.remainder(angle_diff, 2 * np.pi)
    angle_diff = torch.minimum(angle_diff, 2 * np.pi - angle_diff)
    avg_angle_error = torch.mean(angle_diff).item()
    print(f"Average angle error: {avg_angle_error:.6f} radians")

# TODO: Visualization
# 1. Plot the losses over epochs
# 2. Plot the predicted outputs on the unit circle
# 3. Visualize phase alignment and error distributions

def plot_unit_circle_predictions(model, n, title="Predictions on Unit Circle"):
    """Plot model predictions and true n-th roots of unity on the unit circle."""
    with torch.no_grad():
        # Get model predictions
        test_inputs = torch.tensor(np.arange(n), dtype=torch.float32).unsqueeze(1)
        test_inputs = (test_inputs - n / 2) / (n / 2)
        predictions = model(test_inputs).numpy()
        
        # True n-th roots of unity
        angles = (2 * np.pi * np.arange(n)) / n
        true_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Create plot
        plt.figure(figsize=(8, 8))
        # Plot unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
        plt.gca().add_artist(circle)
        # Plot predictions and true points
        plt.scatter(predictions[:, 0], predictions[:, 1], c='red', label='Predictions', alpha=0.5)
        plt.scatter(true_points[:, 0], true_points[:, 1], c='blue', label='True Values', alpha=0.5)
        plt.axis('equal')
        plt.grid(True)
        plt.title(title)
        plt.legend()
        plt.show()

def plot_error_distribution(angle_diff, title="Distribution of Angle Errors"):
    """Plot histogram of angle errors."""
    plt.figure(figsize=(8, 5))
    plt.hist(angle_diff.numpy(), bins=30, density=True)
    plt.xlabel('Angle Error (radians)')
    plt.ylabel('Density')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_training_curves(losses_dict, title="Training Curves"):
    """Plot all loss components over time."""
    plt.figure(figsize=(10, 5))
    for name, values in losses_dict.items():
        plt.plot(values, label=name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# TODO: Implement learning rate scheduling
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
# After model initialization, add scheduler initialization:
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=200,
    verbose=True,
    min_lr=1e-6
)

# In the training loop, after computing average losses:
scheduler.step(avg_loss)  # Update learning rate based on loss
if (epoch + 1) % 100 == 0:
    current_lr = optimizer.param_groups[0]['lr']
    print(
        f"Epoch [{epoch+1}/{epochs}], "
        f"Total Loss: {avg_loss:.6f}, "
        f"Primary Loss: {avg_loss_primary:.6f}, "
        f"HomError Loss: {avg_loss_hom:.6f}, "
        f"Learning Rate: {current_lr:.6f}"
    )

# TODO: Introduce early stopping based on validation loss

# TODO: Experiment with different moduli n and compare performance

# TODO: Compare with a model trained without the HomError term
# - Train a separate model with lambda_hom = 0
# - Compare angle errors and loss curves

# TODO: Save the model and results for further analysis

# TODO: Write functions to modularize code for reusability and clarity
