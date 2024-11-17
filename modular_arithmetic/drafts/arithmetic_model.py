import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Step 1: Data Generation for Mod p
def generate_mod_prime_addition_data(prime, size=1000):
    x = np.random.randint(0, prime, size)
    y = np.random.randint(0, prime, size)
    z = (x + y) % prime
    return x, y, z

# Step 2: Enhanced Feed-Forward Model with More Layers
class ModPrimeNetwork(nn.Module):
    def __init__(self):
        super(ModPrimeNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 100)  # Increased hidden layer size to 100 neurons
        self.fc2 = nn.Linear(100, 100)  # Second hidden layer with 100 neurons
        self.fc3 = nn.Linear(100, 1)   # Output layer
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # Activation for second hidden layer
        return self.fc3(x)

# Step 3: Prepare Inputs and Targets (with Normalization)
def prepare_data(x, y, z, prime):
    inputs = np.stack([x, y], axis=1) / prime  # Normalize inputs by dividing by prime
    targets = z / prime  # Normalize targets by dividing by prime
    return inputs, targets

# Step 4: Training the Model with Learning Rate Decay
def train_model(model, optimizer, inputs, targets, prime, epochs=1000, log_file=None):
    criterion = nn.MSELoss()
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    # Scheduler for learning rate decay
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        scheduler.step()  # Adjust the learning rate based on the schedule

        # Log loss every 100 epochs
        if epoch % 100 == 0 or epoch == epochs - 1:
            log_entry = f'Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}'
            print(log_entry)
            if log_file:
                log_file.write(log_entry + '\n')

# Step 5: Testing the Model with Accuracy Calculation
def test_model(model, prime, log_file=None):
    x_test, y_test, z_test = generate_mod_prime_addition_data(prime, size=100)
    inputs_test, _ = prepare_data(x_test, y_test, z_test, prime)
    inputs_test = torch.tensor(inputs_test, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(inputs_test)
    
    predictions = (outputs.numpy() * prime) % prime  # Denormalize and apply mod p
    predictions = predictions.round().astype(int)  # Round to nearest integer
    
    # Apply modulus wrapping to handle predictions outside the range
    predictions = predictions % prime

    correct = np.sum(predictions.flatten() == z_test)
    accuracy = (correct / len(z_test)) * 100
    accuracy_log = f'\nAccuracy: {accuracy:.2f}% ({correct}/{len(z_test)})\n'
    print(accuracy_log)
    
    if log_file:
        log_file.write(accuracy_log)
    
    # Log detailed test results
    if log_file:
        log_file.write("\nTest Results:\n")
        for i in range(len(x_test)):
            test_result = f"Input: ({x_test[i]}, {y_test[i]}) | Target: {z_test[i]} | Predicted: {predictions[i][0]}"
            print(test_result)
            log_file.write(test_result + '\n')

# Step 6: Save Model and Logs with Timestamped Filenames
def save_model_and_logs(model, prime, epochs):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Model filename
    model_filename = f"models/mod_prime_model_p{prime}_{epochs}epochs_{timestamp}.pt"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")

    # Log filename
    log_filename = f"logs/mod_prime_log_p{prime}_{epochs}epochs_{timestamp}.txt"
    log_file = open(log_filename, "w")
    log_file.write(f"Prime: {prime}, Epochs: {epochs}\n\n")
    print(f"Logs saved as {log_filename}")
    
    return log_file, log_filename

# Main Execution
if __name__ == "__main__":
    prime = 113  # Modulus prime for the experiment
    epochs = 8000  # Number of epochs to train
    data_size = 100000  # Training dataset size

    # Generate training data
    x_train, y_train, z_train = generate_mod_prime_addition_data(prime, size=data_size)
    inputs, targets = prepare_data(x_train, y_train, z_train, prime)

    # Define model and optimizer
    model = ModPrimeNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Start with a moderate learning rate

    # Save model and logs
    log_file, log_filename = save_model_and_logs(model, prime, epochs)

    # Train the model
    train_model(model, optimizer, inputs, targets, prime, epochs, log_file)

    # Test the model on new data
    test_model(model, prime, log_file)

    # Close the log file
    log_file.close()
