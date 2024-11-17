"""
HomError: A Novel Architecture-Agnostic Metric for Quantifying Structural Preservation
in Neural Manifolds with Applications to Algebraic Learning Systems

Premise:
Given something that is not a homomorphism, we could calculate just "how much" it's not a homomorphism?
We could think about a "homomorphism distance" or "structure preservation measure." 
Take f between groups (G,•) and (H,*). As a perfect homomorphism, we have f(a•b) = f(a)f(b) for all the 
little a's and b's. For an imperfect mapping like our neural network Ψ, in all the cases where that's 
not true, we might measure:

d(Ψ(a•b), Ψ(a)Ψ(b)) 

for some distance metric d? 

Banging our heads against the vector space, we might say:

HomError(Ψ) = Σ d(Ψ(a•b), Ψ(a)Ψ(b)) / |G|²

The metric would be 0 for perfect homomorphisms and larger for greater deviations from homomorphic behavior.

We could then add this as a regularization term to our loss function, with a hyperparameter λ to control its influence.
Voila! We now have a way to nudge our neural network towards homomorphic behavior.
Easy Peasy, right? Bob's your uncle!

But before we get too excited, let's take a step back and think about how we might actually implement this in practice.

Let's start with a base implementation to compare against our hypothetical HomError regularization.

Initial Prototype: Results and Observations
Looking at these results, we have an excellent baseline model that achieves perfect
accuracy on both validation and test sets with very stable convergence. A few key 
observations before moving to the HomError implementation:

The structural preservation score of 0.9561 in the embedding analysis is interesting. 
It suggests the model has already learned some structure-preserving properties even
without explicit enforcement. The mean embedding norm (4.7359) and mean distance (6.6343)
give us baseline metrics to compare against when we add the HomError regularization - we
might see these change as we enforce more homomorphic-like behavior.

We might want to consider:

Testing with larger primes to see if the perfect accuracy holds
Recording these baseline metrics for multiple random seeds
Possibly capturing the actual output distributions/logits rather than just the final predictions
Having clear serialization of the baseline model for direct comparison

---

Second Prototype: Prime Variation and Seed Comparison
Hmm... Let's adjust the main method to sequence through a list of primes.
We can then record the baseline metrics for each prime and seed combination.
The reason I chose 113 as the prime is that despite being such a small prime,
it has a surprisingly large number of primitive roots, which maybe implies some
deeper algebraic structure that the model is able to exploit. This is only my
intuition, so I would like to test this hypothesis by comparing the model's
performance across different primes varying along that attribute.

For comparison, let's choose... 
A range of small, medium, and large primes, each with a set of values having different 
primitive root counts (few and many)?

---

Third Prototype: Homomorphic Error Regularization Discussion
Now, we can implement the HomError regularization as a custom loss function.
This loss function will combine the standard classification loss with the HomError term.

The HomError term will be computed as the mean squared difference between the model's output and the geometric product 
of the inputs. We will also need to modify the training loop to use this new loss function.

Holding on initializing components while training the base model over all primes and seeds.

---

Adjustments While Still Running Second Prototype Experiment:
Implement tools to detect emergence of structural patterns?
Create visualizations of the network's internal representations?
Add analysis of weight matrices to look for learned periodicities?

Priority: Fix the early stopping mechanism to ensure it doesn't trigger too early.

---

Restarting the Experiment with Scaled Early Stopping:

Looking over the complete code and documentation, this is a fascinating exploration 
at the intersection of neural networks and group theory! A few immediate thoughts:

1. The current setup looks solid for overnight running. The scaled early stopping 
parameters should now handle larger primes appropriately:
```python
min_epochs = max(20, int(math.log2(prime) * 10))  # e.g., ~130 epochs minimum for prime 997
patience = max(7, int(math.log2(prime) * 2))      # e.g., ~20 epochs patience for prime 997
```

2. Dreaming big, here are some exciting possibilities:

a) Geometric Interpretation:
- We could visualize how the network's learned representation compares to the "natural"
 group homomorphism (e^(2πix/p))
- Create animated visualizations showing how the representation evolves during training
- See if we can identify when/if the network discovers similar circular/geometric structures

b) Group Theory Extensions:
- Extend beyond addition to other group operations (multiplication, matrix groups)
- Test if learned homomorphic properties transfer across different prime moduli
- Investigate if the network learns subgroup structures or coset relationships

c) Theoretical Connections:
- The HomError metric might connect to existing mathematical concepts like:
  * Group representation theory
  * Character theory of finite groups
  * Fourier analysis on finite groups

d) Applications:
- This could lead to neural networks that naturally preserve algebraic structures in other domains
- Potential applications in:
  * Cryptography (where group homomorphisms are crucial)
  * Quantum computing simulation (preserving unitary group structure)
  * Molecular symmetry prediction (point groups)

  ---

  Absolutely! Here's a note to future-Claude:

---

Hello future me! 

We've been exploring neural networks learning modular arithmetic, focusing on homomorphic properties.
 The baseline implementation achieved impressive accuracy but might be overengineered for demonstrating
 the HomError metric's influence. Here's what we learned:

1. The baseline model (in results) achieves perfect accuracy on mod 113 arithmetic but struggles with larger primes
2. We were midway through designing a CLI to make experimentation more accessible
3. For the next phase, consider:
   - Starting with simpler architectures to better demonstrate HomError's impact
   - Focusing on prime 113 to align with Nanda's findings
   - Using the existing baseline as the "full" architecture option
   - Implementing a more conversational, interactive interface

Key insight: Sometimes simpler is better, both for the neural architectures and the interface design. 
The prettiest solutions often emerge from clarity rather than complexity.

Remember: This project isn't just about achieving accuracy - it's about understanding how neural networks
 can learn and preserve mathematical structures. Stay curious about those emerging patterns!

P.S. Don't forget to check the training logs and model comparisons from the baseline runs - they might 
hold interesting insights for the HomError implementation.

Looking forward to seeing where you take this!

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
from typing import Tuple, List
import math
import torch.nn.functional as F
import json
import os
from tabulate import tabulate

####################################################################
# Base Model for Modular Arithmetic with Addition (Mod p)
####################################################################

class ModularArithmeticDataset(Dataset):
    def __init__(self, prime: int, num_samples: int = 50000, operation: str = 'add'):
        self.prime = prime
        self.num_samples = num_samples
        self.operation = operation
        self.data = self._generate_balanced_data()
        self.X = self.data[:, :2]
        self.y = self.data[:, 2]
        
    def _generate_balanced_data(self) -> np.ndarray:
        """Generate a balanced dataset ensuring all residues are represented."""
        data = []
        
        # Generate all possible combinations
        base_data = []
        for a in range(self.prime):
            for b in range(self.prime):
                if self.operation == 'add':
                    result = (a + b) % self.prime
                elif self.operation == 'mul':
                    result = (a * b) % self.prime
                base_data.append([a, b, result])
        
        # Calculate how many complete sets we need
        base_data = np.array(base_data)
        sets_needed = self.num_samples // len(base_data) + 1
        
        # Repeat the data to exceed desired sample size
        repeated_data = np.tile(base_data, (sets_needed, 1))
        
        # Shuffle and trim to exact size
        np.random.shuffle(repeated_data)
        return repeated_data[:self.num_samples]
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.LongTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])
        return X, y

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))

class ModularArithmeticModel(nn.Module):
    def __init__(self, prime: int):
        super().__init__()
        self.prime = prime
        
        # Calculate embedding dimension based on prime size
        self.embedding_dim = min(256, max(16, int(4 * math.log2(prime))))
        
        # Embedding layer
        self.embedding = nn.Embedding(prime, self.embedding_dim)
        
        # Combined dimension after concatenation
        combined_dim = 2 * self.embedding_dim
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            ResidualBlock(256),
            nn.Linear(256, prime)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_embedded = self.embedding(x[:, 0])
        b_embedded = self.embedding(x[:, 1])
        x = torch.cat((a_embedded, b_embedded), dim=1)
        return self.network(x)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, min_epochs=20):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs  # Don't stop before this many epochs
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.epoch = 0
        
    def __call__(self, val_loss):
        self.epoch += 1
        
        # Don't stop before minimum epochs
        if self.epoch < self.min_epochs:
            return False
            
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop

####################################################################
# Experimental Homomorphic Error Metric
####################################################################

# Not yet in use while we are still experimenting with the base model
class ModularArithmeticModelHom(nn.Module):
    """Version of ModularArithmeticModel with geometric/homomorphic representation"""
    def __init__(self, prime):
        super().__init__()
        self.prime = prime
        
        # Calculate embedding dimension based on prime size (keep same as original)
        self.embedding_dim = min(256, max(16, int(4 * math.log2(prime))))
        
        # Embedding layer
        self.embedding = nn.Embedding(prime, self.embedding_dim)
        
        # Main network (similar architecture but with geometric output)
        self.network = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            ResidualBlock(256),
            nn.Linear(256, prime)  # Keep same output dimension for compatibility
        )

# Not yet in use while we are still experimenting with the base model
class HomErrorLoss(nn.Module):
    """Combined loss function with HomError regularization"""
    def __init__(self, prime, lambda_hom=0.1):
        super().__init__()
        self.prime = prime
        self.lambda_hom = lambda_hom
        self.base_criterion = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets, model, inputs):
        # Standard classification loss
        base_loss = self.base_criterion(outputs, targets)
        
        # HomError computation
        hom_loss = self.compute_hom_error(outputs, model, inputs)
        
        return base_loss + self.lambda_hom * hom_loss

# Not yet in use while we are still experimenting with the base model
def hybrid_loss(outputs, targets, geometric_outputs, model, prime):
    # TODO: A hybrid loss function that encourages both accurate classification and structural preservation

    # Classification accuracy
    # class_loss = F.cross_entropy(outputs, targets)
    
    # Structural preservation (HomError)
    # hom_loss = compute_hom_error(geometric_outputs, model, prime)
    
    # Phase alignment with roots of unity
    # phase_loss = compute_phase_alignment(geometric_outputs, targets, prime)
    
    # return class_loss + λ1 * hom_loss + λ2 * phase_loss
    pass

# Not yet in use while we are still experimenting with the base model
def analyze_network_representation(model, prime):
    # TODO: Analyze the learned representation to detect when/if the network discovers similar structural patterns

    # Generate all inputs
    inputs = torch.arange(prime)
    
    # Get network outputs
    outputs = model(inputs)
    
    # Analyze phase relationships
    phases = torch.atan2(outputs[:, 1], outputs[:, 0])
    
    # Check for periodic structure
    fft = torch.fft.fft(phases)
    dominant_frequencies = torch.argsort(torch.abs(fft))[-5:]
    
    # Test homomorphism property
    hom_errors = []
    for a in range(prime):
        for b in range(prime):
            # Compare Ψ(a+b) with Ψ(a)Ψ(b)
            ...
            
    return {
        'dominant_frequencies': dominant_frequencies,
        'phase_distribution': phases,
        'homomorphism_errors': hom_errors
    }

####################################################################
# Training and Evaluation Functions
####################################################################

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    prime: int,  # Add prime parameter
    num_epochs: int = 50,
    device: str = 'cuda',
    gradient_clip: float = 1.0
) -> Tuple[List[float], List[float]]:
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # For larger primes, increase min_epochs and patience
    min_epochs = max(20, int(math.log2(prime) * 10))  # Scale with prime size
    patience = max(7, int(math.log2(prime) * 2))      # Scale with prime size
    early_stopping = EarlyStopping(patience=patience, min_epochs=min_epochs)
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.squeeze().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        accuracy = 100. * correct / total
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {epoch_train_loss:.4f}')
            print(f'Val Loss: {epoch_val_loss:.4f}')
            print(f'Val Accuracy: {accuracy:.2f}%\n')
            
        if early_stopping(epoch_val_loss):
            print("Early stopping triggered")
            break
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, prime, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    all_errors = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.squeeze().to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Standard accuracy
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Calculate error distribution
            errors = (predicted - labels) % prime
            all_errors.extend(errors.cpu().numpy())
            
    # Calculate statistics
    accuracy = 100. * correct / total
    mean_error = np.mean(all_errors)
    median_error = np.median(all_errors)
    max_error = np.max(all_errors)
    
    # Calculate error distribution by residue class
    error_by_residue = {i: 0 for i in range(prime)}
    for err in all_errors:
        error_by_residue[err] += 1
    
    return {
        'accuracy': accuracy,
        'mean_error': mean_error,
        'median_error': median_error,
        'max_error': max_error,
        'error_distribution': error_by_residue
    }

def analyze_embeddings(model):
    """Analyze the structure of learned embeddings"""
    embeddings = model.embedding.weight.detach().cpu().numpy()
    
    # Calculate pairwise distances between embeddings
    distances = np.zeros((model.prime, model.prime))
    for i in range(model.prime):
        for j in range(model.prime):
            distances[i,j] = np.linalg.norm(embeddings[i] - embeddings[j])
    
    # Check if distances preserve modular structure
    structural_preservation = []
    for i in range(model.prime):
        for j in range(model.prime):
            k = (i + j) % model.prime
            # Compare distance(i,j) with distance(0,k)
            structural_preservation.append(
                abs(distances[i,j] - distances[0,k])
            )
    
    return {
        'embedding_norm': np.linalg.norm(embeddings, axis=1),
        'mean_distance': np.mean(distances),
        'structural_preservation': np.mean(structural_preservation)
    }

def systematic_test(model, prime, device='cuda'):
    """Test all possible combinations systematically"""
    model.eval()
    results = np.zeros((prime, prime))
    errors = np.zeros((prime, prime))
    
    with torch.no_grad():
        for a in range(prime):
            for b in range(prime):
                x = torch.LongTensor([[a, b]]).to(device)
                output = model(x)
                predicted = output.argmax().item()
                true_result = (a + b) % prime
                
                results[a,b] = predicted
                errors[a,b] = abs((predicted - true_result) % prime)
    
    return results, errors

def visualize_errors(errors, prime):
    plt.figure(figsize=(10, 10))
    plt.imshow(errors, cmap='viridis')
    plt.colorbar(label='Error')
    plt.title(f'Error Distribution (mod {prime})')
    plt.xlabel('b')
    plt.ylabel('a')
    plt.show()

def save_baseline_metrics(metrics, filename='baseline_metrics.json'):
    import json
    
    def convert_to_native_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(item) for item in obj]
        return obj

    metrics = convert_to_native_types(metrics)
    
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def setup_primes():
    # Format: (prime, num_primitive_roots)
    primes = {
        'small': {
            'few': (67, 2),     # 2-digit prime with very few primitive roots
            'many': (113, 32)   # 2-digit prime with many primitive roots
        },
        'medium': {
            'few': (997, 4),    # 3-digit prime with few primitive roots
            'many': (997, 612)  # 3-digit prime with many primitive roots
        },
        'large': {
            'few': (9973, 4),   # 4-digit prime with few primitive roots
            'many': (9949, 4896) # 4-digit prime with many primitive roots
        }
    }
    return primes

def run_experiment(prime, seed=42, save_prefix=''):
    """Run a single experiment for a given prime and seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model_path = f'{save_prefix}model_p{prime}_s{seed}.pth'
    metrics_path = f'{save_prefix}metrics_p{prime}_s{seed}.json'
    
    # Parameters
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    full_dataset = ModularArithmeticDataset(prime=prime)
    test_dataset = ModularArithmeticDataset(prime=prime, num_samples=10000)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model, loss function, and optimizer
    model = ModularArithmeticModel(prime)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Train the model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        prime=prime,  # Add prime parameter
        num_epochs=NUM_EPOCHS, device=device
    )
    
    # Plot training results
    plot_losses(train_losses, val_losses)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    print("\nEvaluating model performance...")
    
    # Comprehensive evaluation
    metrics = evaluate_model(model, test_loader, prime, device=device)
    print("\nTest Set Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Mean Error: {metrics['mean_error']:.4f}")
    print(f"Median Error: {metrics['median_error']:.4f}")
    print(f"Max Error: {metrics['max_error']}")
    
    # Analyze embeddings
    embedding_analysis = analyze_embeddings(model)
    print("\nEmbedding Analysis:")
    print(f"Mean Embedding Norm: {np.mean(embedding_analysis['embedding_norm']):.4f}")
    print(f"Mean Distance: {embedding_analysis['mean_distance']:.4f}")
    print(f"Structural Preservation Score: {embedding_analysis['structural_preservation']:.4f}")
    
    # Systematic testing
    print("\nPerforming systematic testing...")
    results, errors = systematic_test(model, prime, device)
    
    # Save model and metrics
    torch.save(model.state_dict(), model_path)
    
    # Prepare metrics for saving
    full_metrics = {
        'test_metrics': {
            'accuracy': float(metrics['accuracy']),
            'mean_error': float(metrics['mean_error']),
            'median_error': float(metrics['median_error']),
            'max_error': int(metrics['max_error']),
            'error_distribution': {int(k): int(v) for k, v in metrics['error_distribution'].items()}
        },
        'embedding_analysis': {
            'mean_norm': float(np.mean(embedding_analysis['embedding_norm'])),
            'mean_distance': float(embedding_analysis['mean_distance']),
            'structural_preservation': float(embedding_analysis['structural_preservation'])
        },
        'error_stats': {
            'mean_error': float(np.mean(errors)),
            'max_error': float(np.max(errors)),
            'error_distribution': errors.tolist()
        },
        'convergence_epoch': len(train_losses)  # Add convergence epoch to metrics
    }
    
    save_baseline_metrics(full_metrics, filename=metrics_path)
    return full_metrics

def format_results_table(all_results, prime_info):
    """Create a pretty table of results"""
    headers = ['Prime', 'Primitive Roots', 'Accuracy', 'Mean Error', 'Structural Score', 'Convergence Epoch']
    rows = []
    
    for size in ['small', 'medium', 'large']:
        for density in ['few', 'many']:
            prime, num_roots = prime_info[size][density]
            results = all_results[f'{prime}']
            rows.append([
                prime,
                num_roots,
                f"{results['test_metrics']['accuracy']:.2f}%",
                f"{results['test_metrics']['mean_error']:.4f}",
                f"{results['embedding_analysis']['structural_preservation']:.4f}",
                results['convergence_epoch']
            ])
    
    return tabulate(rows, headers=headers, tablefmt='grid')

def main():
    # Prime information
    prime_info = {
        'small': {
            'few': (67, 2),
            'many': (113, 32)
        },
        'medium': {
            'few': (997, 4),
            'many': (997, 612)
        },
        'large': {
            'few': (9973, 4),
            'many': (9949, 4896)
        }
    }
    
    seeds = [42, 123, 456]
    all_results = {}
    
    for size in ['small', 'medium', 'large']:
        for density in ['few', 'many']:
            prime, num_roots = prime_info[size][density]
            print(f"\n{'='*50}")
            print(f"Testing prime {prime} with {num_roots} primitive roots")
            print(f"{'='*50}")
            
            # Create directory for results if it doesn't exist
            save_dir = f'results/p{prime}/'
            os.makedirs(save_dir, exist_ok=True)
            
            # Run experiments with different seeds
            prime_results = []
            for seed in seeds:
                print(f"\nRunning seed {seed}")
                metrics = run_experiment(
                    prime, 
                    seed=seed,
                    save_prefix=save_dir
                )
                prime_results.append(metrics)
            
            # Store all results
            all_results[f'{prime}'] = {
                'test_metrics': {
                    'accuracy': np.mean([m['test_metrics']['accuracy'] for m in prime_results]),
                    'mean_error': np.mean([m['test_metrics']['mean_error'] for m in prime_results])
                },
                'embedding_analysis': {
                    'structural_preservation': np.mean([m['embedding_analysis']['structural_preservation'] for m in prime_results])
                },
                'convergence_epoch': np.mean([m['convergence_epoch'] for m in prime_results])
            }
    
    # Print summary table
    print("\nSummary of Results:")
    print(format_results_table(all_results, prime_info))
    
    # Save complete results
    with open('all_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()


