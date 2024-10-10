import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.tanh(self.fc1(x))
        # x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train model and collect stats
def train(model, train_loader, optimizer, criterion, num_epochs):
    loss_history = []
    grad_history = []
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_gradients = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Calculate gradient magnitude
            total_gradients += sum((param.grad.norm(2).item() ** 2) for param in model.parameters())
            
            optimizer.step()

            running_loss += loss.item()

        loss_history.append(running_loss / len(train_loader))
        grad_history.append(total_gradients)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    return loss_history, grad_history

# Function to plot training loss and gradient norms
def plot_curves(normalization_loss, no_normalization_loss, normalization_grads, no_normalization_grads):
    epochs = range(1, len(normalization_loss) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, normalization_loss, label='With Normalization', color='b')
    plt.plot(epochs, no_normalization_loss, label='Without Normalization', color='r')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot gradient norms
    plt.subplot(1, 2, 2)
    plt.plot(epochs, normalization_grads, label='With Normalization', color='b')
    plt.plot(epochs, no_normalization_grads, label='Without Normalization', color='r')
    plt.title('Gradient Magnitudes')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Magnitude')
    plt.legend()

    plt.tight_layout()
    plt.savefig("compare_tanh_activation.png")
    # plt.show()

# Prepare MNIST dataset with and without normalization
transform_normalized = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
transform_no_normalization = transforms.Compose([transforms.ToTensor()])

train_dataset_normalized = datasets.MNIST(root='./data', train=True, transform=transform_normalized, download=True)
train_dataset_no_normalization = datasets.MNIST(root='./data', train=True, transform=transform_no_normalization, download=True)

train_loader_normalized = DataLoader(train_dataset_normalized, batch_size=64, shuffle=True)
train_loader_no_normalization = DataLoader(train_dataset_no_normalization, batch_size=64, shuffle=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models, loss function and optimizer
model_normalized = SimpleNN().to(device)
model_no_normalization = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer_normalized = optim.SGD(model_normalized.parameters(), lr=0.01)
optimizer_no_normalization = optim.SGD(model_no_normalization.parameters(), lr=0.01)

# Train models
num_epochs = 10
print("Training with normalization...")
normalized_loss, normalized_gradients = train(model_normalized, train_loader_normalized, optimizer_normalized, criterion, num_epochs)

print("\nTraining without normalization...")
no_normalization_loss, no_normalization_gradients = train(model_no_normalization, train_loader_no_normalization, optimizer_no_normalization, criterion, num_epochs)

# Plot the results
plot_curves(normalized_loss, no_normalization_loss, normalized_gradients, no_normalization_gradients)
