import torch
import torch.nn as nn
from model import PrunableLinear
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



# Sparsity Loss

def sparsity_loss(model):
    loss = torch.tensor(0.0)

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            loss = loss + torch.sum(gates)

    return loss



# Model

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(32 * 32 * 3, 256)
        self.fc2 = PrunableLinear(256, 128)
        self.fc3 = PrunableLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Dataset

transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64)



# Training Setup

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

lambda_val = 0.001   



# Training Loop

epochs = 10

for epoch in range(epochs):
    total_loss = 0.0

    for images, labels in trainloader:
        outputs = model(images)

        ce_loss = criterion(outputs, labels)
        sp_loss = sparsity_loss(model)

        loss = ce_loss + lambda_val * sp_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")



# Evaluation: Accuracy

def evaluate(model, testloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total



# Evaluation: Sparsity

def calculate_sparsity(model, threshold=1e-2):
    total = 0
    zero = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)

            total += gates.numel()
            zero += (gates < threshold).sum().item()

    return 100 * zero / total



# Final Results

accuracy = evaluate(model, testloader)
sparsity = calculate_sparsity(model)

print("Final Test Accuracy:", accuracy)
print("Sparsity Level:", sparsity, "%")