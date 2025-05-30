"""
Name: David Rivard
Topic: Computer Vision REU Assignment
Due Date: 05/30/2025
References:
    * https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    For ResNets:
    * https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch
    For data augmentation:
    * https://docs.pytorch.org/vision/stable/transforms.html
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define normaliztaion & data augmentation for training
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
])

# Define normalization for test data
test_transforms = transforms.Compose([transforms.ToTensor()])

# Load CIFAR-100 dataset (through PyTorch)
training_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transforms)
testing_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transforms)

training_loader = DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=2)
testing_loader = DataLoader(testing_dataset, batch_size=64, shuffle=False, num_workers=2)

# Define residual block used to prevent vanishing gradient
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels)
        )

    def forward(self, x):
        return F.relu(self.layers(x) + x)

# ResNet-based model for the CIFAR-100 dataset
class ResNetModel(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNetModel, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block_group_1 = self._build_residual_group(64, 2)
        self.block_group_2 = self._build_residual_group(128, 2, downsample=True)
        self.block_group_3 = self._build_residual_group(256, 2, downsample=True)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _build_residual_group(self, out_channels, num_blocks, downsample=False):
        layers = []
        in_channels = 64 if out_channels == 64 else out_channels // 2
        if downsample:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        else:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.block_group_1(x)
        x = self.block_group_2(x)
        x = self.block_group_3(x)
        x = self.global_pooling(x)
        x = self.classifier(x)
        return x

# Instantiate ResNet model and use GPU
model = ResNetModel().to(device)

# Using Adam optimizer with label smoothing loss performed best
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# Define training function
def train_model_one_epoch(model, data_loader):
    model.train()
    total_correct, total_samples, total_loss = 0, 0, 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted_labels = predictions.max(1)
        total_samples += labels.size(0)
        total_correct += predicted_labels.eq(labels).sum().item()
    return total_loss / len(data_loader), total_correct / total_samples

# Define evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            _, predicted_labels = predictions.max(1)
            total_samples += labels.size(0)
            total_correct += predicted_labels.eq(labels).sum().item()
    return total_correct / total_samples

if __name__ == "__main__":
    # Start training loop
    train_accuracies, test_accuracies = [], []
    for epoch in range(1, 21):
        train_loss, train_accuracy = train_model_one_epoch(model, training_loader)
        test_accuracy = evaluate_model(model, testing_loader)
        lr_scheduler.step()
        print(f"Epoch {epoch}: Train Acc = {train_accuracy:.4f}, Test Acc = {test_accuracy:.4f}")
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    # Save plots
    plt.figure()
    plt.plot(train_accuracies, label='Train')
    plt.plot(test_accuracies, label='Test')
    plt.title("Config Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("config_accuracy.png")
    plt.close()
