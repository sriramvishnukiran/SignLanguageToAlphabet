import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 23 * 23, 128)
        self.dropout = nn.Dropout(0.20)
        self.fc2 = nn.Linear(128, 27)

    def forward(self, x):
        x = F.relu(self.conv1(self.bn1(x)))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(train_loader, num_epochs, criterion, optimizer, model):
    n_total_steps = len(train_loader)
    for epoch in tqdm(range(num_epochs), desc="Epochs", total=num_epochs, leave=True):
        loss = None
        for i, (images, labels) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", total=n_total_steps, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tqdm.write(
            f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Finished Training')
    PATH = './cnn.pth'
    torch.save(model.state_dict(), PATH)


def evaluate_model(test_loader, model, classes):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(len(classes))]
        n_class_samples = [0 for i in range(len(classes))]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(len(classes)):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')


if __name__ == "__main__":
    # Hyper-parameters
    num_epochs = 20
    batch_size = 1
    learning_rate = 0.001

    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
    ])

    dataset_root = 'data'
    dataset = torchvision.datasets.ImageFolder(
        root=dataset_root, transform=transform)

    train_data = []
    test_data = []

    # Define the fraction of data to keep for training (e.g., 80%)
    train_fraction = 0.9

    for class_idx in range(len(dataset.classes)):
        class_samples = [i for i, (_, label) in enumerate(
            dataset.samples) if label == class_idx]
        _, label = dataset.samples[class_samples[0]]
        random.shuffle(class_samples)
        split_index = int(len(class_samples) * train_fraction)
        train_data += class_samples[:split_index]
        test_data += class_samples[split_index:]

    # Create DataLoader for train and test datasets
    train_dataset = torch.utils.data.Subset(dataset, train_data)
    test_dataset = torch.utils.data.Subset(dataset, test_data)

    # Create DataLoaders for the train and test datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False)

    classes = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m", 13: "n",
               14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u", 21: "v", 22: "w", 23: "x", 24: "y", 25: "z"}

    model = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_model(train_loader, num_epochs, criterion, optimizer, model)
    evaluate_model(test_loader, model, classes)
