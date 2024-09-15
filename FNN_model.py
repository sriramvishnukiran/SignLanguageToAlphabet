import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from sklearn.model_selection import train_test_split

import numpy as np
# import random

# torch.backends.cudnn.deterministic = True
# random.seed(1)
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_model(train_loader, num_epochs, criterion, optimizer, model):
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

    print('Finished Training')
    PATH = './fnn.pth'
    torch.save(model.state_dict(), PATH)


def evaluate_model(test_loader, model, classes):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(26)]
        n_class_samples = [0 for i in range(26)]
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

        for i in range(26):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')


if __name__ == "__main__":
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    for i, data in enumerate(data_dict['data']):
        label = data_dict['labels'][i]
        if len(data) > 42:
            print(i, len(data), label)
    # print(len(data_dict['data']), len(data_dict['data']),
    #       np.array(data_dict['data']).shape)
    data = torch.tensor(data_dict['data'])
    labels = torch.tensor(data_dict['labels'])

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels)

    input_size = 42
    hidden_size = 35  # you can change this
    num_classes = 26
    num_epochs = 20
    batch_size = 4
    learning_rate = 0.001
    classes = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
               13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u", 21: "v", 22: "w", 23: "x", 24: "y", 25: "z"}

    # Data loader
    dataset_train = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=batch_size, shuffle=True)

    dataset_test = TensorDataset(x_test, y_test)
    test_loader = DataLoader(dataset=dataset_test,
                             batch_size=batch_size, shuffle=True)

    # Model, loss, and optimizer
    model = Net(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(train_loader, num_epochs, criterion, optimizer, model)
    evaluate_model(test_loader, model, classes)
