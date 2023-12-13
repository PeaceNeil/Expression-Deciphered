#resnet34
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import seaborn as sns
# Set basic parameters
img_size = 48
epochs = 15
batch_size = 64
learning_rate = 0.0001

# Data augmentation and loading
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    #transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    #transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root='/content/data/train', transform=train_transform)
test_dataset = datasets.ImageFolder(root='/content/data/test', transform=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained ResNet model and modify
model = models.resnet34(pretrained=True)
# Replace the last fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training model code and visualization part remain unchanged, can directly use the code above

# Train the model
train_acc = []
train_loss = []
test_acc = []
test_loss = []
y_true = []
y_pred = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Training
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_epoch_loss = running_loss / len(train_loader)
    train_epoch_acc = 100 * correct / total
    train_loss.append(train_epoch_loss)
    train_acc.append(train_epoch_acc)

    # Test
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_epoch_loss = test_running_loss / len(test_loader)
    test_epoch_acc = 100 * test_correct / test_total
    test_loss.append(test_epoch_loss)
    test_acc.append(test_epoch_acc)

    print(f'Epoch [{epoch+1}/{epochs}], '
          f'Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_acc:.2f}%, '
          f'Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_acc:.2f}%')

# Visualize results
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Training and test accuracy
axs[0].plot(train_acc, label='Train Accuracy')
axs[0].plot(test_acc, label='Test Accuracy')
axs[0].set_title('Training and Test Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend()

# Training and test loss
axs[1].plot(train_loss, label='Train Loss')
axs[1].plot(test_loss, label='Test Loss')
axs[1].set_title('Training and Test Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend()

plt.show()

class_labels = test_dataset.classes

# Generate confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix')
print(cm)
print('\nClassification Report')
print(classification_report(y_true, y_pred, target_names=class_labels))

# Visualize confusion matrix
plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()