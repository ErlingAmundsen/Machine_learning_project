import sys
from random import seed

import torch
import torch.nn as nn
from data import Data
from matplotlib import pyplot as plt
from model import LinearNet
from torch.utils.data import DataLoader

seed(0)

# Hyper-parameters
EPOCHS = 1200
LEARNING_RATE = 0.0001

train = Data(dataset='train')
val = Data(dataset='val')
test = Data(dataset='test')
total = Data(dataset='total')

# Data loader
train_loader = DataLoader(dataset=train, batch_size=len(train), shuffle=True)
val_loader = DataLoader(dataset=val, batch_size=len(val), shuffle=False)
test_loader = DataLoader(dataset=test, batch_size=len(test), shuffle=False)
total_loader = DataLoader(dataset=total, batch_size=len(total), shuffle=False)

# Initialize model
model = LinearNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
total_step = len(total_loader)
loss_hist_train = []
acc_hist_train = []
for epoch in range(EPOCHS):
    total_train = 0
    correct_train = 0
    for i, (images, labels) in enumerate(total_loader):
        # Forward pass
        preds = model(images)
        _, predicted = torch.max(preds.data, 1)
        train_acc = (predicted == labels).sum().item() / len(labels)
        acc_hist_train.append(train_acc)
        loss = criterion(preds, labels)
        loss_hist_train.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save barth loss
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {train_acc:.2%}', end="\r")
print(f'Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {train_acc:.2%}')

fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_hist_train, label="Train loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_hist_train, label="Train accuracy")
plt.yscale("logit")
plt.title("Accuracy")
plt.legend() 
plt.show()

# get test score
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        preds = model(images)
        _, predicted = torch.max(preds.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f} %')
