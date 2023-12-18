import sys

import torch
import torch.nn as nn
from data import Data
from eval import dataset_eval
from matplotlib import pyplot as plt
from model import ConvNet
from torch.utils.data import DataLoader

# Add current directory to path
sys.path.append('../')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
EPOCHS = 150
BATCH_SIZE = 50
LEARNING_RATE = 0.001

# # Data augmentation and normalization 
# train, test = np.load(file="./Data/fashion_train.npy"), np.load(file="./Data/fashion_test.npy")
# total = np.concatenate((train, test))
# X, y = total[:, :-1], total[:, -1]

# # get mean and std of data
# mean = np.mean(X)
# std = np.std(X)

# Load data into datasets
train = Data()
val = Data(val=True)
test = Data(test=True)

# Data loader
train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=False)


# Initialize model
model = ConvNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.000001)

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Train the model
total_step = len(train_loader)
loss_hist_train = []
loss_hist_val = []
acc_hist_train = []
acc_hist_val = []
for epoch in range(EPOCHS):
    train_loss = []
    total_train = 0
    correct_train = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        preds = model(images)
        _, predicted = torch.max(preds.data, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        loss = criterion(preds, labels)
        train_loss.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save barch loss
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}', end="\r")
            
    # calculate epoch train and test metrics
    train_accuracy = correct_train / total_train
    acc_hist_train.append(train_accuracy)
    train_loss = sum(train_loss) / len(train_loss)
    loss_hist_train.append(train_loss)
    val_accuracy, val_loss = dataset_eval(val_loader, model, criterion, device)
    acc_hist_val.append(val_accuracy)
    loss_hist_val.append(val_loss)

    print(f'Epoch [{epoch + 1}/{EPOCHS}]{"": >30}\nTrain Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}\nTrain Accuracy: {train_accuracy:.2%} - Val Accuracy: {val_accuracy:.2%}')
    
    # update learning rate
    scheduler.step(train_loss)

fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_hist_train, label="Train loss")
plt.plot(loss_hist_val, label="Validation loss")
plt.yscale("log")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_hist_train, label="Train accuracy")
plt.plot(acc_hist_val, label="Validation accuracy")
plt.yscale("logit")
plt.title("Accuracy")
plt.legend() 
plt.show()

# model validation
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    model.eval()
    correct_val = 0
    total_val = 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        _, predicted = torch.max(preds.data, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()

    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        _, predicted = torch.max(preds.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the {total_train} train images: {100 * correct_train / total_train}%')
    print(f'Accuracy of the model on the {total_val} validation images: {100 * correct_val / total_val}%')