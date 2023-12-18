import numpy as np
import torch


def dataset_eval(data_loader, model, criterion, device):
    model.eval()
    loss = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss.append(criterion(preds, labels).item())

    accuracy = correct / total
    
    model.train()

    return accuracy, np.mean(loss)

