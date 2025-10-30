import torch
import os

import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from model import ZFnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer1 = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transformer2 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

data = datasets.ImageFolder(os.path.join(r"Covid_Data", "train"), transformer1)
test = datasets.ImageFolder(os.path.join(r"Covid_Data", "test"), transformer2)

targets = torch.tensor(data.targets)
class_counts = torch.bincount(targets)
 
class_weights = (targets.numel() / (len(class_counts) * class_counts.float()))
sample_weights = class_weights[targets]
sampler = WeightedRandomSampler(sample_weights.double(), num_samples=len(sample_weights), replacement=True)

data_loader = DataLoader(data, batch_size=32, num_workers=2, sampler=sampler)
test_loader = DataLoader(test, batch_size=32, num_workers=2, shuffle=False)


def train(model, loader, criterion, epoch):
    best = 0

    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for i in range(epoch):
        model.train()
        loss = 0

        for images, labels in tqdm(loader, desc=f"Epoch {i + 1}"):
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()

            res = model(images)

            current_loss = criterion(res, labels)
            current_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            loss += current_loss.item() * images.size(0)

        epoch_loss = loss / len(loader.dataset)
        print(f"epoch[{i + 1}/{epoch}, loss:{epoch_loss:.10f}]")

        acc = evaluate(model, test_loader, criterion)
        if acc > best:
            best = acc
            save_model(model, "model.pt")

def evaluate(model, loader, criterion):
    model.eval()
    correct = 0
    loss = 0

    cnt = 0
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)

            res = model(images)

            current_loss = criterion(res, labels)
            loss += current_loss.item() * images.size(0)
            probs = torch.softmax(res, dim = 1)
            pred = probs.argmax(dim = 1)
            cnt += labels.size(0)
            correct += (pred == labels).sum().item()

    avg = loss / len(loader.dataset)
    acc = correct / cnt * 100

    print(f"Test Loss: {avg:.10f}, Test Accuracy: {acc:.10f}%")
    return acc

def save_model(model, path):
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    epochs = 10
    model = ZFnet(4).to(device)
    criterion = nn.CrossEntropyLoss()
    train(model, data_loader, criterion, epochs)