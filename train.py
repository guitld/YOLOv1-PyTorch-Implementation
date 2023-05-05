import os
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from dataset import PascalVOCDataset, Compose
from utils import iou, bounding_box
from loss import YoLoss
from model import Yolo
import torchvision.transforms as transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt

def train_fn(train_loader, model, optimizer, loss_fn, device):
    mean_loss = []

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum(mean_loss)/len(mean_loss)

def main():
    print("================================================== PREPARING DATASET  ==================================================")
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
    dataset = PascalVOCDataset('data/100examples.csv', 'data/images', 'data/labels', transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    lr = 2e-5
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    epochs = 1000
    model = Yolo().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = YoLoss()
    loop = trange(epochs)

    print("================================================= INITIALIZING TRAINING =================================================")
    for epoch in loop:
        loss = []
        loss.append(train_fn(loader, model, opt, loss_fn, device))
        loop.set_postfix(loss=loss)

    print(f"Final Loss: {sum(loss) / len(loss)}")

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(loss)
    plt.savefig('loss.png')
    plt.show()

if __name__ == '__main__':
    main()