import os
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import PascalVOCDataset, Compose
from utils import iou
from loss import YoLoss
from model import Yolo
import torchvision.transforms as transforms


print("====== PREPARING DATASET ======")
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
dataset = PascalVOCDataset('data/train.csv', 'data/images', 'data/labels', transform=transform)

loader = torch.utils.data.DataLoader(dataset, batch_size=16)

lr = 2e-5
device = 'cuda' if torch.cuda.is_available else 'cpu'
epochs = 1000

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

model = Yolo().to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)
loss = YoLoss()

print("======= INITIALIZING TRAINING =======")
for epoch in range(epochs):
# pred_boxes, target_boxes = get_bboxes(loader, model, iou_threshold=0.5, threshold=0.4)
# avg = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format='midpoint'0)
# print(f"Train mAP: {avg}")
    train_fn(loader, model, opt, loss)