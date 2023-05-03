import torch
import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionalBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.layers(x)

class Yolo(nn.Module):
    def __init__(self, in_channels=3, grid_size=7, n_classes=20, n_boxes=2):
        super(Yolo, self).__init__()

        self.darknet = nn.Sequential(
            ConvolutionalBlock(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            ConvolutionalBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            ConvolutionalBlock(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),
            ConvolutionalBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvolutionalBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            ConvolutionalBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvolutionalBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            ConvolutionalBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvolutionalBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            ConvolutionalBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvolutionalBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            ConvolutionalBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvolutionalBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            ConvolutionalBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvolutionalBlock(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            ConvolutionalBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvolutionalBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            ConvolutionalBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=2),
            ConvolutionalBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            ConvolutionalBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=2),
            ConvolutionalBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            ConvolutionalBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            ConvolutionalBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            ConvolutionalBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
        )

        self.dense_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * (grid_size**2), 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, (grid_size**2) * (n_classes + n_boxes * 5))
        )

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense_layers(x)
        return x