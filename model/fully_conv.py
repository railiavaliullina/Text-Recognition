import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConvolutionalNetwork(nn.Module):
    def __init__(self, input_channels, class_count):
        super(FullyConvolutionalNetwork, self).__init__()
        lrelu = nn.LeakyReLU()
        pool = nn.MaxPool2d(2, 2)
        self.layers = []
        for depth in [32, 32]:
            self.layers.append(nn.Conv2d(input_channels, depth, 3, padding=1))
            self.layers.append(lrelu)
            input_channels = 32
        self.layers.append(pool)
        for depth in [64, 128, 256]:
            for _ in range(3):
                self.layers.append(nn.Conv2d(input_channels, depth, 3, padding=1))
                self.layers.append(nn.BatchNorm2d(depth))
                self.layers.append(lrelu)
                input_channels = depth
            if depth == 64:
                self.layers.append(pool)
            else:
                self.layers.append(nn.ZeroPad2d((1, 1, 0, 0)))
                self.layers.append(nn.MaxPool2d(2, stride=(2, 1)))

        depths = [512, 512, class_count]
        paddings = [(1, 1, 0, 0), (2, 2, 0, 0), (2, 2, 0, 0)]
        filter_sizes = [(2, 3), (1, 5), (1, 7)]
        for i in range(len(paddings)):
            depth = depths[i]
            self.layers.append(nn.ZeroPad2d(paddings[i]))
            self.layers.append(nn.Conv2d(input_channels, depth, filter_sizes[i]))
            input_channels = depth
            if i < len(paddings) - 1:
                self.layers.append(nn.BatchNorm2d(depth))
                self.layers.append(lrelu)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        x = torch.squeeze(x, dim=2)
        return x