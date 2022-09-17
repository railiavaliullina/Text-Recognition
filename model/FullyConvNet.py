from torch import nn

from configs.model_config import cfg as model_cfg
from model.fully_conv import FullyConvolutionalNetwork


# class FullyConvNet(nn.Module):
#     def __init__(self, cfg):
#         super(FullyConvNet, self).__init__()
#         self.cfg = cfg
#
#         self.relu = nn.LeakyReLU()
#         self.dropout = nn.Dropout(0.2)
#
#         self.conv_0 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
#         self.conv_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
#         self.mp_0 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
#
#         self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
#         self.BN_2 = nn.BatchNorm2d(64)
#         self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
#         self.mp_1 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
#
#         self.conv_5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
#         self.BN_5 = nn.BatchNorm2d(128)
#         self.conv_6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
#         self.mp_2 = nn.MaxPool2d(kernel_size=2, padding=(0, 1), stride=(2, 1))
#
#         self.conv_8 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
#         self.BN_8 = nn.BatchNorm2d(256)
#         self.conv_9 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
#         self.mp_3 = nn.MaxPool2d(kernel_size=(2, 2), padding=(0, 0), stride=(2, 1))
#
#         self.conv_11 = nn.Conv2d(256, 512, kernel_size=(2, 3), padding=(0, 1), stride=1)
#         self.conv_12 = nn.Conv2d(512, 512, kernel_size=(1, 5), padding=(0, 2), stride=1)
#         self.conv_13 = nn.Conv2d(512, self.cfg.num_classes, kernel_size=(1, 7), padding=(0, 3), stride=1)
#
#     def forward(self, x):
#         # print(x.size())
#         x = self.conv_0(x)
#         x = self.relu(x)
#         x = self.conv_1(x)
#         x = self.relu(x)
#         x = self.mp_0(x)
#
#         x = self.conv_2(x)
#         x = self.relu(x)
#         x = self.BN_2(x)
#         x = self.conv_3(x)
#         x = self.relu(x)
#         x = self.mp_1(x)
#
#         x = self.conv_5(x)
#         x = self.relu(x)
#         x = self.BN_5(x)
#         x = self.conv_6(x)
#         x = self.relu(x)
#         x = self.mp_2(x)
#
#         x = self.conv_8(x)
#         x = self.relu(x)
#         x = self.BN_8(x)
#         x = self.conv_9(x)
#         x = self.relu(x)
#         x = self.mp_3(x)
#
#         x = self.conv_11(x)
#         x = self.relu(x)
#         x = self.conv_12(x)
#         x = self.relu(x)
#         x = self.conv_13(x)
#         x = x.squeeze(2)
#         # print(x.size())
#         return x


def get_model():
    model = FullyConvolutionalNetwork(3, 63)
    return model  # .cuda()
