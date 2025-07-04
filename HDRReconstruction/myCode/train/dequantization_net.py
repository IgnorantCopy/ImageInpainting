import utils
import torch
from torch import nn
import torch.nn.functional as F


class DequantizationNet(object):
    def __init__(self, is_trained=True):
        self.is_trained = is_trained

    @staticmethod
    def loss(prediction, ground_truth):
        return torch.mean((prediction - ground_truth) ** 2)

    @staticmethod
    def down_sample(x, out_channels, kernel_size):
        in_channels = x.shape[-1]
        padding = utils.same_padding(x, kernel_size)
        layer = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding),
            nn.LeakyReLU(0.1),
        )
        return layer(x)

    @staticmethod
    def up_sample(x, out_channels, skip_connection):
        x = F.interpolate(x, size=x.shape[-2:], mode='bilinear')
        padding1 = utils.same_padding(x, 3)
        layer1 = nn.Sequential(
            nn.Conv2d(x.shape[-1], out_channels, 3, 1, padding1),
            nn.LeakyReLU(0.1),
        )
        x = layer1(x)
        x = torch.concat([x, skip_connection], -1)
        padding2 = utils.same_padding(x, 3)
        layer2 = nn.Sequential(
            nn.Conv2d(x.shape[-1], out_channels, 3, 1, padding2),
            nn.LeakyReLU(0.1),
        )
        return layer2(x)

    def build_model(self, input_image):
        padding1 = utils.same_padding(input_image, 7)
        layer1 = nn.Sequential(
            nn.Conv2d(input_image.shape[-1], 16, 7, 1, padding1),
            nn.LeakyReLU(0.1),
        )
        x = layer1(input_image)

        padding2 = utils.same_padding(x, 7)
        layer2 = nn.Sequential(
            nn.Conv2d(x.shape[-1], 16, 7, 1, padding2),
            nn.LeakyReLU(0.1),
        )
        s1 = layer2(x)
        s2 = self.down_sample(s1, 32, 5)
        s3 = self.down_sample(s2, 64, 3)
        s4 = self.down_sample(s3, 128, 3)
        x = self.down_sample(s4, 256, 3)

        x = self.up_sample(x, 128, s4)
        x = self.up_sample(x, 64, s3)
        x = self.up_sample(x, 32, s2)
        x = self.up_sample(x, 16, s1)

        padding3 = utils.same_padding(x, 3)
        layer3 = nn.Sequential(
            nn.Conv2d(x.shape[-1], 3, 3, 1, padding3),
            nn.Tanh(),
        )
        return input_image + layer3(x)

    def train(self, batch_size=8, num_epochs=100):
        pass
