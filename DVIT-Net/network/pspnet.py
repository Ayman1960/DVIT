import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, size) for size in sizes]
        )
        self.bottleneck = nn.Conv2d(2560, in_channels, kernel_size=1)

    def _make_stage(self, in_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        return nn.Sequential(prior, conv)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        priors = [
            F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=True)
            for stage in self.stages
        ] + [x]
        t = torch.cat(priors, dim=1)
        bottle = self.bottleneck(t)
        return bottle


class PSPNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(PSPNet, self).__init__()
        original_model = models.resnet34(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
        )
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

        self.psp = PSPModule(512)
        self.drop_1 = nn.Dropout2d(p=0.3)
        self.final = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.psp(x)
        x = self.drop_1(x)
        x = self.final(x)
        x = F.interpolate(x, x_size[2:], mode="bilinear", align_corners=True)
        x = F.sigmoid(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0")
    input = torch.rand(1, 3, 256, 256)
    input = input.to(device)
    net = PSPNet()
    net.to(device)
    segout = net(input)
    print(segout.size())
    pass
