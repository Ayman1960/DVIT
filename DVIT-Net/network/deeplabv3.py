import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)

        self.conv6 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))
        x4 = F.relu(self.bn4(self.conv4(x)))
        x5 = self.avg_pool(x)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = F.relu(self.bn6(self.conv6(x)))

        return x

class Decoder(nn.Module):
    def __init__(self, low_level_inplanes, low_level_planes, aspp_outplanes, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, low_level_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(low_level_planes)
        
        self.conv2 = nn.Conv2d(aspp_outplanes + low_level_planes, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)

    def forward(self, x, low_level_feat):
        low_level_feat = F.relu(self.bn1(self.conv1(low_level_feat)))
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.aspp = ASPP(2048, 256)
        self.decoder = Decoder(256, 48, 256, num_classes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        low_level_feat = x
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        
        x = F.interpolate(x, size=(x.size(2) * 4, x.size(3) * 4), mode='bilinear', align_corners=True)
        return F.sigmoid(x)

# Example usage
if __name__ == "__main__":
    model = DeepLabV3Plus(num_classes=1)
    input_tensor = torch.randn(8, 3, 512, 512)
    output = model(input_tensor)
    print(output.size())  # Output should be [8, 1, 512, 512]
