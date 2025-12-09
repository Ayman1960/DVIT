import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50WithUpsampling(nn.Module):
    def __init__(self, pretrained=False, num_classes=1):
        super(ResNet50WithUpsampling, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)
        if num_classes != 1000:
            self.resnet50.fc = nn.Identity()  # Remove fully connected layer
        
        # Add upsampling path
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1),
            # nn.Sigmoid()  # Use Sigmoid activation for binary classification
        )
    
    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        
        x = self.resnet50.layer1(x)
        print(x.shape)
        x = self.resnet50.layer2(x)
        print(x.shape)
        x = self.resnet50.layer3(x)
        print(x.shape)
        x = self.resnet50.layer4(x)
        print(x.shape)
        
        x = self.upsample(x)
        
        return x

class ResNet34WithUpsampling(nn.Module):
    def __init__(self, pretrained=False, num_classes=1):
        super(ResNet34WithUpsampling, self).__init__()
        self.resnet34 = models.resnet34(pretrained=pretrained)
        if num_classes != 1000:
            self.resnet34.fc = nn.Identity()  # Remove fully connected layer
        
        # Add upsampling path
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Use Sigmoid activation for binary classification
        )
    
    def forward(self, x):
        x = self.resnet34.conv1(x)
        x = self.resnet34.bn1(x)
        x = self.resnet34.relu(x)
        x = self.resnet34.maxpool(x)
        
        x = self.resnet34.layer1(x)
        x = self.resnet34.layer2(x)
        x = self.resnet34.layer3(x)
        x = self.resnet34.layer4(x)
        
        x = self.upsample(x)
        
        return x

if __name__ == "__main__":
    model = ResNet34WithUpsampling(pretrained=True, num_classes=1)
    input_tensor = torch.randn(1, 3, 512, 512)
    output = model(input_tensor)
    print(output.shape)  # The output shape should be [1, 1, 512, 512]
