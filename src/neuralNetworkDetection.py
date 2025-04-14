import torch
from torch import nn

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(
            image_channels, 16, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=16, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=32, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=64, stride=2
        )
        

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
       

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

class block(nn.Module):
    def __init__(self, in_channels,out_channels,identity_downsample=None,stride=1):
        super(block,self).__init__()
        self.expansion = 4
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,3,stride,1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels * self.expansion,1,1,0),
            nn.BatchNorm2d(out_channels*self.expansion)
        )
        self.relu = nn.ReLU()
        self.identity_downsample= identity_downsample

    def forward(self,x):
        identity = x

        x = self.conv_layers(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity

        return self.relu(x)