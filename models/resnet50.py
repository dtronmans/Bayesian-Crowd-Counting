import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torchvision.models.resnet import ResNet, Bottleneck

__all__ = ['resnet50']
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class ModifiedResNet(ResNet):
    def __init__(self, block, layers):
        super(ModifiedResNet, self).__init__(block, layers)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Optionally, add additional upsampling if needed
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.reg_layer(x)
        return torch.abs(x)


def resnet50():
    """ResNet-50 model
        model pre-trained on ImageNet
    """
    model = ModifiedResNet(Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model