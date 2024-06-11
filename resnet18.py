import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torchvision.models.resnet import ResNet, BasicBlock

__all__ = ['resnet18']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

class ModifiedResNet(ResNet):
    def __init__(self, block, layers):
        super(ModifiedResNet, self).__init__(block, layers)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
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
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        x = self.reg_layer(x)
        return torch.abs(x)


def resnet18():
    """ResNet-18 model
        model pre-trained on ImageNet
    """
    model = ModifiedResNet(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model