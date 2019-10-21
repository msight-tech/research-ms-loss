from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torchvision.models as models
from ret_benchmark.modeling import registry


@registry.BACKBONES.register('resnet50')
class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.model.fc(x)  --remove
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'last_linear' in i:
                continue
            self.model.state_dict()[i].copy_(param_dict[i])

