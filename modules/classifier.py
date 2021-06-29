import torch.nn as nn
import torchvision.transforms.functional as tv
from efficientnet_pytorch import EfficientNet


class EffNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

    def forward(self, img):
        img = tv.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return self.model(img)

    @staticmethod
    def process(logits):
        return logits.softmax(dim=1).argmax(dim=1)
