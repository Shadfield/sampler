import torch
import torch.nn as nn

from kornia.enhance import normalize


def mask_iou(pred_mask, target_mask):
    inter = torch.logical_and(pred_mask, target_mask).sum((-1, -2)).to(torch.float)
    union = torch.logical_or(pred_mask, target_mask).sum((-1, -2)).to(torch.float)
    eps = 1e-8
    jaccard = (inter + eps) / (union + eps)

    return jaccard.mean()


class ResnetFCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'fcn_resnet50', pretrained=True)
        self.model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]))
        self.num_classes = num_classes

    def forward(self, img):
        img = normalize(img, self.mean, self.std)
        return self.model(img)['out']

    @staticmethod
    def process(result):
        return result.softmax(dim=1).argmax(dim=1)

    def score(self, result, target, has_background=False):
        # Build confusion matrix
        coords = torch.stack((target.flatten(), result.flatten()))
        vals = torch.ones(target.numel(), dtype=torch.float, device=coords.device)
        confusion_matrix = torch.sparse_coo_tensor(
            coords,
            vals,
            size=(self.num_classes, self.num_classes),
            dtype=torch.float
        )
        confusion_matrix = confusion_matrix.to_dense()
        if has_background:
            confusion_matrix = confusion_matrix[1:, 1:]

        # convert confusion matrix to IoU
        intersect = confusion_matrix.diag()
        union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersect
        iou = (intersect[union != 0]) / (union[union != 0])
        return iou.mean()
