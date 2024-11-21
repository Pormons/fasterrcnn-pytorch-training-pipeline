import torchvision
import torch
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from functools import partial

def create_model(num_classes=81, pretrained=True, coco_model=False):
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.COCO_V1
    )
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32),
    )

    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(81, pretrained=True)
    summary(model)