import torch
from torch import nn
from pysgg.structures.image_list import to_image_list
# from pysgg.modeling.detector.generalized_rcnn import GeneralizedRCNN
from .backbone import build_backbone
from .rpn import build_rpn
from .roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, args):
        super(GeneralizedRCNN, self).__init__()
        self.args = args
        self.backbone = build_backbone(args)
        self.rpn = build_rpn(args, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(args, self.backbone.out_channels)


    def forward(self, images, targets=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        #训练的时候必须提供targets，真实的标注，否则会抛出错误
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        #图像预处理，将不同尺寸的图像填充到相同尺寸
        images = to_image_list(images)
        print("Type of images:", type(images))
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, logger)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            return losses

        return result
    

def build_detection_model(args):
    return GeneralizedRCNN(args) 