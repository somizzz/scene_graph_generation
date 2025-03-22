import torch
import torch.nn as nn
from pysgg.modeling import registry
from pysgg.modeling.box_coder import BoxCoder
from pysgg.modeling.rpn.retinanet.retinanet import build_retinanet
from pysgg.modeling.rpn.anchor_generator import AnchorGenerator
from pysgg.modeling.rpn.inference import RPNPostProcessor
from pysgg.modeling.rpn.loss import (Matcher, 
BalancedPositiveNegativeSampler,
RPNLossComputation,generate_rpn_labels)

def make_rpn_postprocessor(rpn_box_coder, is_train):
    fpn_post_nms_top_n = 1000
    if not is_train:
        fpn_post_nms_top_n = 1000

    pre_nms_top_n = 6000
    post_nms_top_n = 1000
    if not is_train:
        pre_nms_top_n = 6000
        post_nms_top_n = 1000
    fpn_post_nms_per_batch = False
    nms_thresh = 0.7
    min_size = 0
    add_gt = True
    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        fpn_post_nms_per_batch=fpn_post_nms_per_batch,
        add_gt=add_gt,
    )
    return box_selector

def make_anchor_generator(args):
    anchor_sizes = [32, 64, 128, 256, 512]
    aspect_ratios = [0.2323283, 0.63365731, 1.28478321, 3.15089189]
    anchor_stride = [4, 8, 16, 32, 64]
    straddle_thresh = 0
    assert len(anchor_stride) == len(
        anchor_sizes
    ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh
    )
    return anchor_generator


def make_rpn_loss_evaluator(box_coder):
    matcher = Matcher(
        0.7,
        0.3,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        1000, 0.25
    )

    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels
    )
    return loss_evaluator

class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, args, in_channels):
        super(RPNModule, self).__init__()

        self.args = args

        anchor_generator = make_anchor_generator(args)

        rpn_head = registry.RPN_HEADS['SingleConvRPNHead']
        head = rpn_head(
            args, in_channels, 256, anchor_generator.num_anchors_per_location()[0]
        )

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_postprocessor(rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        # if self.cfg.MODEL.RPN_ONLY:
        #     # When training an RPN-only model, the loss is determined by the
        #     # predicted objectness and rpn_box_regression values and there is
        #     # no need to transform the anchors into predicted boxes; this is an
        #     # optimization that avoids the unnecessary transformation.
        #     boxes = anchors
        # else:
        #     # For end-to-end models, anchors must be transformed into boxes and
        #     # sampled into a training batch.
        with torch.no_grad():
            boxes = self.box_selector_train(
                anchors, objectness, rpn_box_regression, targets
            )
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        # if self.cfg.MODEL.RPN_ONLY:
        #     # For end-to-end models, the RPN proposals are an intermediate state
        #     # and don't bother to sort them in decreasing score order. For RPN-only
        #     # models, the proposals are the final output and we return them in
        #     # high-to-low confidence order.
        #     inds = [
        #         box.get_field("objectness").sort(descending=True)[1] for box in boxes
        #     ]
        #     boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_rpn(args,in_channels):
    return RPNModule(args,in_channels)