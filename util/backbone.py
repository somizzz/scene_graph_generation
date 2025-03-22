from collections import OrderedDict

from torch import nn
import torch.nn.functional as F

from pysgg.layers import Conv2d
from pysgg.layers.batch_norm import FrozenBatchNorm2d
from pysgg.modeling import registry
from pysgg.modeling.make_layers import conv_with_kaiming_uniform
from pysgg.modeling.backbone.resnet import _STAGE_SPECS,_TRANSFORMATION_MODULES,_make_stage
from pysgg.modeling.backbone import fpn as fpn_module


class StemWithFixedBatchNorm(nn.Module):
    def __init__(self, out_channels = 64):
        super(StemWithFixedBatchNorm, self).__init__()

        # out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = FrozenBatchNorm2d(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

class ResNet(nn.Module):
    def __init__(self,in_channels = 64,
                out_channels = 256,
                conv_body = 'R-101-FPN',
                trans_func = 'BottleneckWithFixedBatchNorm'):
        super(ResNet,self).__init__()
        STAGE_WITH_DCN = [False,False,False,False]
        self.stem = StemWithFixedBatchNorm(out_channels=in_channels)
        stage_specs = _STAGE_SPECS[conv_body]
        transformation_module = _TRANSFORMATION_MODULES[trans_func]

        num_groups = 32
        width_per_group = 8
        stage2_bottleneck_channels = num_groups * width_per_group 
        stage2_out_channels = out_channels
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = STAGE_WITH_DCN[stage_spec.index -1]
            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                num_groups,
                False,
                first_stride=int(stage_spec.index > 1) + 1,
                dcn_config={
                    "stage_with_dcn": stage_with_dcn,
                    "with_modulated_dcn": False,
                    "deformable_groups": 1,
                }
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(2)
    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs

def build_resnet_fpn_backbone(args):
    body = ResNet(in_channels=args.res_in_channels,out_channels=args.res_out_channels)
    in_channels_stage2 = args.res_out_channels 
    out_channels = args.backbone_out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            False, False),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(args):
    return build_resnet_fpn_backbone(args)