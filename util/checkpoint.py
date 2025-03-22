# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

# from pysgg.utils.c2_model_loading import load_c2_format
from pysgg.utils.imports import import_file
from pysgg.utils.model_serialization import load_state_dict
from pysgg.utils.model_zoo import cache_url
from pysgg.utils.checkpoint import Checkpointer
from pysgg.utils.c2_model_loading import (
    _load_c2_pickled_weights,
    _C2_STAGE_NAMES,
    _rename_weights_for_resnet,
    _rename_conv_weights_for_deformable_conv_layers
)

def rename_conv_weights_for_deformable_conv_layers(state_dict):
    import re
    logger = logging.getLogger(__name__)
    logger.info("Remapping conv weights for deformable conv weights")
    layer_keys = sorted(state_dict.keys())
    for ix, stage_with_dcn in enumerate([False,False,False,False], 1):
        if not stage_with_dcn:
            continue
        for old_key in layer_keys:
            pattern = ".*layer{}.*conv2.*".format(ix)
            r = re.match(pattern, old_key)
            if r is None:
                continue
            for param in ["weight", "bias"]:
                if old_key.find(param) == -1:
                    continue
                new_key = old_key.replace(
                    "conv2.{}".format(param), "conv2.conv.{}".format(param)
                )
                logger.info("pattern: {}, old_key: {}, new_key: {}".format(
                    pattern, old_key, new_key
                ))
                state_dict[new_key] = state_dict[old_key]
                del state_dict[old_key]
    return state_dict

def load_resnet_c2_format(cfg, f):
    state_dict = _load_c2_pickled_weights(f)
    conv_body = 'R-101-F'
    arch = conv_body.replace("-C4", "").replace("-C5", "").replace("-FPN", "")
    arch = arch.replace("-RETINANET", "")
    stages = _C2_STAGE_NAMES[arch]
    state_dict = _rename_weights_for_resnet(state_dict, stages)
    # ***********************************
    # for deformable convolutional layer
    state_dict = rename_conv_weights_for_deformable_conv_layers(state_dict)
    # ***********************************
    return dict(model=state_dict)

class DetectronCheckpointer(Checkpointer):
    def __init__(
            self,
            args,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
            save_to_disk=None,
            logger=None,
            custom_scheduler=False,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger, custom_scheduler
        )
        self.args = args

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "pysgg.config.paths_catalog", "pysgg/config/paths_catalog.py", True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://"):])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_resnet_c2_format(self.args, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
