import os
import json
import logging
import os
import random
from collections import defaultdict, OrderedDict, Counter
import pickle
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
from typing import List

from pysgg.utils.comm import get_world_size, is_main_process, synchronize,get_rank
from pysgg.utils.imports import import_file
from pysgg.utils.miscellaneous import save_labels
from pysgg.data.datasets.visual_genome import load_info,load_image_filenames,get_VG_statistics #,load_graphs
from pysgg.data.datasets.bi_lvl_rsmp import apply_resampling
from pysgg.data.build import make_data_sampler,make_batch_data_sampler
from pysgg.data.collate_batch import BatchCollator
from pysgg.data.transforms import transforms as T
from pysgg.structures.bounding_box import BoxList
from pysgg.structures.boxlist_ops import boxlist_iou, split_boxlist, cat_boxlist

#数据预处理函数
def build_transforms(is_train=True):
    if is_train:
        min_size = [600]
        max_size = 1000
        flip_horizontal_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0
    else:
        min_size = 600
        max_size = 1000
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = True
    normalize_transform = T.Normalize(
        mean=[102.9801,115.9465,115.9465], std=[1.0,1.0,1.0], to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            color_jitter,
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_horizontal_prob),
            T.RandomVerticalFlip(flip_vertical_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform

BOX_SCALE = 1024  # Scale at which we have the boxes


def resampling_dict_generation(dataset, category_list, logger):

    logger.info("using resampling method:" + dataset.resampling_method)
    repeat_dict_dir = None
    curr_dir_repeat_dict = os.path.join(dataset.args.output_dir, "repeat_dict.pkl")
    if repeat_dict_dir is not None and repeat_dict_dir != "" or os.path.exists(curr_dir_repeat_dict):
        if os.path.exists(curr_dir_repeat_dict):
            repeat_dict_dir = curr_dir_repeat_dict

        logger.info("load repeat_dict from " + repeat_dict_dir)
        with open(repeat_dict_dir, 'rb') as f:
            repeat_dict = pickle.load(f)

        return repeat_dict

    else:
        logger.info(
            "generate the repeat dict according to hyper_param on the fly")

        if dataset.resampling_method in ["bilvl", 'lvis']:
            # when we use the lvis sampling method,
            global_rf = 0.1
            logger.info(f"global repeat factor: {global_rf};  ")
            if dataset.resampling_method == "bilvl":
                # share drop rate in lvis sampling method
                dataset.drop_rate = 0.9
                logger.info(f"drop rate: {dataset.drop_rate};")
            else:
                dataset.drop_rate = 0.0
        else:
            raise NotImplementedError(dataset.resampling_method)

        F_c = np.zeros(len(category_list))
        for i in range(len(dataset)):
            anno = dataset.get_groundtruth(i)
            tgt_rel_matrix = anno.get_field('relation')
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs,
                                          tgt_tail_idxs].contiguous().view(-1)

            for each_rel in tgt_rel_labs:
                F_c[each_rel] += 1

        total = sum(F_c)
        F_c /= (total + 1e-11)

        rc_cls = {
            i: 1 for i in range(len(category_list))
        }
        global_rf = 0.1

        reverse_fc = global_rf / (F_c[1:] + 1e-11)
        reverse_fc = np.sqrt(reverse_fc)
        final_r_c = np.clip(reverse_fc, a_min=1.0, a_max=np.max(reverse_fc) + 1)
        # quantitize by random number
        rands = np.random.rand(*final_r_c.shape)
        _int_part = final_r_c.astype(int)
        _frac_part = final_r_c - _int_part
        rep_factors = _int_part + (rands < _frac_part).astype(int)

        for i, rc in enumerate(rep_factors.tolist()):
            rc_cls[i + 1] = int(rc)

        repeat_dict = {}
        for i in range(len(dataset)):
            anno = dataset.get_groundtruth(i)
            tgt_rel_matrix = anno.get_field('relation')
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0).numpy()
            tgt_head_idxs = tgt_pair_idxs[:, 0].reshape(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].reshape(-1)
            tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].numpy(
            ).reshape(-1)

            hit_rel_labels_r_c = []
            curr_rel_lables = []

            for rel_label in tgt_rel_labs:
                if rel_label not in curr_rel_lables:
                    curr_rel_lables.append(rel_label)
                    hit_rel_labels_r_c.append(rc_cls[rel_label])

            hit_rel_labels_r_c = np.array(hit_rel_labels_r_c)

            r_c = 1
            if len(hit_rel_labels_r_c) > 0:
                r_c = int(np.max(hit_rel_labels_r_c))
            repeat_dict[i] = r_c

        repeat_dict['cls_rf'] = rc_cls


        return repeat_dict


def boxlist_dist(boxlist1:BoxList,boxlist2:BoxList):
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)
    box1,box2 = boxlist1.bbox,boxlist2.bbox

    a = torch.stack(((box1[:,2] - box1[:,0])/2,(box1[:,3]-box1[:,1])/2),dim = 1)
    b = torch.stack(((box2[:,2] - box2[:,0])/2,(box2[:,3]-box2[:,1])/2),dim = 1)

    # 计算距离矩阵
    dist_matrix = torch.cdist(a, b) / BOX_SCALE
    return dist_matrix

#用于加载 Visual Genome 数据集中图像、对象框、类别、属性和关系数据的函数。
#获取所有图片的box框，box类别，属性及关系信息
def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap,is_filter = True):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5 ；包含对象框、类别和关系数据
        split: (train, val, or test)
        num_im: Number of images we want 要加载的图像数量
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.) 是否过滤没有关系的图像
        filter_non_overlap: If training, filter images that dont overlap.是否过滤没有重叠对象对的图像
    Return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    #从 HDF5 文件中读取数据集划分信息 data_split。
    data_split = roi_h5['split'][:]
    split_flag = 2 if split == 'test' else 0 #2代表测试集；0代表训练或验证
    #创建布尔掩码 split_mask，标记当前划分对应的图像。
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    #保证后续的处理就不会包含那些没有边界框的图像
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0
    #根据不同的参数和需求来筛选出特定的数据集子集，例如训练集、验证集或测试集，并控制加载的图像数量。
    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[: num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[: num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]
    #根据 image_index 更新了 split_mask，
    # 以明确哪些图像在当前的数据集划分中会被使用。
    # 接着，从 HDF5 文件中加载了这些图像的类别标签、属性和边界框信息，
    # 并进行了一些基本的数据合理性检查。
    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, : 2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, : 2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0]
            == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
            obj_idx = _relations[i_rel_start: i_rel_end
                                 + 1] - i_obj_start  # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            # (num_rel, 3), representing sub, obj, and pred
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)


        if is_filter:
            t1,t2 = 0.1,0.1 ## 这里t1和t2对应你公式中的参数，你自己可以修改
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            dist = boxlist_dist(boxes_i_obj,boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            dist_over = dist[rels[:, 0], rels[:, 1]]
            inc = np.where((rel_overs > t2) | (dist_over<t1))[0]
            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue 

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, gt_attributes, relationships

class VGDataset(torch.utils.data.Dataset):

    def __init__(self, args,split = 'train',transforms=None,
                num_im=-1, num_val_im=5000, check_img_file=False,
                filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # for debug
        #
        # num_im = 20000
        # num_val_im = 1000
        if args.debug:
            num_im = 6000
            num_val_im = 600
        assert split in {'train', 'val', 'test'}
        self.flip_aug = flip_aug #是否使用水平翻转增强
        self.split = split
        self.img_dir = args.img_dir #图像文件目录
        self.dict_file = args.dict_file #包含类别、关系和属性映射的文件。
        self.roidb_file = args.roidb_file #包含对象框、类别和关系标注的文件。
        self.image_file = args.image_file #包含图像文件名和信息的文件。
        self.filter_non_overlap = filter_non_overlap and self.split == 'train' #是否过滤没有重叠的对象对（仅在训练集时启用）。
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train' #是否过滤重复的关系（仅在训练集时启用）。
        self.transforms = transforms #数据增强和预处理操作。
        self.repeat_dict = None
        self.check_img_file = check_img_file
        self.args = args
        # self.remove_tail_classes = False

        #调用 load_info 函数，从 dict_file 中加载类别、关系和属性的映射信息。
        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(
            self.dict_file)  # contiguous 151, 51 containing __background__

        logger = logging.getLogger("pysgg.dataset")
        self.logger = logger
        #创建一个字典 categories，将类别索引映射到类别名称。
        self.categories = {i: self.ind_to_classes[i]
                           for i in range(len(self.ind_to_classes))}
#         调用 load_graphs 函数，从 roidb_file 中加载图像的关系图数据，包括：
                 # split_mask: 数据集的划分掩码。
                 # gt_boxes: 真实对象框。
                 # gt_classes: 真实对象类别。
                 # gt_attributes: 真实对象属性。
                 # relationships: 对象之间的关系。
        self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
            self.roidb_file, self.split, num_im, num_val_im=num_val_im,
            filter_empty_rels=True,
            filter_non_overlap=self.filter_non_overlap,
        )
        #load_image_filenames 函数，加载图像文件名和图像信息。
        self.filenames, self.img_info = load_image_filenames(
            self.img_dir, self.image_file, self.check_img_file)  # length equals to split_mask
        #根据 split_mask 过滤出当前划分（train、val 或 test）对应的图像文件名和图像信息。
        self.filenames = [self.filenames[i]
                          for i in np.where(self.split_mask)[0]]
        self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]
        #创建一个索引列表 idx_list，用于遍历数据集。
        self.idx_list = list(range(len(self.filenames)))
        #创建一个字典 id_to_img_map，将索引映射到图像 ID。
        self.id_to_img_map = {k: v for k, v in enumerate(self.idx_list)}
        #初始化 pre_compute_bbox 为 None，用于存储预计算的对象框。
        self.pre_compute_bbox = None

        if self.split == 'train':
            self.resampling_method = 'bilvl'
            # assert self.resampling_method in ['bilvl', 'lvis']

            self.global_rf = 0.1
            self.drop_rate = 0.9
            # creat repeat dict in main process, other process just wait and load
            if get_rank() == 0:
                repeat_dict = resampling_dict_generation(self, self.ind_to_predicates, logger)
                self.repeat_dict = repeat_dict
                with open(os.path.join(args.output_dir, "repeat_dict.pkl"), "wb") as f:
                    pickle.dump(self.repeat_dict, f)

            synchronize()
            self.repeat_dict = resampling_dict_generation(self, self.ind_to_predicates, logger)

            duplicate_idx_list = []
            for idx in range(len(self.filenames)):
                r_c = self.repeat_dict[idx]
                duplicate_idx_list.extend([idx for _ in range(r_c)])
            self.idx_list = duplicate_idx_list

        # if cfg.MODEL.ROI_RELATION_HEAD.REMOVE_TAIL_CLASSES and self.split == 'train':
        #     self.remove_tail_classes = True
    
    #获取单个样本 (__getitem__)
    def __getitem__(self, index):
        # if self.split == 'train':
        #    while(random.random() > self.img_info[index]['anti_prop']):
        #        index = int(random.random() * len(self.filenames))
        if self.repeat_dict is not None:# 上面这个值是none，这个略过
            index = self.idx_list[index]

        img = Image.open(self.filenames[index]).convert("RGB")
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']),
                  ' ', str(self.img_info[index]['height']), ' ', '=' * 20)

        target = self.get_groundtruth(index, flip_img=False)
        # todo add pre-compute boxes
        pre_compute_boxlist = None
        if self.pre_compute_bbox is not None:
            # index by image id
            pre_comp_result = self.pre_compute_bbox[int(
                self.img_info[index]['image_id'])]
            boxes_arr = torch.as_tensor(pre_comp_result['bbox']).reshape(-1, 4)
            pre_compute_boxlist = BoxList(boxes_arr, img.size, mode='xyxy')
            pre_compute_boxlist.add_field(
                "pred_scores", torch.as_tensor(pre_comp_result['scores']))
            pre_compute_boxlist.add_field(
                'pred_labels', torch.as_tensor(pre_comp_result['cls']))

        if self.transforms is not None:
            if pre_compute_boxlist is not None:
                # cat the target and precompute boxes and transform them together
                targets_len = len(target)
                target.add_field("scores", torch.zeros((len(target))))
                all_boxes = cat_boxlist([target, pre_compute_boxlist])
                img, all_boxes = self.transforms(img, all_boxes)
                resized_boxes = split_boxlist(
                    all_boxes, (targets_len, targets_len + len(pre_compute_boxlist)))
                target = resized_boxes[0]
                target.remove_field("scores")
                pre_compute_boxlist = resized_boxes[1]
                target = (target, pre_compute_boxlist)
            else:
                img, target = self.transforms(img, target)

        return img, target, index

    def get_statistics(self):
        fg_matrix, bg_matrix, rel_counter_init = get_VG_statistics(self,
                                                 must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = fg_matrix / fg_matrix.sum(2)[:, :, None] + eps

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_attributes,
        }

        rel_counter = Counter()

        for i in tqdm(self.idx_list):
            
            relation = self.relationships[i].copy()  # (num_rel, 3)
            if self.filter_duplicate_rels:
                # Filter out dupes!
                assert self.split == 'train'
                old_size = relation.shape[0]
                all_rel_sets = defaultdict(list)
                for (o0, o1, r) in relation:
                    all_rel_sets[(o0, o1)].append(r)
                relation = [(k[0], k[1], np.random.choice(v))
                            for k, v in all_rel_sets.items()]
                relation = np.array(relation, dtype=np.int32)

            if self.repeat_dict is not None:
                relation, _ = apply_resampling(i, 
                                               relation,
                                               self.repeat_dict,
                                               self.drop_rate,)

            for i in relation[:, -1]:
                if i > 0:
                    rel_counter[i] += 1

        cate_num = []
        cate_num_init = []
        cate_set = []
        counter_name = []

        sorted_cate_list = [i[0] for i in rel_counter_init.most_common()]
        lt_part_dict = [None, 'b', 't', 't', 't', 'b', 'b', 'b', 'h', 'b', 't', 'b', 't', 't', 't', 't', 'b',
                        't', 't', 'b', 'h', 'b', 'h', 'b', 't', 'b', 't', 't', 't', 'h', 'h', 'h', 't', 'b',
                        't', 'b', 't', 't', 'b', 't', 'b', 'b', 't', 'b', 't', 't', 'b', 'b', 'h', 'b', 'b']
        for cate_id in sorted_cate_list:
            if lt_part_dict[cate_id] == 'h':
                cate_set.append(0)
            if lt_part_dict[cate_id] == 'b':
                cate_set.append(1)
            if lt_part_dict[cate_id] == 't':
                cate_set.append(2)

            counter_name.append(self.ind_to_predicates[cate_id])  # list start from 0
            cate_num.append(rel_counter[cate_id])  # dict start from 1
            cate_num_init.append(rel_counter_init[cate_id])  # dict start from 1

        pallte = ['r', 'g', 'b']
        color = [pallte[idx] for idx in cate_set]


        fig, axs_c = plt.subplots(2, 1, figsize=(13, 10), tight_layout=True)
        fig.set_facecolor((1, 1, 1))

        axs_c[0].bar(counter_name, cate_num_init, color=color, width=0.6, zorder=0)
        axs_c[0].grid()
        plt.sca(axs_c[0])
        plt.xticks(rotation=-90, )

        axs_c[1].bar(counter_name, cate_num, color=color, width=0.6, zorder=0)
        axs_c[1].grid()
        axs_c[1].set_ylim(0, 50000)
        plt.sca(axs_c[1])
        plt.xticks(rotation=-90, )

        save_file = os.path.join(self.args.output_dir, f"rel_freq_dist.png")
        fig.savefig(save_file, dpi=300)


        return result

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, evaluation=False, flip_img=False, inner_idx=True):
        if not inner_idx: #not true = false 跳过
            # here, if we pass the index after resampeling, we need to map back to the initial index
            if self.repeat_dict is not None:
                index = self.idx_list[index]

        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE：1024
        #gt_boxes调用load_graphs获取box信息，类别，关系信息
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h) 
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v))
                        for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        relation_non_masked = None
        if self.repeat_dict is not None:
            relation, relation_non_masked = apply_resampling(index, 
                                                              relation,
                                                             self.repeat_dict,
                                                             self.drop_rate,)
        # add relation to target
        num_box = len(target)
        relation_map_non_masked = None
        if self.repeat_dict is not None:
            relation_map_non_masked = torch.zeros((num_box, num_box), dtype=torch.long)


        relation_map = torch.zeros((num_box, num_box), dtype=torch.long)
        for i in range(relation.shape[0]):
            # Sometimes two objects may have multiple different ground-truth predicates in VisualGenome.
            # In this case, when we construct GT annotations, random selection allows later predicates
            # having the chance to overwrite the precious collided predicate.
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] != 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
                    if relation_map_non_masked is not None  :
                        relation_map_non_masked[int(relation_non_masked[i, 0]), 
                                                int(relation_non_masked[i, 1])] = int(relation_non_masked[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
                if relation_map_non_masked is not None  :
                    relation_map_non_masked[int(relation_non_masked[i, 0]), 
                                            int(relation_non_masked[i, 1])] = int(relation_non_masked[i, 2])


        target.add_field("relation", relation_map, is_triplet=True)
        if relation_map_non_masked is not None :
             target.add_field("relation_non_masked", relation_map_non_masked.long(), is_triplet=True)


        target = target.clip_to_image(remove_empty=False)
        target.add_field("relation_tuple", torch.LongTensor(
                relation))  # for evaluation
        return target

    def __len__(self):
        return len(self.idx_list)


def get_dataset_distribution(args,train_data):
    """save relation frequency distribution after the sampling etc processing
    the data distribution that model will be trained on it

    Args:
        train_data ([type]): [description]
        dataset_name ([type]): [description]
    """
    # 
    if is_main_process():
        print("Get relation class frequency distribution on dataset.")
        pred_counter = Counter()
        for i in tqdm(range(len(train_data))):
            tgt_rel_matrix = train_data.get_groundtruth(i, inner_idx=False).get_field("relation")
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1).numpy()
            for each in tgt_rel_labs:
                pred_counter[each] += 1

        with open(os.path.join(args.output_dir, "pred_counter.pkl"), 'wb') as f:
            pickle.dump(pred_counter, f)

        from pysgg.data.datasets.visual_genome import HEAD, TAIL, BODY
        
        head = HEAD
        body = BODY
        tail = TAIL

        count_sorted = []
        counter_name = []
        cate_set = []
        cls_dict = train_data.ind_to_predicates

        for idx, name_set in enumerate([head, body, tail]):
            # sort the cate names accoding to the frequency
            part_counter = []
            for name in name_set:
                part_counter.append(pred_counter[name])
            part_counter = np.array(part_counter)
            sorted_idx = np.flip(np.argsort(part_counter))

            # reaccumulate the frequency in sorted index
            for j in sorted_idx:
                name = name_set[j]
                cate_set.append(idx)
                counter_name.append(cls_dict[name])
                count_sorted.append(pred_counter[name])

        count_sorted = np.array(count_sorted)

        fig, axs_c = plt.subplots(1, 1, figsize=(16, 5), tight_layout=True)
        palate = ['r', 'g', 'b']
        color = [palate[idx] for idx in cate_set]
        axs_c.bar(counter_name, count_sorted, color=color)
        axs_c.grid()
        plt.xticks(rotation=-60)
        axs_c.set_ylim(0, 50000)
        fig.set_facecolor((1, 1, 1))

        save_file = os.path.join(args.output_dir, "rel_freq_dist.png")
        fig.savefig(save_file, dpi=300)
    synchronize()

def get_dataset_statistics(args):
    """
    get dataset statistics (e.g., frequency bias) from training data
    will be called to help construct FrequencyBias module
    """
    logger = logging.getLogger(__name__)
    logger.info('-' * 100)
    logger.info('get dataset statistics...')
    data_statistics_name = ''.join(args.dataset_name) + '_train_statistics'
    save_file = os.path.join(args.output_dir, "{}.cache".format(data_statistics_name))
    if os.path.exists(save_file):
        logger.info('Loading data statistics from: ' + str(save_file))
        logger.info('-' * 100)
        return torch.load(save_file, map_location=torch.device("cpu"))
    dataset = VGDataset(args,split='train')
    if "VG_stanford" in args.dataset_name:
        get_dataset_distribution(args,dataset)

    result = dataset.get_statistics()
    logger.info('Save data statistics to: ' + str(save_file))
    logger.info('-' * 100)
    torch.save(result, save_file)
    return result

#用于创建训练，验证和测试的数据加载器
def make_data_loader(args,mode='train', is_distributed=True, start_iter=0):
    assert mode in {'train', 'val', 'test'}
    num_gpus = get_world_size()
    is_train = mode == 'train'
    if is_train:
        images_per_batch = args.train_pre_batch #default=4
        assert (
                images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus #4=4//1
        shuffle = True
        num_iters = args.max_iter
    else:
        images_per_batch = args.test_pre_batch #default=1
        assert (
                images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    # if images_per_gpu > 1:
    #     logger = logging.getLogger(__name__)
    #     logger.warning(
    #         "When using more than one image per GPU you may encounter "
    #         "an out-of-memory (OOM) error if your GPU does not have "
    #         "sufficient memory. If this happens, you can reduce "
    #         "SOLVER.IMS_PER_BATCH (for training) or "
    #         "TEST.IMS_PER_BATCH (for inference). For training, you must "
    #         "also adjust the learning rate and schedule length according "
    #         "to the linear scaling rule. See for example: "
    #         "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
    #     )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1]


    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    #数据预处理函数
    transforms = build_transforms(is_train)
    #加载数据集，并根据模式（train、val 或 test）进行初始化
    datasets = VGDataset(args,split=mode,transforms=transforms)
    #将数据集包装为列表，便于后续处理
    datasets = [datasets]
    if is_train:
        # save category_id to label name mapping
        #如果是训练模式，保存类别 ID 到标签名称的映射
        save_labels(datasets, args.output_dir)

    data_loaders = []
    for dataset in datasets:
        # print('============')
        # print(len(dataset))
        # print(images_per_gpu)
        # print('============')
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BatchCollator(args.size_divisbility) 
        num_workers = args.num_workers
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders[0]


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    # parser.add_argument('--output_dir',type=str,default='checkpoints/sgdet-BGNNPredictor')
    parser.add_argument('--dataset_name',type=str,default='VG_stanford')
    parser.add_argument('--dataset_name_test',type=str,default='VG_stanford_test')
    parser.add_argument('--dataset_name_val',type=str,default='VG_stanford_val')
    #图像存放的目录
    parser.add_argument('--img_dir',type=str,default='datasets/vg/stanford_spilt/VG_100k_images')
    # parser.add_argument('--glove_dir',type=str,default='datasets/vg/stanford_spilt/glove') 
    #该文件存储了 Region of Interest (ROI) 信息，包含 边界框 (bounding boxes)、类别 (class labels) 和 关系 (relationships)。
    parser.add_argument('--roidb_file',type=str,default='datasets/vg/VG-SGG-with-attri.h5')
    #JSON 文件包含 类别 (objects) 和关系 (relationships) 的映射，例如 "car" 可能对应类别 ID 10。
    parser.add_argument('--dict_file',type=str,default='datasets/vg/VG-SGG-dicts-with-attri.json')
    #这个 JSON 文件存储了 图片的文件名信息，用于从 img_dir 目录中加载图片
    parser.add_argument('--image_file',type=str,default='datasets/vg/image_data.json')
    parser.add_argument('--debug',type=bool,default = False)
    args = parser.parse_args()
    #加载vg数据集
    data = VGDataset(args=args,split='val',transforms=None)

    print(len(data))