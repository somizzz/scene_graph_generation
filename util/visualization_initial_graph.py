import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from util.dataset import VGDataset
from pysgg.data.datasets.visual_genome import load_info,load_image_filenames
from util.dataset import load_graphs
import argparse

class GraphVisualizer:
    def __init__(self, args):
        """
        Initialize the GraphVisualizer class.
        :param gt_boxes: List of ground truth bounding boxes for each image.
        :param relationships: List of relationship pairs (subj_idx, obj_idx) for each image.
        """
        self.args = args
        self.img_dir = args.img_dir  # Assuming args has an img_dir attribute
        self.image_file = args.image_file  # Assuming args has an image_file attribute
        self.check_img_file = False
        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(
        args.dict_file)  # contiguous 151, 51 containing __background__
        self.load_first_image_info(roidb_file=args.roidb_file)

        self.filenames, self.img_info = load_image_filenames(
            self.img_dir, self.image_file, self.check_img_file)  # length equals to split_mask
        self.filenames = [self.filenames[i]
                          for i in np.where(self.split_mask)[0]]
        self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]
        self.idx_list = list(range(len(self.filenames)))

        self.id_to_img_map = {k: v for k, v in enumerate(self.idx_list)}

        print('load end')
        # self.gt_boxes = gt_boxes
        # self.relationships = relationships
        # self.dataset = VGDataset(args, split='train')

    def load_first_image_info(self,
        roidb_file, split='train', 
        num_im=-1, num_val_im=100, 
        filter_empty_rels=True, 
        filter_non_overlap=True, 
        is_filter=False):
        """
        加载第一张图片的信息（对象框、类别、属性和关系）。
        
        参数:
            roidb_file: HDF5 文件路径，包含对象框、类别和关系数据。
            split: 数据集划分，默认为 'train'。
            num_im: 要加载的图像数量，默认为 -1（加载所有图像）。
            num_val_im: 验证集图像数量，默认为 5000。
            filter_empty_rels: 是否过滤没有关系的图像，默认为 True。
            filter_non_overlap: 是否过滤没有重叠对象对的图像，默认为 True。
            is_filter: 是否启用额外的过滤条件（如 IoU 和距离过滤），默认为 True。
        
        返回:
            第一张图片的信息，包括对象框、类别、属性和关系。
        """
        # 调用 load_graphs 函数加载数据
        
        split_mask, boxes, gt_classes, gt_attributes, relationships = load_graphs(
            roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap, is_filter
        )
        
        # 检查是否有加载到数据
        if len(boxes) == 0:
            raise ValueError("没有加载到任何图像数据！")
        self.boxes = boxes
        self.gt_classes = gt_classes
        self.relationships = relationships
        self.split_mask = split_mask

    def visualize_single_image(self, index):
        """
        Process and visualize the data of the i-th image.
        :param i: Image index.
        """
        #获取指定索引 index 的图像的类别和关系数据
        classes = self.gt_classes[index]
        rel = self.relationships[index]
        #创建一个有向图
        gragh = nx.DiGraph()
        # idx = 0
        # nodes = []
        for i,name in enumerate(classes):
            gragh.add_node(i,name = self.ind_to_classes[name],type = 'obj')

        for (i,j,name) in rel:
            gragh.add_edge(i,j,name = self.ind_to_predicates[name])
        # 获取节点标签，这里我们用'name'属性作为节点的标签
        node_labels = nx.get_node_attributes(gragh, 'name')
        # 获取边的标签（即边的名字）
        edge_labels = nx.get_edge_attributes(gragh, 'name')
        # 使用spring_layout生成初始布局位置，然后按比例放大
        pos = nx.spring_layout(gragh,k = 1)  # 生成初始布局
        # scale_factor = 100.0  # 自定义缩放因子，数值越大，边看起来越长
        # for k, v in pos.items():
        #     pos[k] = v * scale_factor
        # 绘制图形
        plt.figure(figsize=(10, 8))  # 设置图像大小
        nx.draw(gragh, 
                pos=pos,
                with_labels=True, 
                labels=node_labels,  # 使用自定义的节点标签
                node_color='lightblue', 
                edge_color='gray',
                node_size=500,  # 调整节点大小
                font_size=8,    # 调整字体大小
                font_weight='bold')  # 字体加粗
        # 添加边的标签
        nx.draw_networkx_edge_labels(gragh, pos, edge_labels=edge_labels)


        plt.title("Graph Visualization")  # 设置图表标题
        # 保存图形到文件
        plt.savefig(f'{args.output_dir}/graph_{index}.png')  # 你可以更改文件名和路径
        plt.show()  # 显示图表

        # return self._prepare_adjacency_matrix_single_image(proposals, relationships)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument('--dataset_name', type=str, default='VG_stanford')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')  # Add debug flag
    parser.add_argument('--img_dir', type=str, default='/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/stanford_spilt/VG_100k_images')
    parser.add_argument('--dict_file', type=str, default='/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/VG-SGG-dicts-with-attri.json')
    parser.add_argument('--roidb_file', type=str, default='/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/VG-SGG-with-attri.h5')
    parser.add_argument('--image_file', type=str, default='/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/image_data.json')
    parser.add_argument('--output_dir', type=str, default='vusial_output', help='Directory to save output files')
    args = parser.parse_args()

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    visualizer = GraphVisualizer(args)

    # Visualize the first image
    visualizer.visualize_single_image(0)
