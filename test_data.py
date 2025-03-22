import json
from PIL import Image

from pysgg.data.datasets.visual_genome import load_image_filenames
from util.dataset import load_graphs
# def display_first_image_info(img_dir, image_file, check_img_file=True):
#     """
#     Display information and image of the first image in the dataset.
#     Parameters:
#         img_dir: Directory containing the images.
#         image_file: JSON file containing image information.
#         check_img_file: Whether to check if the image file exists.
#     """
#     # Load image filenames and info,加载图像文件名和图像信息。
#     fns, img_info = load_image_filenames(img_dir, image_file, check_img_file)

#     if not fns:
#         print("No valid images found!")
#         return

#     # Get the first image filename and info
#     first_image_fn = fns[0]
#     first_image_info = img_info[0]

#     print("first image filenames:",fns[0])
#     print("first image information:",img_info[0])
#     # first image filenames: /home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/stanford_spilt/VG_100k_images/1.jpg
#     # first image information: {'width': 800, 'url': 'https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg', 'height': 600, 'image_id': 1, 'coco_id': None, 'flickr_id': None, 'anti_prop': 0.0}

#     # Display image information
#     print("First Image Information:")
#     print(json.dumps(first_image_info, indent=4))  # Pretty-print the info

#     # Load and display the image
#     try:
#         img = Image.open(first_image_fn)
#         print(f"\nImage Size: {img.size}")  # Print image dimensions
#         img.show()  # Open the image using the default image viewer
#     except Exception as e:
#         print(f"Failed to load image: {e}")

import h5py
import numpy as np

def load_first_image_info(roidb_file, split='train', num_im=-1, num_val_im=5000, filter_empty_rels=True, filter_non_overlap=True, is_filter=False):
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
    #print("真实box信息：",boxes)
    
    # 检查是否有加载到数据
    if len(boxes) == 0:
        raise ValueError("没有加载到任何图像数据！")
    
    # 获取第一张图片的信息
    first_image_boxes = boxes[0]
    first_image_classes = gt_classes[0]
    first_image_relationships = relationships[0]
    
    # 返回第一张图片的信息
    return {
        "boxes": first_image_boxes,  # 对象框 (x1, y1, x2, y2)
        "classes": first_image_classes,  # 对象类别
        "relationships": first_image_relationships  # 对象关系 (sub, obj, pred)
    }

def load_mapping_dicts(dict_file):
    """
    加载映射字典（对象类别和谓词）。
    
    参数:
        dict_file: JSON 文件路径，包含对象类别和谓词的映射。
    
    返回:
        idx_to_label: 对象类别 ID 到名称的映射。
        idx_to_predicate: 谓词 ID 到名称的映射。
    """
    with open(dict_file, 'r') as f:
        data = json.load(f)
    
    # 获取映射字典
    idx_to_label = data['idx_to_label']  # 对象类别 ID 到名称的映射
    idx_to_predicate = data['idx_to_predicate']  # 谓词 ID 到名称的映射
    
    return idx_to_label, idx_to_predicate

def map_first_image_info(first_image_info, idx_to_label, idx_to_predicate):
    """
    将第一张图片的信息映射为可读的名称。
    
    参数:
        first_image_info: 第一张图片的信息（对象框、类别、关系）。
        idx_to_label: 对象类别 ID 到名称的映射。
        idx_to_predicate: 谓词 ID 到名称的映射。
    
    返回:
        映射后的第一张图片信息。
    """
    # 映射对象类别
    object_class_names = [idx_to_label[str(class_id)] for class_id in first_image_info["classes"]]
    
    # 映射对象关系
    mapped_relationships = []
    for rel in first_image_info["relationships"]:
        subject_index, object_index, predicate_id = rel
        subject_name = object_class_names[subject_index]
        object_name = object_class_names[object_index]
        predicate_name = idx_to_predicate[str(predicate_id)]
        mapped_relationships.append(f"{subject_name} -- {predicate_name} --> {object_name}")
    
    # 返回映射后的信息
    return {
        "boxes": first_image_info["boxes"],
        "classes": object_class_names,
        "relationships": mapped_relationships
    }


# 示例调用
if __name__ == "__main__":
    # 设置数据集路径
    roidb_file = "/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/VG-SGG-with-attri.h5"  # 替换为你的 HDF5 文件路径
    dict_file = "/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/VG-SGG-dicts-with-attri.json"
    # 加载第一张图片的信息
    try:
        first_image_info = load_first_image_info(roidb_file)
        print("第一张图片的原始信息：")
        print("对象框：", first_image_info["boxes"])
        print("对象类别：", first_image_info["classes"])
        print("对象关系：", first_image_info["relationships"])
        
        # 加载映射字典
        idx_to_label, idx_to_predicate = load_mapping_dicts(dict_file)
        
        # 映射第一张图片的信息
        mapped_info = map_first_image_info(first_image_info, idx_to_label, idx_to_predicate)
        print("\n第一张图片的映射信息：")
        print("对象框：", mapped_info["boxes"])
        print("对象类别：", mapped_info["classes"])
        print("对象关系：", mapped_info["relationships"])
    except ValueError as e:
        print(e)

# if __name__ == "__main__":
#     # Set your dataset paths here
#     img_dir = "/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/stanford_spilt/VG_100k_images"  # Replace with your image directory
#     image_file = "/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/image_data.json"  # Replace with your JSON file path

#     # Display the first image info and image
#     display_first_image_info(img_dir, image_file)

#output:
#     First Image Information:
# {
#     "width": 800,
#     "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg",
#     "height": 600,
#     "image_id": 1,
#     "coco_id": null,
#     "flickr_id": null,
#     "anti_prop": 0.0
# }

# Image Size: (800, 600)


import networkx as nx
import matplotlib.pyplot as plt

# 对象节点和关系节点
object_nodes = ['arm', 'boy', 'food', 'hair', 'hand', 'handle', 'plate', 'pole', 'roof', 'shirt', 'sign', 'logo', 'man']
relation_nodes = ['logo -- on --> shirt', 'boy -- has --> arm', 'man -- has --> hand', 'man -- holding --> plate', 'sign -- with --> pole']

# 创建二分图
B = nx.Graph()

# 添加节点
B.add_nodes_from(object_nodes, bipartite=0)  # 对象节点属于集合 0
B.add_nodes_from(relation_nodes, bipartite=1)  # 关系节点属于集合 1

# 添加边（连接对象节点和关系节点）
edges = [
    ('logo', 'logo -- on --> shirt'), ('shirt', 'logo -- on --> shirt'),
    ('boy', 'boy -- has --> arm'), ('arm', 'boy -- has --> arm'),
    ('man', 'man -- has --> hand'), ('hand', 'man -- has --> hand'),
    ('man', 'man -- holding --> plate'), ('plate', 'man -- holding --> plate'),
    ('sign', 'sign -- with --> pole'), ('pole', 'sign -- with --> pole')
]
B.add_edges_from(edges)

# 设置节点位置
pos = {}
pos.update((node, (1, i)) for i, node in enumerate(object_nodes))  # 对象节点在左侧
pos.update((node, (2, i)) for i, node in enumerate(relation_nodes))  # 关系节点在右侧

# 绘制图形
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(B, pos, nodelist=object_nodes, node_color='lightblue', label='Objects')
nx.draw_networkx_nodes(B, pos, nodelist=relation_nodes, node_color='lightgreen', label='Relations')
nx.draw_networkx_edges(B, pos, edge_color='gray')
nx.draw_networkx_labels(B, pos)

# 添加图例和标题
plt.legend(scatterpoints=1)
plt.title("Bipartite Graph: Objects and Relations")
plt.show()