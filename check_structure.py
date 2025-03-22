import h5py

def print_hdf5_structure(file, indent=0):
    """
    递归打印 HDF5 文件的结构。
    
    参数:
        file: HDF5 文件或组。
        indent: 缩进级别（用于格式化输出）。
    """
    for key in file.keys():
        item = file[key]
        print(" " * indent + f"├── {key}")  # 打印当前项的名称
        if isinstance(item, h5py.Group):
            # 如果当前项是一个组，递归打印其内容
            print_hdf5_structure(item, indent + 4)
        elif isinstance(item, h5py.Dataset):
            # 如果当前项是一个数据集，打印其形状和数据类型
            print(" " * (indent + 4) + f"├── shape: {item.shape}")
            print(" " * (indent + 4) + f"└── dtype: {item.dtype}")

# 打开 HDF5 文件
file_path = "/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/VG-SGG-with-attri.h5"
with h5py.File(file_path, 'r') as f:
    print("HDF5 文件结构：")
    print_hdf5_structure(f)

import json

# 假设 dict_file 是 JSON 文件
dict_file = "/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/VG-SGG-dicts-with-attri.json"

# 加载 JSON 文件
with open(dict_file, 'r') as f:
    data = json.load(f)

# 打印字典的键和对应的值类型
print("字典键及其值类型：")
for key, value in data.items():
    print(f"键: {key}, 值类型: {type(value)}")

