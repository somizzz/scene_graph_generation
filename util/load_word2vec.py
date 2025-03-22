import os
import sys
import torch
import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm
import requests 
import gzip
import shutil

import gensim.downloader as api

# 解压 .bin.gz 文件（如果尚未解压）

bin_path = "/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/stanford_spilt/word2vec/GoogleNews-vectors-negative300.bin"


# 加载模型
try:
    word2vec_model = KeyedVectors.load_word2vec_format(bin_path, binary=True)
    print("模型加载成功！")
    print("词汇表大小:", len(word2vec_model.key_to_index))
except Exception as e:
    print(f"加载模型时出错: {e}")



def load_word_vectors(root, wv_type, dim):
    """
    加载 Word2Vec 词向量并转换为 PyTorch 格式。
    :param root: 词向量文件的存储目录。
    :param wv_type: 词向量文件名（不含扩展名）。
    :param dim: 词向量的维度。
    :return: (wv_dict, wv_arr, wv_size)
        - wv_dict: 单词到索引的字典。
        - wv_arr: 词向量张量，形状为 [词汇表大小, dim]。
        - wv_size: 词向量的维度。
    """
    if isinstance(dim, int):
        dim = str(dim) + 'd'

    # 文件路径
    fname_bin = os.path.join(root, wv_type + '.bin')
    fname_txt = os.path.join(root, wv_type + '.txt')
    fname_pt = os.path.join(root, wv_type + '.' + dim + '.pt')

    # 如果有 .pt 直接加载，避免重复解析 .bin
    if os.path.isfile(fname_pt):
        print(f"🔹 直接加载 PyTorch 词向量: {fname_pt}")
        try:
            return torch.load(fname_pt, map_location=torch.device("cpu"))
        except Exception as e:
            raise RuntimeError(f"⚠️ 加载 .pt 失败: {e}")

    # 如果没有 .pt，就加载 .bin 或 .txt
    if os.path.isfile(fname_bin):
        print(f"🔹 加载 Word2Vec .bin 文件: {fname_bin}")
        binary_format = True
        fname = fname_bin
    elif os.path.isfile(fname_txt):
        print(f"🔹 加载 Word2Vec .txt 文件: {fname_txt}")
        binary_format = False
        fname = fname_txt
    else:
        raise RuntimeError(f"找不到 {fname_bin} 或 {fname_txt}")

    try:
        # 加载 Word2Vec 模型
        wv_model = KeyedVectors.load_word2vec_format(fname, binary=binary_format)

        # 转换为 PyTorch 格式
        wv_tokens = list(wv_model.index_to_key)  # 获取单词表
        wv_arr = torch.tensor(wv_model.vectors, dtype=torch.float32)  # 词向量矩阵
        wv_size = wv_model.vector_size  # 词向量维度
        print(f"🔹 加载的词向量维度: {wv_size}")  # 打印词向量维度
        wv_dict = {word: i for i, word in enumerate(wv_tokens)}  # 单词 → 索引映射

        # 保存为 .pt 以加快下次加载
        torch.save((wv_dict, wv_arr, wv_size), fname_pt)
        print(f"词向量转换完成，并保存为 .pt: {fname_pt}")

        return wv_dict, wv_arr, wv_size
    except Exception as e:
        raise RuntimeError(f" 解析 .bin 或 .txt 失败: {e}")


def obj_edge_vectors(names, wv_dir, wv_type='GoogleNews-vectors-negative300', wv_dim=300):
    """
    为给定的物体类别名称加载 Word2Vec 词向量。
    :param names: 物体类别名称列表。
    :param wv_dir: 词向量文件的存储目录。
    :param wv_type: 词向量类型（文件名，不含扩展名）。
    :param wv_dim: 词向量的维度。
    :return: 词向量张量，形状为 [len(names), wv_dim]。
    """
    # 加载词向量
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)
    print(f"🔹 加载的词向量维度: {wv_size}")  # 打印词向量维度

    # 初始化词向量张量
    vectors = torch.randn(len(names), wv_dim, dtype=torch.float32)  # 直接初始化

    # 为每个名称加载词向量
    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # 尝试用名称中最长的单词匹配
            lw_token = max(token.split(' '), key=len)  # 直接取最长单词
            print(f"🔹 '{token}' -> '{lw_token}' (尝试匹配)")

            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print(f" 无法找到 '{token}' 的词向量")

    return vectors

# obj_classes=["cat","dog"]
# word2vec_dir="/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/stanford_spilt/word2vec"

# if __name__== "__main__":
#     obj_embed_vecs = obj_edge_vectors(
#             obj_classes, wv_dir=word2vec_dir, wv_dim=300
#         )