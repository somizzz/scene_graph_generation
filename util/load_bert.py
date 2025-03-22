# from transformers import AutoTokenizer, AutoModelForMaskedLM
# import os

# # 指定本地保存路径，例如 VSCode 项目文件夹
# save_directory = "/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/stanford_spilt/bert"  # 你可以更改这个路径

# # 下载并保存模型
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

# # 保存模型和 tokenizer
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)

# print(f"Model saved in {os.path.abspath(save_directory)}")


import torch
from transformers import AutoTokenizer, AutoModel

def obj_edge_vectors_bert(names, wv_dir, wv_dim=768):
    """
    用 BERT 生成类别名称的词向量
    :param names: 物体类别名称列表
    :param wv_dir: 预训练 BERT 模型的本地路径
    :param wv_type: BERT 模型类型（默认 'bert-base-uncased'）
    :param wv_dim: 词向量维度（默认 768）
    :return: (len(names), 768) 的张量
    """
    # 1️⃣ 加载 BERT tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(wv_dir)
    model = AutoModel.from_pretrained(wv_dir)
    model.eval()  # 关闭训练模式
    
    # 2️⃣ 创建一个随机初始化的向量（如果 BERT 计算失败，默认使用它）
    vectors = torch.randn(len(names), wv_dim)

    # 3️⃣ 计算 BERT 词向量
    with torch.no_grad():  # 关闭梯度计算
        for i, name in enumerate(names):
            inputs = tokenizer(name, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)

            # 取 [CLS] 作为类别的整体向量
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)

            # 存入 vectors
            vectors[i] = cls_embedding.squeeze(0)

    return vectors  # (len(names), 768)
