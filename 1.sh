#!/bin/bash
#SBATCH -A p_zhu                 # 指定项目名称
#SBATCH --partition=gpu              # 选择有效的分区
#SBATCH --gres=gpu:1               # 请求 1 个 GPU
#SBATCH --job-name=test_pysgg  # 作业名称
#SBATCH --time=03:00:00              # 设置运行时间
#SBATCH --mail-type=ALL              # 邮件通知类型
#SBATCH --mail-user=drgck8@inf.elte.hu  # 邮件地址
#SBATCH --mem-per-gpu=50000
#SBATCH --nodes=1
 

 
# 执行训练脚本
/home/p_zhuzy/miniconda3/envs/pysgg/bin/python -u /home/p_zhuzy/p_zhu/PySGG-main/train.py