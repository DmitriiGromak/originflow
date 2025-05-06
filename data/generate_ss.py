
import argparse
import dataclasses
import functools as fn
import glob

import pandas as pd
import os
import multiprocessing as mp
import time
from Bio import PDB
import numpy as np
import mdtraj as md
# 创建一个映射字典来将字符转换为数字
mapping_dict = {'NA': 0, 'C': 1, 'E': 2, 'H': 3}
# 使用 numpy 的 vectorize 方法来应用这个映射
vectorized_mapping = np.vectorize(mapping_dict.get)

file_path='/home/junyu/project/monomer_test/ss/alphabeta/'

pdbs = glob.glob(file_path + '*.pdb')

# 打开一个文件用于写入结果
with open('alphabeta.txt', 'w') as f:
    for pdb in pdbs:
        traj = md.load(pdb)
        # 计算二级结构
        pdb_ss = md.compute_dssp(traj, simplified=True)
        pdb_ss = vectorized_mapping(pdb_ss).tolist()[0]
        # 获取文件名
        pdb_name = pdb.split('/')[-1].split('.')[0]
        # 将结果写入文件
        f.write(f"{pdb_name} {pdb_ss}\n")

print("结果已写入 beta.txt 文件中")