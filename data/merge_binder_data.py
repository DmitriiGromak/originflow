import pickle

# 定义两个pkl文件的路径
pkl_file1 = '/media/junyu/DATA/pdbbind/reference.pkl'
pkl_file2 = '/home/junyu/project/frame-flow/data/3_2_all_2048_40_cutTrue_cutlength360_clusterFalse_pdbbind_binder_target.pkl'
output_pkl = '/media/junyu/DATA/merged_binder_file.pkl'

# 加载两个pkl文件的内容
with open(pkl_file1, 'rb') as f1:
    data1 = pickle.load(f1)

with open(pkl_file2, 'rb') as f2:
    data2 = pickle.load(f2)

# 合并两个pkl文件的内容
# 如果这两个文件的内容都是字典或列表，你可以直接合并它们
# 这里假设内容是列表
merged_data = data1 + data2

# 将合并后的内容保存为新的pkl文件
with open(output_pkl, 'wb') as f_out:
    pickle.dump(merged_data, f_out)

print(f'Merged pkl file saved as {output_pkl}')
