import json


def modify_json(input_file, output_file):
    # 读取JSON文件
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 遍历每个设计
    for design in data:
        # 获取第二个序列对象的副本
        second_sequence = design["sequences"][1].copy()

        # 在所有VVKFMDV...序列前添加GQNHHE
        prefix = "GQNHHE"
        for seq_obj in design["sequences"][1:]:
            seq_obj["proteinChain"]["sequence"] = prefix + seq_obj["proteinChain"]["sequence"]

        # 添加第三个序列（复制第二个序列）
        design["sequences"].append(second_sequence)

    # 保存修改后的JSON
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


# 使用示例
input_file = "/home/junyu/project/binder_target/1bj1/preprocessed/HOTSPOT_center_begin_chainW/binder_pdbbind384_finetune_ssmask_again_a_by_sample_binder_bylength_hotspot/ESMfold_mini_ca_seq8/designs/design/1bj1_chianw.json"
output_file = "/home/junyu/project/binder_target/1bj1/preprocessed/HOTSPOT_center_begin_chainW/binder_pdbbind384_finetune_ssmask_again_a_by_sample_binder_bylength_hotspot/ESMfold_mini_ca_seq8/designs/design/1bj1_chianw_modified.json"
modify_json(input_file, output_file)
