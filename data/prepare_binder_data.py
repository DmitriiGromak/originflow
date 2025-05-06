#!/usr/bin/env python3
import os
import argparse
import yaml
import subprocess
from omegaconf import OmegaConf

def update_yaml_config(pdb_dir):
    """更新原始yaml配置文件中的csv_path"""
    # 读取原始配置文件
    config_path = "/home/junyu/project/frame-flow-main/configs/binder_design.yaml"
    config = OmegaConf.load(config_path)
    
    # 备份原始配置
    original_csv_path = config.data.dataset.csv_path
    
    # 更新csv_path
    config.data.dataset.csv_path = os.path.join(pdb_dir, 'preprocessed/metadata.csv')
    
    # 保存修改后的配置
    OmegaConf.save(config, config_path)
    
    return original_csv_path

def prepare_binder_data(pdb_dir):
    """准备binder设计所需的所有数据"""
    # 确保输入路径是绝对路径
    pdb_dir = os.path.abspath(pdb_dir)
    print(f"处理PDB文件夹: {pdb_dir}")

    # 创建preprocessed目录
    preprocessed_dir = os.path.join(pdb_dir, 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)

    # 1. 运行process_pdb_files.py
    print("Step 1: 处理PDB文件...")
    process_cmd = [
        'python', 
        'process_pdb_files.py',
        '--pdb_dir', pdb_dir
    ]
    subprocess.run(process_cmd, check=True)

    # 2. 更新yaml配置文件
    print("Step 2: 更新配置文件...")
    original_csv_path = update_yaml_config(pdb_dir)

    try:
        # 3. 运行collect_pkl.py
        print("Step 3: 收集数据到pkl文件...")
        collect_cmd = [
            'python',
            'collect_pkl.py'
        ]
        subprocess.run(collect_cmd, check=True)

        print(f"数据处理完成！")
        print(f"生成的文件位于: {preprocessed_dir}")
    
    finally:
        # 恢复原始配置
        config = OmegaConf.load("/home/junyu/project/frame-flow-main/configs/binder_design.yaml")
        config.data.dataset.csv_path = original_csv_path
        OmegaConf.save(config, "/home/junyu/project/frame-flow-main/configs/binder_design.yaml")

def main():
    parser = argparse.ArgumentParser(description='准备binder设计所需的数据')
    parser.add_argument(
        '--pdb_dir',
        help='包含PDB文件的目录路径',
        required=True
    )
    args = parser.parse_args()

    prepare_binder_data(args.pdb_dir)

if __name__ == '__main__':
    main()
