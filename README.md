# Originflow

[English](#english) | [中文](#chinese)

<a name="english"></a>

### **Binder Design Request** ✉️

If you are interested in designing a **binder** using the OriginFlow model, please feel free to contact us via email at **joreyyan@buaa.edu.cn** 📧. To initiate your binder design request, please include the following details:

- **Target PDB file** 🧬
- **Target chain** 🔗
- **Any specific design requirements** (e.g., affinity, structural constraints, or other preferences) 🎯

We will be happy to assist with the binder design process tailored to your needs. Please reach out with the required information, and we will get back to you promptly! 🤝


# Originflow Model Documentation

Code for "Robust and Reliable de novo Protein Design: A Flow-Matching-Based Protein Generative Model Achieves Remarkably High Success Rates"
(https://www.biorxiv.org/content/10.1101/2025.04.29.651154v1)

## Overview

The Originflow model is primarily used for the following protein generation tasks:
- Unconditional protein generation
- Motif-based protein generation
- Secondary structure-specified protein generation
- Symmetric protein generation
- Binder protein generation

## Code and Weights

Weights can be downloaded from: https://drive.google.com/file/d/1saiYp4K0HKeXYzcedB7f_TbB0l4A3iH3/view?usp=sharing

Multiple models have been fine-tuned for different tasks:

- **Monomer**: `motif.ckpt` - We found that weights with light motif masking perform better on unconditional generation tasks
- **Monomer_ss**: `last.ckpt` - Basically the same as the previous weights
- **Sym**: `sym.ckpt` - The above weights should also work, but we didn't use them in our experiments
- **Binder**: `weight/binder/binder.ckpt` - For binder generation

**Note**: Different tasks were trained at different times, and sometimes the network model had slight variations, so the code and weights need to correspond.

## Configuration

The main parameters for each task are controlled by the yaml file.

Most settings don't need to be changed, mainly the weights:
- `ckpt_path`: Model weight path
- `output_dir`: Output directory
- `name`: Project name

## Environment

There are two main environments:

1. **Originflow environment**: Refer to Originflows.yaml
2. **ESM environment**: Refer to the genie environment from Westlake University

## Generation Tasks

### Binder Generation

1. **Data Preprocessing**:
   - Process the structure into a reference.pkl using `data/process_pdb_files.py`
   - This process removes unnecessary information, preserves macromolecular structure, reshapes residueindex, etc.
   - Mainly modify `--pdb_dir`, which will generate a metadata.csv in `--write_dir`
   - Generate a reference.pkl through `prepare_binder_data.py`

2. **Generate binder**:
   - Use `experiments/Originflow_binder.py`
   - Main methods: `sample_binder_bylength_hotspot` or `sample_binder_reference_chains` and others
   - The main difference lies in the establishment of the initial origin point, roughly divided into: using the target protein's centroid as the origin, using reference points as the origin, or adjusting to use the combined centroid of the original target and binder as the origin
   - This step is crucial because when the model initializes the structure, it references the radius, which is a function of sequence length. If the initial position is not well chosen, the generated structure may intersect with the target or be too far away
   - Key parameters:
     - `target`: Target chain index in reference, note that after data processing, this index may be different from the original PDB index. Note this is a complex IDX, not a chain id. Generally, the binder is processed as two COMIDs, target as one, binder as one
     - `length`: Binder sequence length, recommended 50~200
     - `hotspot`: Residue index of key points on the target in reference, note these aren't necessarily all hotspots. The purpose is to provide a centralization position, hoping the generated structure will be centered here. So if the target has a different shape, this can be adjusted
     - `fixed_by_chain`: Fixed chains that don't participate in diffusion, this is actually the target's chain idx, can be one or multiple. For symmetric chains, one is recommended
     - `base_path`: Recommended to be the original pdb path
     - `ref_path`: Address of the generated reference.pkl

3. **Post-processing**:
   - Generated results are diverse, some good, some not so good (mainly due to limited training data and small model size, this could be improved with more training)
   - More importantly, it's a Monte Carlo-like iterative process using ESMFOLD and MPNN
   - ESMFOLD mainly considers not using msa, could be replaced with AF3 (but not tried yet)
   - ESM mainly refers to the genie project's pipeline, in `genie/evaluations/pipeline/evaluate_binder_pdb.py` (modified from original genie)
   - Input the generated sample folder path, specify fixed chain numbers in `pipeline.evaluate`, (A,B,C,) format
   - The reason for not using 1-2-3 format is that in the iterative design process, we might directly refill sequences for previously predicted structures, so ABC format is more convenient
   - In `genie/evaluations/pipeline/pipeline_binder_pdb.py`, we need to adjust which chain's sequence to fill. Generally defaults to the above ABC specified chains, but note that I only wrote mappings for A~E, as we usually don't exceed 3 chains. If more are needed, this can be modified
   - **The code for calling MPNN to generate sequences and using ESMFOLD for structure prediction is located in the `evaluations` folder. After installing the genie environment, you can directly run the relevant code in the `evaluations` folder to perform sequence filling and structure prediction.**
   - Workflow: backbone design → MPNN sequence filling → ESMFOLD structure prediction
   - After running, look at plddt and pAE of the obtained structures. A subfolder will be generated in the target folder, check the info.csv inside
   - Generally, I look at plddt to take the top 10~20, then find structurally different ones, like taking a few alpha, a few beta, preferably looking different
   - For the selected ones, repeat the above MPNN sequence filling-ESMFOLD structure prediction process, meaning only need to run the sequence filling structure generation part, at most 1~2 times, I usually run 2 times
   - Use AF3 for complex prediction of binder and target parts. If target is multi-chain, copy the target chain sequence twice
   - Find those with iplddt > 0.8 in AF3, if none, take the highest, then repeat MPNN sequence filling-ESMFOLD structure prediction process. At this time, may need to fix two chains (if target has two chains)
   - An AF3 result includes 5 structures, can do for all 5 or just one. Generally after 1~3 rounds of AF3, under this structure, there will definitely be binder sequences achieving iplddt>0.8, completing the design

#### Example: VEGF
1. First process the structure, generate pkl
2. Fix one chain best, as these two chains are symmetric, fix one chain, generate 150 reference sequences
3. Fix chain A, redesign sequence for B
4. In info.csv, sort by plddt, take top 10, if top 10 structures are all similar, look at top 20 (ESM's plddt reference value is not significant), generally end up finding 5~10 structurally different ones
5. Copy target A chain once, +binder, send to AF3
6. Redesign sequence for high-scoring AF3 results, at this time fix two chains in ESM

### Unconditional Generation

Use `experiments/Originflow_un.py` to directly generate monomers or multi-chain complexes.

If specifying secondary structure, please refer to this format. If there's a reference pdb, you can use `data/generate_ss.py` to generate the txt reference file.

### Motif Generation and SYM Generation
Please refer to the article's appendix for operation instructions.

### Binder-related PDB Files

All binder PDB files (including structures and sequences) mentioned in the paper have been organized in the `case` folder for direct use.

## Contact and Acknowledgments

If you're interested in this project, feel free to contact me (joreyyan@buaa.edu.cn) for discussion and collaboration.  
Twitter: [@joreyyan](https://twitter.com/joreyyan)  
We support rapid development of binder-related applications, welcome interested friends for collaboration.

Special thanks to the following open-source projects and tools for their support and inspiration:
- [AlphaFold2](https://www.deepmind.com/research/open-source/alphafold)
- [ESMFold](https://github.com/facebookresearch/esm)
- [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- [genie](https://github.com/aqlaboratory/genie)

For those interested in AI + Biology (AI BIO), welcome to contact and exchange ideas for mutual progress!

---

<a name="chinese"></a>
# Originflow 模型文档

Code of "Robust and Reliable de novo Protein Design: A Flow-Matching-Based Protein Generative Model Achieves Remarkably High Success Rates"
(https://www.biorxiv.org/content/10.1101/2025.04.29.651154v1)

## 概述

Originflow 模型主要用于以下蛋白质生成任务：
- 蛋白质的无条件生成
- 基于 Motif 的蛋白质生成
- 指定二级结构的蛋白质生成
- 对称蛋白质生成
- Binder 蛋白质生成

## 代码与权重

权重可以通过：https://drive.google.com/file/d/1saiYp4K0HKeXYzcedB7f_TbB0l4A3iH3/view?usp=sharing
下载

针对不同任务微调了多个模型：

- **Monomer**: `motif.ckpt` - 我们发现轻量 motif 掩码后的权重在无条件生成任务上表现更好
- **Monomer_ss**: `last.ckpt` - 基本与上一个权重一致
- **Sym**: `sym.ckpt` - 应该使用上面的权重也行，只不过我做实验的时候没有用
- **Binder**: `weight/binder/binder.ckpt` - 用于 binder 生成

**请注意**：几个任务分别在不同时期训练，有的时候网络模型有些许变化，所以代码和权重要对应。

## 配置

每一个任务里面的主要参数与文件里面那个 yaml 控制。

基本是大部分不用改，主要改的是权重：
- `ckpt_path`：模型权重路径
- `output_dir`：输出目录
- `name`：项目名称

## 环境

主要有两个环境：

1. **Originflow 环境**：参照 Originflows.yaml
2. **ESM 环境**：参照西湖大学李子青的 genie 那个环境就行

## 生成任务

### Binder 生成

1. **预处理数据**：
   - 将结构处理为一个 reference.pkl，使用 `data/process_pdb_files.py`
   - 这个过程会删除无用信息，保留大分子结构，重塑 residueindex 等等
   - 主要修改其中的 `--pdb_dir`，之后会在 `--write_dir` 生成一个metadata.csv
   - 通过 `prepare_binder_data.py` 将其生成一个 reference.pkl

2. **生成 binder**：
   - 使用 `experiments/Originflow_binder.py`
   - 主要使用方法：`sample_binder_bylength_hotspot` 或 `sample_binder_reference_chains` 以及其他
   - 几种的主要区别在于初始原点的确立，大致上分为：以 target 蛋白的质心作为原点，或者以参考点位作为原点，或者可以调节以原来 target 和 binder 的综合质心为原点
   - 这一步的影响是至关重要的，因为模型初始化结构的时候，是借鉴参考半径的，参考半径是序列长度的函数，如果初始位点选不好，生成结构容易与 target 交错或者拉开过远的距离
   - 关键参数：
     - `target`：在 reference 里面 target 链的索引，注意经过数据处理之后，这里的索引可能和 pdb 原始索引不同，要查看一下。注意这个是复合物 IDX，不是 chain id，一般会把 binder 处理为两个 COMID，target 是一个，binder 是一个
     - `length`：binder 序列的长度，建议 50~200
     - `hotspot`：在 reference 里面 target 上关键点位的 residue index，注意这里不一定全是 hotspot，这里的目的是给定一个中心化位置，希望生成的综合结构以这里为质心，所以如果 target 是不同形状，这里可以调整
     - `fixed_by_chain`：固定的链条，不参与扩散，这个其实就是 target 的 chain idx，可以是一个，也可以是多个，如果是对称链条，建议一个
     - `base_path`：建议为原始 pdb 的路径
     - `ref_path`：生成 reference.pkl 的地址

3. **后处理**：
   - 生成的结果很多样，有好的，有不好的（主要是训练数据少，模型小，其他愿意再扩大训练，这里可以做更高）
   - 更重要的是一个类似蒙特卡洛的迭代过程，我们使用 ESMFOLD 和 MPNN
   - ESMFOLD 主要是考虑不用 msa，其实可以换为 AF3（但还没试过）
   - ESM 主要是参照 genie 项目的 pipeline 来做的，在 `genie/evaluations/pipeline/evaluate_binder_pdb.py`（相对原始 genie 做了修改）
   - 输入生成样本的文件夹路径，在 `pipeline.evaluate` 这里指定固定链条的序号，(A,B,C,) 这种
   - 这里之所以不是 1-2-3 这种，是因为在迭代设计过程中，有可能直接对前次预测出来的结构，再直接再次填充序列，所以接受 ABC 这种更方便
   - 在 `genie/evaluations/pipeline/pipeline_binder_pdb.py` 里面，我们要调整填充哪条链的序列。一般默认上述的 ABC 指定链条，但这里注意一下，我只写了 A~E 的映射，因为一般也不会超过 3 条，如果多了，这里可以改
   - **调用 MPNN 生成序列并使用 ESMFOLD 预测结构的相关代码位于 `evaluations` 文件夹中。安装好 genie 环境后，可以在该环境中直接运行 `evaluations` 文件夹里的相关代码来进行序列填充和结构预测。**
   - 工作流：backbone 设计 → MPNN 填充序列 → ESMFOLD 预测结构
   - 跑完之后，得到的结构，看 plddt 和 pAE，会在目标文件夹下面生成一个子文件夹，看里面的 info.csv 就行
   - 一般我就看 plddt 取前 10~20 个，然后找结构不一样的，比如取几个 alpha 的，取几个 beta 的，最好看起来不一样
   - 针对取出来的几个，重复上述 MPNN 填充序列-ESMFOLD 预测结构过程，也就是只需要跑序列填充结构生成这部分就行了，最多 1~2 次，我一般跑 2 次
   - 将 binder 和 target 的部分使用 AF3 的复合物预测，如果 target 是多链，这里就复制两次 target 链条序列
   - 找到 AF3 里面 iplddt 大于 0.8 的，如果没有，就取最高的，然后重复 MPNN 填充序列-ESMFOLD 预测结构过程，这个时候，可能要固定两条链（如果 target 是两条链的话）
   - 一个 AF3 结果包括 5 个结构，可以针对 5 个都做，也可以针对一个做。一般 1~3 轮 AF3 之后，在该结构下，肯定有能达到 iplddt>0.8 的 binder 序列出来，完成设计

#### 例子：VEGF
1. 首先处理结构，生成 pkl
2. 固定一条链最好，因为这两个链条是对称的，固定一条链，进行生成 150 条参考序列
3. 固定链条 A，进行 B 的序列重新设计
4. info.csv 里面，按照 plddt 排序，取前 10 就行，如果前 10 结构都一个样，就看到前 20（ESM 的 plddt 参考意义不大），一般最后找出来 5~10 个长得不一样的结构就行
5. 把 target A 链复制一遍，+binder，送入 AF3
6. 针对 AF3 高分结果进行重新序列设计，这个时候 ESM 里面固定两条链

### 无条件生成

使用 `experiments/Originflow_un.py` 可以直接生成 monomer，也可以生成多链复合物。

如果指定二级结构，则请参照这种形式来，如果有参照的 pdb，可以使用 `data/generate_ss.py` 来生成该 txt 参考文件。

### Motif生成 SYM 生成
可以参考文章的附件，按照附件操作

### Binder 相关 PDB 文件

论文中提到的所有 binder 的 PDB 文件（包括结构和序列）均已整理在 `case` 文件夹下，可直接使用。

## 联系方式与致谢

如对本项目感兴趣，欢迎联系我（joreyyan@buaa.edu.cn），共同探讨与合作。  
推特（Twitter）：[@joreyyan](https://twitter.com/joreyyan)  
我们支持快速开发 binder 相关应用，欢迎有合作意向的朋友多多交流。

特别感谢以下开源项目和工具对本工作的支持与启发：
- [AlphaFold2](https://www.deepmind.com/research/open-source/alphafold)
- [ESMFold](https://github.com/facebookresearch/esm)
- [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- [genie](https://github.com/aqlaboratory/genie)

对 AI + 生物（AI BIO）方向感兴趣的同仁，欢迎随时联系交流，共同进步！ 
