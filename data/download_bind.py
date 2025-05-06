

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

# 下载PDB文件的函数
def download_pdb(pdb_code):
    url = f'https://files.rcsb.org/download/{pdb_code}.pdb'
    response = requests.get(url)
    if response.status_code == 200:
        pdb_path = os.path.join(pdb_folder, f'{pdb_code}.pdb')
        with open(pdb_path, 'wb') as file:
            file.write(response.content)
        return f'Downloaded {pdb_code}'
    else:
        return f'Failed to download {pdb_code}: Status code {response.status_code}'

# CSV文件路径和列名
csv_file_path = '/media/junyu/DATA/pdbbind.csv'
pdb_code_column = 'PDB code'

# 创建文件夹
pdb_folder = '/media/junyu/DATA/pdbbind/'
os.makedirs(pdb_folder, exist_ok=True)

# 读取CSV文件
df = pd.read_csv(csv_file_path)
pdb_codes = df[pdb_code_column].dropna().unique()

# 使用多线程进行下载，并使用tqdm显示进度
with ThreadPoolExecutor(max_workers=20) as executor:
    # 使用futures和tqdm模块跟踪进度
    futures = [executor.submit(download_pdb, code) for code in pdb_codes]
    for future in tqdm(as_completed(futures), total=len(pdb_codes), desc='Downloading PDB files'):
        print(future.result())