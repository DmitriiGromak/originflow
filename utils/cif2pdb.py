import os
from Bio.PDB import PDBIO, MMCIFParser
import glob
def convert_cif_to_pdb(mmcif_path):
    p = MMCIFParser()
    struc = p.get_structure("", mmcif_path)
    io = PDBIO()
    io.set_structure(struc)
    pdb_path = mmcif_path.replace('.cif', '.pdb')
    io.save(pdb_path)

if __name__ == "__main__":
    input_directory = '/home/junyu/project/binder_target/1bj1/preprocessed/finial_design/前4轮好的/'  # 替换为CIF文件所在目录的路径

    alldir=os.listdir(input_directory)
    for dir in alldir:

        cifs=glob.glob(input_directory+dir+'/**.cif')
        for i in cifs:

            convert_cif_to_pdb(i)
