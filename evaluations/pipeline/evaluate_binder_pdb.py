import argparse
import glob
import os
import re
from pipeline_binder_pdb import Pipeline
from inverse_fold_models.proteinmpnn import ProteinMPNN
from fold_models.esmfold import ESMFold
import shutil
import torch
import esm



def main(args,num_samples):

	# inverse fold model
	inverse_fold_model = ProteinMPNN(num_samples=num_samples)
	#
	# # # fold model
	fold_model = ESMFold()

	# pipeline
	#pipeline = Pipeline(None,None)
	pipeline = Pipeline(inverse_fold_model,fold_model)

	# additional information
	info = {}
	if args.motif_filepath:
		info['motif_filepath'] = args.motif_filepath

	# evaluate
	pipeline.evaluate(args.input_dir, args.output_dir,fixed_chains=['B','C'], info=info,redesign_partaa=True)


if __name__ == '__main__':
	# pdbs_dir='/home/junyu/project/monomer_test/basemode/hybrid_sample_step500/sample/'
	# pdbs=glob.glob(os.path.join(pdbs_dir, '**_0_**.pdb'))
	# dest_dir='/home/junyu/project/monomer_test/basemode/hybrid_sample_step500/sample_mini/'
	# for i in pdbs:
	# 	pdb_name = i.split('/')[-1]
	# 	lens=int(i.split('/')[-1].split('.')[0].split('_')[-1])
	# 	if lens%20==0:
	# 		source_pdb = i
	# 		destination_pdb = os.path.join(dest_dir, pdb_name)
	# 		shutil.copy(source_pdb, destination_pdb)

	# dirs=os.listdir('//home/junyu/project/monomer_test/RFdiffusion/motif/')

	#names=['4jhw']  #'5trv','6e6r','6exz'

	for nums in [50]:
	# directories=['4zyp_motif']
		pdb_dir='/home/junyu/project/binder_target/1bj1/preprocessed/finial_design/MD_redesign//'
		dirs=os.listdir(pdb_dir)
		# 过滤出所有的文件夹
		directories = [item for item in dirs if os.path.isdir(os.path.join(pdb_dir, item))]

		for name in directories:
				#for dir in dirs:
				dir=f'{pdb_dir}/{name}/'
				print('deal with  '+dir)
				input_dir=dir+'/'
				output_dir = dir + f'/ESMfoldmini_ca_seq{nums}/'
				parser = argparse.ArgumentParser()
				parser.add_argument('--input_dir',default=input_dir, type=str, help='Input directory', required=False)
				parser.add_argument('--output_dir', default=output_dir,type=str, help='Output directory', required=False)
				short_name=name.split('__')[0]
				parser.add_argument('--motif_filepath',default= None, type=str, help='Motif filepath (for motif scaffolding evaluation)')

				args = parser.parse_args()
				main(args,num_samples=nums)

