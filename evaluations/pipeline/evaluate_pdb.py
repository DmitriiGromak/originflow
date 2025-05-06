import argparse
import glob
import os
import re
from pipeline_pdb import Pipeline
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
	pipeline.evaluate(args.input_dir, args.output_dir, info=info)


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

	for nums in [8]:
	# directories=['4zyp_motif']
		pdb_dir='/home/junyu/project/motif_test/base_neigh/base_rcsbcluster30_fixtopo_motif_1000_fix_update_generateall/2024-06-02_00-02-49/motif/RECHECK_usemotifmodel_find_best_monomer/'


		dirs=os.listdir(pdb_dir)
		# 过滤出所有的文件夹
		directories = [item for item in dirs if os.path.isdir(os.path.join(pdb_dir, item))]


		print(directories)

		for name in directories:
				#for dir in dirs:
				dir=f'{pdb_dir}/{name}/'
				print('deal with  '+dir)
				input_dir=dir+'/'
				output_dir = dir + f'/ESMfold_results_8/'
				parser = argparse.ArgumentParser()
				parser.add_argument('--input_dir',default=input_dir, type=str, help='Input directory', required=False)
				parser.add_argument('--output_dir', default=output_dir,type=str, help='Output directory', required=False)
				short_name=name.split('__')[0]
				print(short_name)
				parser.add_argument('--motif_filepath',default= None, type=str, help='Motif filepath (for motif scaffolding evaluation)')
	#input_dir+f'/native/{short_name}_native.pdb'  input_dir+f'/native/{short_name}_motif_native.pdb',
				args = parser.parse_args()
				main(args,num_samples=nums)

			#