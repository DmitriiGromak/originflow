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



	# pipeline
	pipeline = Pipeline(None,None)


	# additional information
	info = {}
	if args.motif_filepath:
		info['motif_filepath'] = args.motif_filepath

	# evaluate
	pipeline.evaluate(args.input_dir, args.output_dir, info=info,eva_pair=True)


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



	input_dir='//home/junyu/project/monomer_test/RFdiffusion/rf_sample/'
	output_dir=input_dir

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir',default=input_dir, type=str, help='Input directory', required=False)
	parser.add_argument('--output_dir', default=output_dir,type=str, help='Output directory', required=False)

	parser.add_argument('--motif_filepath',default= None, type=str, help='Motif filepath (for motif scaffolding evaluation)')
#input_dir+f'/native/{short_name}_native.pdb'  input_dir+f'/native/{short_name}_motif_native.pdb',
	args = parser.parse_args()
	main(args,num_samples=1)

#