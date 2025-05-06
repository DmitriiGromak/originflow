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



def main():



	# pipeline
	pipeline = Pipeline(None,None)

	# additional information
	info = {}

	paths = '/home/junyu/project/binder_target/5o45/pdbfinetunemode/500_steps/binder_pdbbind384_finetune_ssmask_again_a_bylen_pdbbind_/out_0711_ESMfold_mini_ca_seq8/chain_pdbs/'
	# pipeline._compute_chainpdb_scores(pdbs_dir=paths,structures_dir=paths,output_dir=paths+'/output/',verbose=None
	# 					 )
	pipeline._aggregate_tmscores(scores_dir=paths+'/output/scores/',output_dir=paths+'/output/',verbose=None)



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

	main()