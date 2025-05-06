from typing import Any
import torch
import torch.nn.functional as F
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from analysis import metrics 
from analysis import utils as au
from models.flow_model import FlowModel,FlowModel_binder,FlowModel_binder_sidechain,FlowModel_seqdesign
from models import utils as mu
from data.interpolant import Interpolant_10
from data import utils as du
from data.motif_sample import MotifSampler
from models.noise_schedule import OTNoiseSchedule




def generate_random_list(total_length):
    while True:
        num_parts = random.randint(2, 6)  # Choose a random number of parts between 2 and 8
        parts = []
        remaining = total_length

        for i in range(num_parts - 1):
            if remaining > 10 * (num_parts - i):
                part = random.randint(10, remaining - 10 * (num_parts - i - 1))
            else:
                break
            parts.append(part)
            remaining -= part

        # Add the last part
        if remaining >= 10:
            parts.append(remaining)
            return parts
        # If the last part isn't valid, loop again


class FlowModule(LightningModule):

    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)

        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant
        self._eps=1e-6



        self.design_task = None


        if 'corrupt_mode' not in self._exp_cfg:
            print('\n  old version no corrupt mode')
            self.model = FlowModel(cfg.model,mode='base')
            # self._exp_cfg.corrupt_mode='base'
        # Set-up vector field prediction model
        else:
            if self._exp_cfg.corrupt_mode == 'binder':
                self.model = FlowModel_binder(cfg.model)
            elif self._exp_cfg.corrupt_mode == 'motif':
                self.model = FlowModel(cfg.model)

            elif self._exp_cfg.corrupt_mode == 'sidechain':
                # self.pretrained_part  = FlowModel_binder(cfg.model)
                self.model = FlowModel_binder_sidechain(cfg.model)
            elif self._exp_cfg.corrupt_mode == 'base':   # base for mononer or complex
                self.model = FlowModel(cfg.model,mode='base')

            elif self._exp_cfg.corrupt_mode == 'base_ss':   # base for mononer or complex
                self.model = FlowModel(cfg.model,mode='base_ss')


            elif self._exp_cfg.corrupt_mode == 'fbb':   # base for mononer or complex
                self.model = FlowModel_seqdesign(cfg.model)




        # Set-up interpolant
        self.interpolant = Interpolant_10(cfg.interpolant)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        # Noise schedule
        self.noise_schedule=OTNoiseSchedule()



    def predict_step_FUNC(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'
        num_batch=2

        self.interpolant._sample_cfg.num_timesteps = 500

        self._sample_write_dir = '/home/junyu/project/monomer_test/homo_heto/monomder_ss/' + str(
            self.interpolant._sample_cfg.num_timesteps) + '/sample/'
        os.makedirs(self._sample_write_dir, exist_ok=True)

        from chroma.data.protein import Protein
        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()

        sample_id = 0
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return

        sample_length = batch['num_res'].item()



        sample_id = batch['sample_id'].item()
        self.interpolant.set_device(device)
        res_mask=torch.ones((sample_id,sample_length),device=device)
        if self._exp_cfg.corrupt_mode == 'motif':
            samples = self.val_complex( num_batch, sample_length,self._exp_cfg.corrupt_mode)
            # samples, fixed_mask = self.val_motif(batch, sample_id, sample_length, res_mask)
            X = samples[0]
            C = samples[1]
            S = samples[2]
        elif self._exp_cfg.corrupt_mode == 'binder':
            samples = self.val_complex(num_batch, sample_length,self._exp_cfg.corrupt_mode)
            #samples, noisy_batch = self.val_binder(batch, num_batch, num_res, res_mask)
            X = samples[0]
            C = samples[1]
            S = samples[2]
            bf = samples[3]




        for i in range(num_batch):
            if self._exp_cfg.corrupt_mode == 'binder':

                p = Protein.from_XCSB(X, C, S, bf)
                p.to_PDB(os.path.join(
                    self._sample_write_dir,
                    f'sample_{i}_idx_{batch_idx}_len_{sample_length}.pdb'))


            else:
                p = Protein.from_XCS(X[i].unsqueeze(0), C[i].unsqueeze(0), S[i].unsqueeze(0))
                p.to_PDB(os.path.join(
                    self._sample_write_dir,
                    f'sample_{i}_idx_{batch_idx}_len_{sample_length}.pdb'))

    def predict_step(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'
        num_batch = 1
        sample_length = batch['num_res'].item()

        sample_id = batch['sample_id'].item()
        self.interpolant.set_device(device)

        if self.design_task == 'sym':
            self.predict_step_sym(batch, batch_idx)
        elif self.design_task=='monomer':
            if self._exp_cfg.corrupt_mode == 'base':
                self.predict_step_base(batch, batch_idx)
            elif  self._exp_cfg.corrupt_mode == 'base_ss':
                print('corrupt mode is base_ss')
                self.predict_step_base(batch, batch_idx)
            elif  self._exp_cfg.corrupt_mode == 'motif':
                print('corrupt mode is motif')
                self.predict_step_base(batch, batch_idx)

            else:
                print('corrupt mode not defined')

        elif self.design_task=='homomer':

            self.predict_step_homo(batch, batch_idx)

        elif self.design_task=='motif':

            self.predict_step_motif(batch, batch_idx)
        elif self.design_task=='monomer_ss':

            self.predict_step_base_ss(batch, batch_idx)
        elif self.design_task=='base_ss':
            self.predict_step_base_ss(batch, batch_idx)
        else:
            self.predict_step_FUNC(batch, batch_idx)

            samples = self.val_complex(num_batch, sample_length, self._exp_cfg.corrupt_mode)

    def predict_step_base(self, batch, batch_idx):

        print('design for  , ',self.design_task)

        methods=self._infer_cfg.interpolant.sampling.methods
        num_timesteps=self._infer_cfg.interpolant.sampling.num_timesteps

        self.interpolant._sample_cfg.num_timesteps=num_timesteps
        print(f'design case in, {num_timesteps} steps', )

        self._sample_write_dir=(self._output_dir+f'/{self.design_task}_{methods}_temp'+str(self.interpolant._cfg.temp)+'_'
                                +str(num_timesteps)+'_0116/')


        os.makedirs(self._sample_write_dir, exist_ok=True)
        print('write in ,', self._sample_write_dir)


        from chroma.data.protein import Protein
        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()

        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return

        num_batch = 1



        samples= interpolant.sample(
            num_batch, sample_length, self.model
        )
        X = samples[0]
        C = samples[1]
        S = samples[2]

        for i in range(num_batch):
            X_i = X[i].unsqueeze(0)
            C_i = C[i].unsqueeze(0)
            S_i = S[i].unsqueeze(0)

            p = Protein.from_XCS(X_i, C_i, S_i)
            p.to_PDB(os.path.join(
                self._sample_write_dir,
                f'sample_{i}_idx_{batch_idx}_len_{sample_length}.pdb'))


    def predict_step_base_ss(self, batch, batch_idx):
        '''
        design with ss
        '''

        print('design for base , ',self.design_task)
        self.interpolant._sample_cfg.num_timesteps=500
        methods='cvode_ss'

        pdb_name=self._cfg.pdb_name
        ss_list_str=self._cfg.ss_list_str

        print('design: ', pdb_name,ss_list_str)

        self._sample_write_dir=self._output_dir+f'/32_{methods}_{pdb_name}_temp'+str(self.interpolant._cfg.temp)+'_'+str(self.interpolant._sample_cfg.num_timesteps)+'/'
        os.makedirs(self._sample_write_dir, exist_ok=True)
        print('write in ,', self._sample_write_dir)


        from chroma.data.protein import Protein
        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()

        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return

        num_batch = 1
        ss= eval(ss_list_str)


        # 随机生成长度在60到200之间


        if         methods=='cvode':
            samples= interpolant.sample(
                num_batch, sample_length, self.model
            )
        else:

            samples = self.interpolant.hybrid_Complex_sample(
                num_batch,
                [len(ss)],
                self.model,
                ss,
            )

        X = samples[0]
        C = samples[1]
        S = samples[2]

        for i in range(num_batch):
            X_i = X[i].unsqueeze(0)
            C_i = C[i].unsqueeze(0)
            S_i = S[i].unsqueeze(0)

            p = Protein.from_XCS(X_i, C_i, S_i)
            p.to_PDB(os.path.join(
                self._sample_write_dir,
                f'sample_{i}_idx_{batch_idx}_len_{sample_length}.pdb'))

    def predict_step_homo(self, batch, batch_idx):

        print('design for  , ', self.design_task)

        methods = self._infer_cfg.interpolant.sampling.methods
        num_timesteps = self._infer_cfg.interpolant.sampling.num_timesteps

        self.interpolant._sample_cfg.num_timesteps = num_timesteps
        print(f'design case in, {num_timesteps} steps', )

        self._sample_write_dir = (self._output_dir + f'/32_{methods}_temp' + str(self.interpolant._cfg.temp) + '_'
                                  + str(num_timesteps) + '/')

        os.makedirs(self._sample_write_dir, exist_ok=True)
        print('write in ,', self._sample_write_dir)






        from chroma.data.protein import Protein
        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()

        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return

        num_batch = 1

        numres=generate_random_list(sample_length)

        samples = self.interpolant.hybrid_Complex_sample(
            num_batch,
            numres,
            self.model,
        )

        X = samples[0]
        C = samples[1]
        S = samples[2]

        for i in range(num_batch):
            X_i = X[i].unsqueeze(0)
            C_i = C[i].unsqueeze(0)
            S_i = S[i].unsqueeze(0)

            p = Protein.from_XCS(X_i, C_i, S_i)
            p.to_PDB(os.path.join(
                self._sample_write_dir,
                f'sample_{i}_idx_{batch_idx}_len_{sample_length}.pdb'))
    def predict_step_sym(self, batch, batch_idx):
        print(batch_idx, 'sym')


        self.interpolant._sample_cfg.num_timesteps=500
        methods='cvode_sym'

        sym_mode=self._cfg.inference.sym
        print('predict_step_sym: ', sym_mode)
        self._sample_write_dir=self._output_dir+f'/sym_{sym_mode}_{methods}_try_temp'+str(self.interpolant._cfg.temp)+'_'+str(self.interpolant._sample_cfg.num_timesteps)+'/'
        os.makedirs(self._sample_write_dir, exist_ok=True)



        from chroma.data.protein import Protein
        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()
        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return

        num_batch = 1



        samples = self.interpolant.hybrid_Complex_sym_sample(
            num_batch,
            [sample_length],
            self.model,
            symmetry=sym_mode,
        )

        X = samples[0]
        C = samples[1]
        S = samples[2]

        for i in range(num_batch):
            X_i = X[i].unsqueeze(0)
            C_i = C[i].unsqueeze(0)
            S_i = S[i].unsqueeze(0)

            p = Protein.from_XCS(X_i, C_i, S_i)
            p.to_PDB(os.path.join(
                self._sample_write_dir,
                f'sample_{i}_idx_{batch_idx}_len_{sample_length}.pdb'))
    def predict_step_motif(self, batch, batch_idx):

        print('design for  , ',self.design_task)

        methods=self._infer_cfg.interpolant.sampling.methods
        num_timesteps=self._infer_cfg.interpolant.sampling.num_timesteps

        self.interpolant._sample_cfg.num_timesteps=num_timesteps
        print(f'design case in, {num_timesteps} steps', )


        print(self._output_dir)


        ref_pdb = '/home/junyu/project/motif/rf_pdb_pkl/5tpn.pkl'
        domain=ref_pdb.split('/')[-1].split('.')[0]
        ref_data = du.read_pkl(ref_pdb)
        input_str="10-40,A163-181,10-40"

        # 使用当前时间戳设置种子值
        # random.seed(time.time())
        total_length=random.randint(50, 75)



        sampler = MotifSampler(input_str, total_length)
        results = sampler.get_results()
        print(f"Letter segments: {results['letter_segments']}")
        print(f"Number segments: {results['number_segments']}")
        print(f"Total motif length: {results['total_motif_length']}")
        print(f"Random sample total length: {results['random_sample_total_length']}")
        print(f"Sampled lengths: {results['sampled_lengths']}")
        final_output = sampler.get_final_output()
        print(f"Final output: {final_output}")




        designname=ref_pdb.split('/')[-1].split('.')[0]
        self._sample_write_dir=self._output_dir+f'/trysampleupdate_{designname}_{methods}_temp'+str(self.interpolant._cfg.temp)+'_'+str(self.interpolant._sample_cfg.num_timesteps)+'/'

        #self._sample_write_dir='/home/junyu/project/monomer_test/base_neigh/rcsb_ipagnnneigh_cluster_motif_updateall_reeidx_homo_heto/2024-03-12_12-52-05/last_256/rcsb/motif_1bcf/'
        os.makedirs(self._sample_write_dir, exist_ok=True)
        os.makedirs(self._sample_write_dir+'/native/', exist_ok=True)
        os.makedirs(self._sample_write_dir+'/motif_masks/', exist_ok=True)


        from .Proflow import process_input
        from chroma.data.protein import Protein
        import tqdm

        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        # native structure
        positions = np.take(ref_data['atom_positions'], ref_data['modeled_idx'], axis=0)[..., [0, 1, 2, 4], :]
        chain_index = np.take(ref_data['chain_index'], ref_data['modeled_idx'], axis=0)
        aatype = np.take(ref_data['aatype'], ref_data['modeled_idx'], axis=0)
        p = Protein.from_XCS(torch.tensor(positions).unsqueeze(0), torch.tensor(chain_index).unsqueeze(0),
                             torch.tensor(aatype).unsqueeze(0))
        p.to_PDB(self._sample_write_dir + f'/native/{domain}_motif_native.pdb')

        _, _, _, indices_mask = process_input(final_output, ref_data)
        np.savetxt(self._sample_write_dir + f'/motif_masks/motif_native.npy', indices_mask)

        for i in tqdm.tqdm(range(10)):
            total_length = random.randint(50, 75)
            sampler = MotifSampler(input_str, total_length)
            results = sampler.get_results()
            print(f"Letter segments: {results['letter_segments']}")
            print(f"Number segments: {results['number_segments']}")
            print(f"Total motif length: {results['total_motif_length']}")
            print(f"Random sample total length: {results['random_sample_total_length']}")
            print(f"Sampled lengths: {results['sampled_lengths']}")
            final_output = sampler.get_final_output()
            print(f"Final output: {final_output}")

            init_motif, fixed_mask, aa_motifed, indices_mask = process_input(final_output, ref_data)



            chain_idx = torch.ones_like(fixed_mask).unsqueeze(0)
            fixed_mask = fixed_mask.unsqueeze(0)
            bbatoms = torch.tensor(init_motif)[..., [0, 1, 2, 4], :].unsqueeze(0).to(self.device).float()
            sample_length = bbatoms.shape[1]



            X, C, S = self.interpolant.try_motif_sample(
                1,
                [sample_length],
                self.model,
                chain_idx=chain_idx.to(self.device),
                native_X=bbatoms.to(self.device),
                mode='motif',
                fixed_mask=fixed_mask.to(self.device),

            )

            native_aatype=torch.tensor(aa_motifed).unsqueeze(0).to(S.device)
            S=native_aatype*fixed_mask.to(S.device)
            p = Protein.from_XCS(X, C, S, )


            # 输出到文本文件
            p.to_PDB(self._sample_write_dir+f'/{domain}_motif' + str(i) + '.pdb')
            np.savetxt(self._sample_write_dir + f'/motif_masks/{domain}_motif' + str(i) +'_mask.npy', fixed_mask.cpu().numpy())

            # positions = torch.nonzero(fixed_mask.squeeze(0) == 1).squeeze(-1)
            # # 转换为列表
            # positions_list = positions.tolist()
            # # 转换为逗号分隔的字符串
            # positions_str = ', '.join(map(str, positions_list))
            # with open(self._sample_write_dir + '/5yui_motif_info.txt', 'w') as f:
            #
            #         f.write(f"> 5yui_motif, fixed area \n")
            #         f.write(f"> in native \n")
            #         f.write(f"> {parama}  \n")
            #         f.write(f"> in design \n")
            #         f.write(f"{positions_str}\n")


    def predict_step_motif_fixlength(self, batch, batch_idx):

        print('design for  , ',self.design_task)

        methods=self._infer_cfg.interpolant.sampling.methods
        num_timesteps=self._infer_cfg.interpolant.sampling.num_timesteps

        self.interpolant._sample_cfg.num_timesteps=num_timesteps
        print(f'design case in, {num_timesteps} steps', )


        print(self._output_dir)


        ref_pdb = '/home/junyu/project/motif/rf_pdb_pkl/5tpn.pkl'
        domain=ref_pdb.split('/')[-1].split('.')[0]
        ref_data = du.read_pkl(ref_pdb)
        input_str="10-40,A163-181,10-40"

        # 使用当前时间戳设置种子值
        # random.seed(time.time())
        total_length=random.randint(50, 75)



        sampler = MotifSampler(input_str, total_length)
        results = sampler.get_results()
        print(f"Letter segments: {results['letter_segments']}")
        print(f"Number segments: {results['number_segments']}")
        print(f"Total motif length: {results['total_motif_length']}")
        print(f"Random sample total length: {results['random_sample_total_length']}")
        print(f"Sampled lengths: {results['sampled_lengths']}")
        final_output = sampler.get_final_output()
        print(f"Final output: {final_output}")




        designname=ref_pdb.split('/')[-1].split('.')[0]
        self._sample_write_dir=self._output_dir+f'/motif_motifdesign_{designname}_{methods}_temp'+str(self.interpolant._cfg.temp)+'_'+str(self.interpolant._sample_cfg.num_timesteps)+'/'

        #self._sample_write_dir='/home/junyu/project/monomer_test/base_neigh/rcsb_ipagnnneigh_cluster_motif_updateall_reeidx_homo_heto/2024-03-12_12-52-05/last_256/rcsb/motif_1bcf/'
        os.makedirs(self._sample_write_dir, exist_ok=True)
        os.makedirs(self._sample_write_dir+'/native/', exist_ok=True)
        os.makedirs(self._sample_write_dir+'/motif_masks/', exist_ok=True)


        from .Proflow import process_input
        from chroma.data.protein import Protein
        import tqdm

        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        # native structure
        positions = np.take(ref_data['atom_positions'], ref_data['modeled_idx'], axis=0)[..., [0, 1, 2, 4], :]
        chain_index = np.take(ref_data['chain_index'], ref_data['modeled_idx'], axis=0)
        aatype = np.take(ref_data['aatype'], ref_data['modeled_idx'], axis=0)
        p = Protein.from_XCS(torch.tensor(positions).unsqueeze(0), torch.tensor(chain_index).unsqueeze(0),
                             torch.tensor(aatype).unsqueeze(0))
        p.to_PDB(self._sample_write_dir + f'/native/{domain}_motif_native.pdb')

        _, _, _, indices_mask = process_input(final_output, ref_data)
        np.savetxt(self._sample_write_dir + f'/motif_masks/motif_native.npy', indices_mask)

        for i in tqdm.tqdm(range(10)):
            total_length = random.randint(50, 75)
            sampler = MotifSampler(input_str, total_length)
            results = sampler.get_results()
            print(f"Letter segments: {results['letter_segments']}")
            print(f"Number segments: {results['number_segments']}")
            print(f"Total motif length: {results['total_motif_length']}")
            print(f"Random sample total length: {results['random_sample_total_length']}")
            print(f"Sampled lengths: {results['sampled_lengths']}")
            final_output = sampler.get_final_output()
            print(f"Final output: {final_output}")

            init_motif, fixed_mask, aa_motifed, indices_mask = process_input(final_output, ref_data)



            chain_idx = torch.ones_like(fixed_mask).unsqueeze(0)
            fixed_mask = fixed_mask.unsqueeze(0)
            bbatoms = torch.tensor(init_motif)[..., [0, 1, 2, 4], :].unsqueeze(0).to(self.device).float()
            sample_length = bbatoms.shape[1]



            X, C, S = self.interpolant.hybrid_motif_sample(
                1,
                [sample_length],
                self.model,
                chain_idx=chain_idx.to(self.device),
                native_X=bbatoms.to(self.device),
                mode='motif',
                fixed_mask=fixed_mask.to(self.device),

            )

            native_aatype=torch.tensor(aa_motifed).unsqueeze(0).to(S.device)
            S=native_aatype*fixed_mask.to(S.device)
            p = Protein.from_XCS(X, C, S, )


            # 输出到文本文件
            p.to_PDB(self._sample_write_dir+f'/{domain}_motif' + str(i) + '.pdb')
            np.savetxt(self._sample_write_dir + f'/motif_masks/{domain}_motif' + str(i) +'.npy', fixed_mask.cpu().numpy())

            # positions = torch.nonzero(fixed_mask.squeeze(0) == 1).squeeze(-1)
            # # 转换为列表
            # positions_list = positions.tolist()
            # # 转换为逗号分隔的字符串
            # positions_str = ', '.join(map(str, positions_list))
            # with open(self._sample_write_dir + '/5yui_motif_info.txt', 'w') as f:
            #
            #         f.write(f"> 5yui_motif, fixed area \n")
            #         f.write(f"> in native \n")
            #         f.write(f"> {parama}  \n")
            #         f.write(f"> in design \n")
            #         f.write(f"{positions_str}\n")
