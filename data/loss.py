

import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple, Union
from chroma.layers.structure import backbone, hbonds,  rmsd


class ReconstructionLosses(nn.Module):
    """Compute diffusion reconstruction losses for protein backbones.

    Args:
        diffusion (DiffusionChainCov): Diffusion object parameterizing a
            forwards diffusion over protein backbones.
        loss_scale (float): Length scale parameter used for setting loss error
            scaling in units of Angstroms. Default is 10, which corresponds to
            using units of nanometers.
        rmsd_method (str): Method used for computing RMSD superpositions. Can
            be "symeig" (default) or "power" for power iteration.

    Inputs:
        X0_pred (torch.Tensor): Denoised coordinates with shape
            `(num_batch, num_residues, 4, 3)`.
        X (torch.Tensor): Unperturbed coordinates with shape
            `(num_batch, num_residues, 4, 3)`.
        C (torch.LongTensor): Chain map with shape `(num_batch, num_residues)`.
        t (torch.Tensor): Diffusion time with shape `(batch_size,)`.
            Should be on [0,1].

    Outputs:
        losses (dict): Dictionary of reconstructions computed across different
            metrics. Metrics prefixed with `batch_` will be batch-averaged scalars
            while other metrics should be per batch member with shape
            `(num_batch, ...)`.
    """

    def __init__(
        self,

        loss_scale: float = 10.0,
        rmsd_method: str = "symeig",
    ):
        super().__init__()

        self.loss_scale = loss_scale
        self._loss_eps = 1e-5

        # Auxiliary losses
        self.loss_rmsd = rmsd.BackboneRMSD(method=rmsd_method)
        self.loss_fragment = rmsd.LossFragmentRMSD(method=rmsd_method)
        self.loss_fragment_pair = rmsd.LossFragmentPairRMSD(method=rmsd_method)
        self.loss_neighborhood = rmsd.LossNeighborhoodRMSD(method=rmsd_method)

        self.loss_distance = backbone.LossBackboneResidueDistance()

        self.loss_functions = {

            # "rmsd": self._loss_rmsd,
            # "pseudoelbo": self._loss_pseudoelbo,
            "fragment": self._loss_fragment,
            # "pair": self._loss_pair,
            "neighborhood": self._loss_neighborhood,
            "distance": self._loss_distance,

        }

    def _batch_average(self, loss, C):
        weights = (C > 0).float().sum(-1)
        return (weights * loss).sum() / (weights.sum() + self._loss_eps)

    # def _loss_elbo(self, losses, X0_pred, X, C, t, w=None, X_t_2=None):
    #     losses["elbo"], losses["batch_elbo"] = self.noise_perturb.elbo(X0_pred, X, C, t)

    def _loss_rmsd(self, losses, X0_pred, X, C,loss_mask,  w=None, X_t_2=None):
        _, rmsd_denoise = self.loss_rmsd.align(X, X0_pred, C)

        rmsd_per_item =  rmsd_denoise * w.squeeze(-1)

        losses["global_mse"] = self._batch_average(rmsd_per_item, C)


    def _loss_pseudoelbo(self, losses, X0_pred, X, C, loss_mask, w=None, X_t_2=None):
        # Unaligned residual pseudoELBO
        unaligned_mse = ((X - X0_pred) / self.loss_scale).square().sum(-1).mean(-1)
        losses["batch_pseudoelbo_X"] = unaligned_mse

    def _loss_fragment(self, losses, X0_pred, X, C, loss_mask, w=None, X_t_2=None):
        # Aligned Fragment MSE loss
        mask = (C > 0).float()
        rmsd_fragment = self.loss_fragment(X0_pred, X, C)
        rmsd_fragment= rmsd_fragment * w
        rmsd_fragment=rmsd_fragment*loss_mask

        fragment_mse_normalized = ((
                (mask * rmsd_fragment.square()).sum(1)
                / ((mask ).sum(1) + self._loss_eps)
            )
        )
        losses["fragment_mse"] = fragment_mse_normalized
        losses["batch_fragment_mse"] = self._batch_average(fragment_mse_normalized, C)

    def _loss_pair(self, losses, X0_pred, X, C, loss_mask, w=None, X_t_2=None):
        # Aligned Pair MSE loss
        rmsd_pair, mask_ij_pair = self.loss_fragment_pair(X0_pred, X, C)

        pair_mse_normalized = (
            (
                (mask_ij_pair * rmsd_pair.square()).sum([1, 2])
                / (
                    (mask_ij_pair ).sum([1, 2])
                    + self._loss_eps
                )
            )
        )
        losses["pair_mse"] = pair_mse_normalized
        losses["batch_pair_mse"] = self._batch_average(pair_mse_normalized, C)

    def _loss_neighborhood(self, losses, X0_pred, X, C, loss_mask, w=None, X_t_2=None):
        # Neighborhood MSE
        rmsd_neighborhood, mask = self.loss_neighborhood(X0_pred, X, C)
        rmsd_neighborhood= rmsd_neighborhood * w
        rmsd_neighborhood=rmsd_neighborhood*loss_mask
        neighborhood_mse_normalized = (
             (
                (mask * rmsd_neighborhood.square()).sum(1)
                / ((mask ).sum(1) + self._loss_eps)
            )
        )
        losses["neighborhood_mse"] = neighborhood_mse_normalized
        losses["batch_neighborhood_mse"] = self._batch_average(
            neighborhood_mse_normalized, C
        )



    def _loss_distance(self, losses, X0_pred, X, C,loss_mask,  w=None, X_t_2=None):
        '''
        Distance loss of center of mass between predicted and ground truth
        '''
        num_res=loss_mask.shape[-1]
        # Distance MSE
        mask = (C > 0).float()*loss_mask

        distance_mse = self.loss_distance(X0_pred, X, C)*w

        distance_mse_normalized = self.loss_scale * (
             (mask * distance_mse).sum(1)/ ((mask ).sum(1) + self._loss_eps)
        )


        losses["distance_mse"] = distance_mse_normalized
        losses["batch_distance_mse"] = self._batch_average(distance_mse_normalized, C)







    @torch.no_grad()
    def estimate_metrics(
        self,
        X0_func: Callable,
        X: torch.Tensor,
        C: torch.LongTensor,
        num_samples: int = 50,
        deterministic_seed: int = 0,
        use_noise: bool = True,
        return_samples: bool = False,
        tspan: Tuple[float] = (1e-4, 1.0),
    ):
        """Estimate time-averaged reconstruction losses of protein backbones.

        Args:
            X0_func (Callable): A denoising function that maps `(X, C, t)` to `X0`.
            X (torch.Tensor): A tensor of protein backboone (num) features with shape
                `(batch_size, num_residues, 4, 3)`.
            C (torch.Tensor): A tensor of condition features with shape `(batch_size,
                num_residues)`.
            num_samples (int, optional): The number of time steps to sample for
            estimating the ELBO. Default is 50.
            use_noise (bool): If True, add noise to each structure before denoising.
                Default is True. When False this can be used for estimating if
                if structures are fixed points of the denoiser across time.
            deterministic_seed (int, optional): The seed for generating random noise.
                Default is 0.
            return_samples (bool): If True, include intermediate sampled
                values for each metric. Default is false.
            tspan (tuple[float]): Tuple of floats indicating the diffusion
                times between which to integrate.

        Returns:
            metrics (dict): A dictionary of reconstruction metrics averaged over
                time.
            metrics_samples (dict, optional): A dictionary of in metrics
                averaged over time.
        """
        #
        X = backbone.impute_masked_X(X, C)
        with torch.random.fork_rng():
            torch.random.manual_seed(deterministic_seed)
            T = np.linspace(1e-4, 1.0, num_samples)
            losses = []
            for t in tqdm(T.tolist(), desc="Integrating diffusion metrics"):
                X_noise = self.noise_perturb(X, C, t=t) if use_noise else X
                X_denoise = X0_func(X_noise, C, t)
                losses_t = self.forward(X_denoise, X, C, t)

                # Discard batch estimated objects
                losses_t = {
                    k: v
                    for k, v in losses_t.items()
                    if not k.startswith("batch_") and k != "rmsd_ratio"
                }
                losses.append(losses_t)

            # Transpose list of dicts to a dict of lists
            metrics_samples = {k: [d[k] for d in losses] for k in losses[0].keys()}

            # Average final metrics across time
            metrics = {
                k: torch.stack(v, 0).mean(0)
                for k, v in metrics_samples.items()
                if isinstance(v[0], torch.Tensor)
            }
        if return_samples:
            return metrics, metrics_samples
        else:
            return metrics


    def forward(
        self,
        X0_pred: dict,
        X1: dict,
        Xt: dict,
        C: torch.LongTensor,
        t: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        # Collect all losses and tensors for metric tracking
        losses = {}
        X_t_2 = Xt

        # the /1-t
        norm_scale= 1/(1 - torch.min(
            t, torch.tensor(0.9)))
        self.loss_denom = torch.sum(loss_mask, dim=-1) * 3

        for _loss in self.loss_functions.values():
            _loss(losses, X0_pred, X1, C,loss_mask,w=norm_scale, X_t_2=X_t_2)
        return losses



if __name__ == '__main__':
    rl=ReconstructionLosses().cuda()
    X0_pred=torch.rand([3,10,4,3,]).cuda()
    X1 = torch.rand([3, 10, 4, 3, ]).cuda()
    Xt = torch.rand([3, 10, 4, 3, ]).cuda()
    C=torch.ones([3,10]).cuda()
    mask = torch.ones([3, 10]).cuda()
    t=torch.tensor([0.5,0.6,0.3]).cuda()
    ls=rl(X0_pred,X1,Xt,C,t,mask)