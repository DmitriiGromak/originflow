import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from tqdm.auto import tqdm
from chroma.layers.structure import backbone, hbonds, mvn, rmsd
class OTNoiseSchedule:
    """
    A general noise schedule for the General Gaussian Forward Path, where noise is added
    to the input signal.

    The noise is modeled as Gaussian noise with mean `alpha_t x_0` and variance
     `sigma_t^2`, with 'x_0 ~ p(x_0)' The time range of the noise schedule is
     parameterized with a user-specified logarithmic signal-to-noise ratio (SNR) range,
    where  `snr_t = alpha_t^2 / sigma_t^2` is the SNR at time `t`.

    In addition, the object defines a quantity called the scaled signal-to-noise ratio
    (`ssnr_t`), which is given by `ssnr_t = alpha_t^2 / (alpha_t^2 + sigma_t^2)`
    and is a helpful quantity for analyzing the performance of signal processing
    algorithms under different noise conditions.

    This object implements a few standard noise schedule:

        'log_snr': variance-preserving process with linear log SNR schedule
        (https://arxiv.org/abs/2107.00630)

        'ot_linear': OT schedule
        (https://arxiv.org/abs/2210.02747)

        've_log_snr': variance-exploding process with linear log SNR s hedule
        (https://arxiv.org/abs/2011.13456 with log SNR noise schedule)

    User can also implement their own schedules by specifying alpha_func, sigma_func
    and compute_t_range.

    """

    def __init__(
        self, log_snr_range: Tuple[float, float] = (-7.0, 13.5), kind: str = "ot_linear",
    ) -> None:
        super().__init__()

        if kind not in ["log_snr", "ot_linear", "ve_log_snr"]:
            raise NotImplementedError(
                f"noise type {kind} is not implemented,                            only"
                " log_snr and ot_linear are supported "
            )
        self.kind = kind
        self.log_snr_range = log_snr_range

        l_min, l_max = self.log_snr_range

        # map t \in [0, 1] to match the prescribed log_snr range
        self.t_max = self.compute_t_range(l_min)
        self.t_min = self.compute_t_range(l_max)
        self._eps = 1e-5

    def t_map(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """map t in [0, 1] to [t_min, t_max]

        Args:
            t (Union[float, torch.Tensor]): time

        Returns:
            torch.Tensor: mapped time
        """
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float()

        t_max = self.t_max.to(t.device)
        t_min = self.t_min.to(t.device)
        t_tilde = t_min + (t_max - t_min) * t

        return t_tilde

    def derivative(self, t: torch.Tensor, func: Callable) -> torch.Tensor:
        """compute derivative of a function, it supports bached single variable inputs

        Args:
            t (torch.Tensor): time variable at which derivatives are taken
            func (Callable): function for derivative calculation

        Returns:
            torch.Tensor: derivative that is detached from the computational graph
        """
        with torch.enable_grad():
            t.requires_grad_(True)
            derivative = grad(func(t).sum(), t, create_graph=False)[0].detach()
            t.requires_grad_(False)
        return derivative

    def tensor_check(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """convert input to torch.Tensor if it is a float

        Args:
            t ( Union[float, torch.Tensor]): input

        Returns:
            torch.Tensor: converted torch.Tensor
        """
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float()
        return t

    def alpha_func(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """alpha function that scales the mean, usually goes from 1. to 0.

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: alpha value
        """

        t = self.tensor_check(t)

        if self.kind == "log_snr":
            l_min, l_max = self.log_snr_range
            t_min, t_max = self.t_min, self.t_max
            log_snr = (1 - t) * l_max + t * l_min

            log_alpha = 0.5 * (log_snr - F.softplus(log_snr))
            alpha = log_alpha.exp()
            return alpha

        elif self.kind == "ve_log_snr":
            return 1 - torch.relu(-t)  # make this differentiable

        elif self.kind == "ot_linear":
            return t

    def sigma_func(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """sigma function that scales the standard deviation, usually goes from 0. to 1.

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: sigma value
        """
        t = self.tensor_check(t)
        l_min, l_max = self.log_snr_range

        if self.kind == "log_snr":
            alpha = self.alpha(t)
            return (1 - alpha.pow(2)).sqrt()

        elif self.kind == "ve_log_snr":
            # compute sigma value given snr range

            l_min, l_max = self.log_snr_range
            t_min, t_max = self.t_min, self.t_max
            log_snr = (1 - t) * l_max + t * l_min
            return torch.exp(-log_snr / 2)

        elif self.kind == "ot_linear":
            return 1-t

    def alpha(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute alpha value for the mapped time in [t_min, t_max]

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: alpha value
        """
        return self.alpha_func(self.t_map(t))

    def sigma(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute sigma value for mapped time in [t_min, t_max]

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: sigma value
        """
        return self.sigma_func(self.t_map(t))

    def alpha_deriv(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute alpha derivative for mapped time in [t_min, t_max]

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: time derivative of alpha_func
        """
        t_tilde = self.t_map(t)
        alpha_deriv_t = self.derivative(t_tilde, self.alpha_func).detach()
        return alpha_deriv_t

    def sigma_deriv(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute sigma derivative for the mapped time in [t_min, t_max]

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: sigma derivative
        """
        t_tilde = self.t_map(t)
        sigma_deriv_t = self.derivative(t_tilde, self.sigma_func).detach()
        return sigma_deriv_t

    def beta(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute the drift coefficient for the OU process of the form
        $dx = -\frac{1}{2} \beta(t) x dt + g(t) dw_t$

        Args:
            t (Union[float, torch.Tensor]): t in [0, 1]

        Returns:
            torch.Tensor: beta(t)
        """
        # t = self.t_map(t)
        alpha = self.alpha(t).detach()
        t_map = self.t_map(t)
        alpha_deriv_t = self.alpha_deriv(t)
        beta = -2.0 * alpha_deriv_t / alpha

        return beta

    def g(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute drift coefficient for the OU process:
        $dx = -\frac{1}{2} \beta(t) x dt + g(t) dw_t$

        Args:
            t (Union[float, torch.Tensor]): t in [0, 1]

        Returns:
            torch.Tensor: g(t)
        """
        if self.kind == "log_snr":
            t = self.t_map(t)
            g = self.beta(t).sqrt()

        else:
            alpha_deriv = self.alpha_deriv(t)
            alpha_prime_div_alpha = alpha_deriv / self.alpha(t)
            sigma_deriv = self.sigma_deriv(t)
            sigma_prime_div_sigma = sigma_deriv / self.sigma(t)

            g_sq = (
                2
                * (sigma_deriv - alpha_prime_div_alpha * self.sigma(t))
                * self.sigma(t)
            )
            g = g_sq.sqrt()

        return g

    def SNR(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Signal-to-Noise(SNR) ratio  mapped in the allowed log_SNR range

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: SNR value
        """
        t = self.tensor_check(t)

        if self.kind == "log_snr":
            SNR = self.log_SNR(t).exp()

        else:
            SNR = self.alpha(t).pow(2) / (self.sigma(t).pow(2))

        return SNR

    def log_SNR(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """log SNR value

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: log SNR value
        """
        t = self.tensor_check(t)

        if self.kind == "log_snr":
            l_min, l_max = self.log_snr_range
            log_snr = (1 - t) * l_max + t * l_min

        elif self.kind == "ot_linear":
            log_snr = self.SNR(t).log()

        return log_snr

    def compute_t_range(self, log_snr: Union[float, torch.Tensor]) -> torch.Tensor:
        """Given log(SNR) range : l_max, l_min to compute the time range.
        Hand-derivation is required for specific noise schedules.
        This function is essentially the inverse of logSNR(t)

        Args:
            log_snr (Union[float, torch.Tensor]): logSNR value

        Returns:
            torch.Tensor: the inverse logSNR
        """
        log_snr = self.tensor_check(log_snr)
        l_min, l_max = self.log_snr_range

        if self.kind == "log_snr":
            t = (1 / (l_min - l_max)) * (log_snr - l_max)

        elif self.kind == "ot_linear":
            t = ((0.5 * log_snr).exp() + 1).reciprocal()

        elif self.kind == "ve_log_snr":
            t = (1 / (l_min - l_max)) * (log_snr - l_max)

        return t

    def SNR_derivative(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """the derivative of SNR(t)

        Args:
            t (Union[float, torch.Tensor]): t in [0, 1]

        Returns:
            torch.Tensor: SNR derivative
        """
        t = self.tensor_check(t)

        if self.kind == "log_snr":
            snr_deriv = self.SNR(t) * (self.log_snr_range[0] - self.log_snr_range[1])

        elif self.kind == "ot_linear":
            snr_deriv = self.derivative(t, self.SNR)
        return snr_deriv

    def SSNR(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Signal to Signal+Noise Ratio (SSNR) = alpha^2 / (alpha^2 + sigma^2)
           SSNR monotonically goes from 1 to 0 as t going from 0 to 1.

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: SSNR value
        """
        t = self.tensor_check(t)
        return self.SNR(t) / (self.SNR(t) + 1)

    def SSNR_inv(self, ssnr: torch.Tensor) -> torch.Tensor:
        """the inverse of SSNR

        Args:
            ssnr (torch.Tensor): ssnr in [0, 1]

        Returns:
            torch.Tensor: time in [0, 1]
        """
        l_min, l_max = self.log_snr_range
        if self.kind == "log_snr":
            return ((ssnr / (1 - ssnr)).log() - l_max) / (l_min - l_max)
        elif self.kind == "ot_linear":
            # the value of SNNR_inv(t=0.5) need to be determined with L'Hôpital rule
            # the inver SNNR_function is solved anyltically:
            # see woflram alpha result: https://tinyurl.com/bdh4es5a
            singularity_check = (ssnr - 0.5).abs() < self._eps
            ssnr_mask = singularity_check.float()
            ssnr = ssnr_mask * (0.5 + self._eps) + (1.0 - ssnr_mask) * ssnr

            return (ssnr + (-ssnr * (ssnr - 1)).sqrt() - 1) / (2 * ssnr - 1)

    def SSNR_inv_deriv(self, ssnr: Union[float, torch.Tensor]) -> torch.Tensor:
        """SSNR_inv derivative. SSNR_inv is a CDF like quantity, so its derivative is a PDF-like quantity

        Args:
            ssnr (Union[float, torch.Tensor]): SSNR in [0, 1]

        Returns:
            torch.Tensor: derivative of SSNR
        """
        ssnr = self.tensor_check(ssnr)
        deriv = self.derivative(ssnr, self.SSNR_inv)
        return deriv

    def prob_SSNR(self, ssnr: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute prob (SSNR(t)), the minus sign is accounted for the inversion of integration range

        Args:
            ssnr (Union[float, torch.Tensor]): SSNR value

        Returns:
            torch.Tensor: Prob(SSNR)
        """
        return -self.SSNR_inv_deriv(ssnr)

    def linear_logsnr_grid(self, N: int, tspan: Tuple[float, float]) -> torch.Tensor:
        """Map uniform time grid to respect logSNR schedule

        Args:
            N (int): number of steps
            tspan (Tuple[float, float]): time span (t_start, t_end)

        Returns:
            torch.Tensor: time grid as torch.Tensor
        """

        logsnr_noise = GaussianNoiseSchedule(
            kind="log_snr", log_snr_range=self.log_snr_range
        )

        ts = torch.linspace(tspan[0], tspan[1], N + 1)
        SSNR_vp = logsnr_noise.SSNR(ts)
        grid = self.SSNR_inv(SSNR_vp)

        # map from t_tilde back to t
        grid = (grid - self.t_min) / (self.t_max - self.t_min)

        return grid