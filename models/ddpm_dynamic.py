import torch
from torch import nn
from torch import Tensor
from typing import Dict
from torch.nn import functional as F
from utils.class_registry import ClassRegistry


noise_scheduler_registry = ClassRegistry()


def get_coeffs_primitives(
    T: int = 1000,
    beta_min: float = 1e-4,
    beta_max: float = 2e-2,
) -> Dict[str, Tensor]:
    
    betas = torch.linspace(beta_min, beta_max, T).float()
    alphas = 1 - betas

    sqrt_alphas = torch.sqrt(alphas)
    alphas_hat = torch.cumprod(alphas, dim=0)

    alphas_hat_prev = torch.cat([torch.FloatTensor([1.]), alphas_hat[:-1]])
    sqrt_alphas_hat = torch.sqrt(alphas_hat)
    sqrt_1m_alphas_hat = torch.sqrt(1.0 - alphas_hat)
    posterior_mean_coef1 = ((1 - alphas_hat_prev) / (1 - alphas_hat)) * torch.sqrt(alphas)
    posterior_mean_coef2 = (torch.sqrt(alphas_hat_prev) / (1 - alphas_hat)) * betas
    posterior_variance = ((1 - alphas_hat_prev) / (1 - alphas_hat)) * betas

    return {
        "betas": betas,
        "alphas": alphas,
        "sqrt_alphas_hat": sqrt_alphas_hat,
        "sqrt_1m_alphas_hat": sqrt_1m_alphas_hat,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
        "posterior_variance": posterior_variance
    }


def extract_values_from_times(values: Tensor, times: torch.LongTensor) -> Tensor:
    values = values[times]
    return values[:, None, None, None]


@noise_scheduler_registry.add_to_registry(name="ddpm")
class DDPMDynamic(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        beta_min = model_config['beta_min']
        beta_max = model_config['beta_max']
        self.T = model_config['num_timesteps']

        coeffs_primitives = get_coeffs_primitives(self.T, beta_min, beta_max)

        for name, tensor in coeffs_primitives.items():
            self.register_buffer(name, tensor)


    def sample_time_on_device(self, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        return torch.randint(0, self.T, (batch_size,), device=device)


    def sample_from_posterior_q(
        self,
        x_t: Tensor,
        x_0: Tensor,
        t: torch.LongTensor
    ) -> Tensor:
        posterior_mean =  extract_values_from_times(self.posterior_mean_coef1, t) * x_t + \
                          extract_values_from_times(self.posterior_mean_coef2, t) * x_0
        posterior_variance = extract_values_from_times(self.posterior_variance, t)
        eps = torch.randn_like(x_0).to(x_0.device)
        return posterior_mean + torch.sqrt(posterior_variance) * eps


    def get_x_zero(
        self,
        x_t: Tensor,
        eps: Tensor,
        t: torch.LongTensor
    ) -> Tensor:
        sqrt_alphas_hat = extract_values_from_times(self.sqrt_alphas_hat, t)
        sqrt_1m_alphas_hat = extract_values_from_times(self.sqrt_1m_alphas_hat, t)

        x_0 = (x_t - sqrt_1m_alphas_hat * eps) / sqrt_alphas_hat
        return x_0


    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x_0 = batch['x_0']
        t = batch['t']
        eps = batch['eps']
        mean = extract_values_from_times(self.sqrt_alphas_hat, t) * x_0
        variance = extract_values_from_times(self.sqrt_1m_alphas_hat, t)
        x_t = mean + variance * eps
        return {
            "x_t": x_t,
            "eps": eps
        }
