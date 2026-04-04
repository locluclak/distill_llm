import torch
from torch.optim import Optimizer
import math

class SGLD(Optimizer):
    """Stochastic Gradient Langevin Dynamics Sampler with preconditioning."""
    def __init__(self, params, lr=1e-2, precondition_decay_rate=0.95,
                 num_pseudo_batches=1, num_burn_in_steps=3000, diagonal_bias=1e-8):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(lr=lr, precondition_decay_rate=precondition_decay_rate,
                        num_pseudo_batches=num_pseudo_batches,
                        num_burn_in_steps=num_burn_in_steps, diagonal_bias=diagonal_bias)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None: continue

                state = self.state[parameter]
                lr = group["lr"]
                num_pseudo_batches = group["num_pseudo_batches"]
                precondition_decay_rate = group["precondition_decay_rate"]
                gradient = parameter.grad.data

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(parameter)

                state["iteration"] += 1
                momentum = state["momentum"]
                momentum.add_((1.0 - precondition_decay_rate) * ((gradient ** 2) - momentum))

                if state["iteration"] > group["num_burn_in_steps"]:
                    sigma = 1. / torch.sqrt(torch.tensor(lr))
                else:
                    sigma = torch.zeros_like(parameter)

                preconditioner = 1. / torch.sqrt(momentum + group["diagonal_bias"])
                scaled_grad = (0.5 * preconditioner * gradient * num_pseudo_batches +
                               torch.normal(0, 1, size=gradient.shape).to(gradient.device) * sigma * torch.sqrt(preconditioner))

                parameter.data.add_(-lr * scaled_grad)
        return loss

class SWAG:
    """
    Stochastic Weight Averaging-Gaussian (SWAG) implementation.
    Reference: Maddox et al., "A Simple Baseline for Bayesian Uncertainty in Deep Learning" (NeurIPS 2019).
    
    This class tracks the running mean and squared mean of model parameters to approximate 
    a Gaussian posterior, allowing for Bayesian uncertainty estimation and ensembling.
    """
    def __init__(self, model, max_num_models=20, var_clamp=1e-30):
        """
        Initialize SWAG tracker.
        
        Args:
            model (torch.nn.Module): The model to track.
            max_num_models (int): Maximum number of deviations to store for low-rank covariance (rank K).
            var_clamp (float): Minimum variance for numerical stability.
        """
        self.model = model
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp

        self.n_models = 0
        self.params_info = [] # Store references to parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params_info.append({
                    'name': name,
                    'param': param,
                    'mean': torch.zeros_like(param.data, device='cpu'),
                    'sq_mean': torch.zeros_like(param.data, device='cpu'),
                    'cov_mat_sqrt': [] # List of deviations for low-rank approx
                })

    def collect_model(self):
        """
        Update the running mean and squared mean of the parameters.
        Should be called periodically (e.g., at the end of every epoch after some burn-in).
        """
        self.n_models += 1
        for info in self.params_info:
            p_data = info['param'].data.cpu()
            
            # Update running mean: E[w]
            info['mean'] = (info['mean'] * (self.n_models - 1) + p_data) / self.n_models
            
            # Update running squared mean: E[w^2]
            info['sq_mean'] = (info['sq_mean'] * (self.n_models - 1) + p_data**2) / self.n_models
            
            # Store deviation for low-rank covariance approximation
            dev = p_data - info['mean']
            info['cov_mat_sqrt'].append(dev)
            if len(info['cov_mat_sqrt']) > self.max_num_models:
                info['cov_mat_sqrt'].pop(0)

    def sample(self, scale=1.0, use_cov=True):
        """
        Sample model weights from the Gaussian posterior and load them into the model.
        
        Args:
            scale (float): Scaling factor for the variance (e.g., 0.5 as per paper for BMA).
            use_cov (bool): Whether to use low-rank covariance or only diagonal variance.
        """
        if self.n_models == 0:
            raise ValueError("No models collected. Call collect_model() first.")

        # Sample low-rank component vector z_2 ~ N(0, I_K)
        rank = len(self.params_info[0]['cov_mat_sqrt'])
        z_2 = torch.randn(rank) if (use_cov and rank > 1) else None

        for info in self.params_info:
            mean = info['mean']
            # Variance: E[w^2] - E[w]^2
            var = torch.clamp(info['sq_mean'] - mean**2, self.var_clamp)
            
            # Diagonal sample: z_1 ~ N(0, I_d)
            z_1 = torch.randn_like(mean)
            diag_part = torch.sqrt(var) * z_1
            
            if z_2 is not None:
                # Low-rank part: D * z_2 / sqrt(K-1)
                # D is the matrix of deviations from the mean
                D = torch.stack(info['cov_mat_sqrt'], dim=0) # (K, ...)
                K = D.size(0)
                # Flatten D to (K, numel) for matrix multiplication
                D_flat = D.view(K, -1)
                low_rank_part = (z_2 @ D_flat).view_as(mean) / math.sqrt(K - 1)
                
                # Full sample
                sample = mean + (scale / math.sqrt(2.0)) * (diag_part + low_rank_part)
            else:
                # Diagonal only sample
                sample = mean + scale * diag_part
            
            # Load sample into model parameter
            info['param'].data.copy_(sample.to(info['param'].device))

    def get_mean_model(self):
        """Load the SWA (mean) weights into the model."""
        for info in self.params_info:
            info['param'].data.copy_(info['mean'].to(info['param'].device))

    def save(self, path):
        """Save SWAG state to a file."""
        state = {
            'n_models': self.n_models,
            'max_num_models': self.max_num_models,
            'params_info': [
                {
                    'name': info['name'],
                    'mean': info['mean'],
                    'sq_mean': info['sq_mean'],
                    'cov_mat_sqrt': info['cov_mat_sqrt']
                } for info in self.params_info
            ]
        }
        torch.save(state, path)

    def load(self, path):
        """Load SWAG state from a file."""
        state = torch.load(path)
        self.n_models = state['n_models']
        self.max_num_models = state['max_num_models']
        
        # Create a mapping for quick lookup
        name_to_info = {info['name']: info for info in self.params_info}
        
        for saved_info in state['params_info']:
            if saved_info['name'] in name_to_info:
                info = name_to_info[saved_info['name']]
                info['mean'] = saved_info['mean']
                info['sq_mean'] = saved_info['sq_mean']
                info['cov_mat_sqrt'] = saved_info['cov_mat_sqrt']