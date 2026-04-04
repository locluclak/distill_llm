import torch
from torch.optim import Optimizer

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