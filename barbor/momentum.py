import torch

def apply_momentum(
    param: torch.Tensor,
    grad: torch.Tensor,
    alpha: torch.Tensor,
    state: dict,
    momentum: float,
    dampening: float,
    nesterov: bool
):
    """Apply momentum to parameter update
    
    Args:
        param: Parameter tensor to update
        grad: Gradient of parameter
        alpha: Step size
        state: Optimizer state for the parameter
        momentum: Momentum factor
        dampening: Momentum dampening factor
        nesterov: Whether to use Nesterov momentum
    """
    if 'momentum_buffer' not in state:
        state['momentum_buffer'] = torch.zeros_like(param)
    
    buf = state['momentum_buffer']
    
    if nesterov:
        # Nesterov momentum
        grad_corrected = grad.add(buf, alpha=momentum)
        param.data.add_(grad_corrected, alpha=-alpha.item() if torch.is_tensor(alpha) else -alpha)
        buf.mul_(momentum).add_(grad, alpha=1 - dampening)
    else:
        # Standard momentum
        buf.mul_(momentum).add_(grad, alpha=1 - dampening)
        param.data.add_(buf, alpha=-alpha.item() if torch.is_tensor(alpha) else -alpha)