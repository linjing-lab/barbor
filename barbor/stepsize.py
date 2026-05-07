import torch
from typing import Union
from enum import Enum

class StepSizeMethod(Enum):
    BB1 = 'bb1'
    BB2 = 'bb2'
    ALTERNATING = 'alternating'

def compute_step_size(
    s_dot_s: torch.Tensor,
    s_dot_y: torch.Tensor,
    y_dot_y: torch.Tensor,
    step: int,
    method: Union[str, StepSizeMethod],
    gamma: float = 1e-8,
    safe_guard: float = 1e-8,
    device: torch.device = None
) -> torch.Tensor:
    """Compute step size using Barzilai-Borwein method
    
    Args:
        s_dot_s: s·s where s = x_k - x_{k-1}
        s_dot_y: s·y where y = ∇f_k - ∇f_{k-1}
        y_dot_y: y·y
        step: Current iteration number
        method: Step size calculation method
        gamma: Regularization parameter
        safe_guard: Numerical safety parameter
        device: Device for tensor creation
        
    Returns:
        Computed step size
    """
    if isinstance(method, str):
        method = StepSizeMethod(method.lower())
    
    if method == StepSizeMethod.BB1:
        return _bb1_step(s_dot_s, s_dot_y, gamma, safe_guard, device)
    elif method == StepSizeMethod.BB2:
        return _bb2_step(s_dot_y, y_dot_y, gamma, safe_guard, device)
    elif method == StepSizeMethod.ALTERNATING:
        # Alternate between BB1 and BB2
        if step % 2 == 1:
            return _bb1_step(s_dot_s, s_dot_y, gamma, safe_guard, device)
        else:
            return _bb2_step(s_dot_y, y_dot_y, gamma, safe_guard, device)
    else:
        raise ValueError(f"Unsupported step size calculation method: {method}")

def _bb1_step(
    s_dot_s: torch.Tensor,
    s_dot_y: torch.Tensor,
    gamma: float,
    safe_guard: float,
    device: torch.device
) -> torch.Tensor:
    """Compute BB1 step size"""
    if torch.abs(s_dot_y) < safe_guard:
        if torch.abs(s_dot_s) < safe_guard:
            return torch.tensor(1.0, device=device)
        else:
            return s_dot_s.clone()
    
    denominator = s_dot_y + gamma
    if denominator <= 0:
        denominator = torch.abs(denominator) + gamma
    
    return s_dot_s / denominator

def _bb2_step(
    s_dot_y: torch.Tensor,
    y_dot_y: torch.Tensor,
    gamma: float,
    safe_guard: float,
    device: torch.device
) -> torch.Tensor:
    """Compute BB2 step size"""
    if torch.abs(y_dot_y) < safe_guard:
        if torch.abs(s_dot_y) < safe_guard:
            return torch.tensor(1.0, device=device)
        else:
            return s_dot_y.clone()
    
    denominator = y_dot_y + gamma
    if denominator <= 0:
        denominator = torch.abs(denominator) + gamma
    
    return s_dot_y / denominator