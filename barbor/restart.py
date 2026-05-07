import torch
from typing import Union
from enum import Enum

class RestartCondition(Enum):
    GRADIENT = 'gradient'
    ANGLE = 'angle'
    BOTH = 'both'

def check_restart_condition(
    s: torch.Tensor,
    y: torch.Tensor,
    grad: torch.Tensor,
    prev_grad: torch.Tensor,
    condition: Union[str, RestartCondition],
    tol: float = 0.9
) -> bool:
    """Check if restart condition is met
    
    Restart conditions help prevent BB method from diverging on non-convex problems
    
    Args:
        s: Parameter difference (x_k - x_{k-1})
        y: Gradient difference (∇f_k - ∇f_{k-1})
        grad: Current gradient
        prev_grad: Previous gradient
        condition: Restart condition type
        tol: Tolerance for restart condition
        
    Returns:
        True if restart condition is met
    """
    if isinstance(condition, str):
        condition = RestartCondition(condition.lower())
    
    if condition == RestartCondition.GRADIENT:
        return _check_gradient_condition(s, y, tol)
    elif condition == RestartCondition.ANGLE:
        return _check_angle_condition(grad, prev_grad, tol)
    elif condition == RestartCondition.BOTH:
        return _check_both_conditions(s, y, grad, prev_grad, tol)
    else:
        raise ValueError(f"Unsupported restart condition: {condition}")

def _check_gradient_condition(s: torch.Tensor, y: torch.Tensor, tol: float) -> bool:
    """Check gradient-based restart condition"""
    s_norm = torch.norm(s)
    y_norm = torch.norm(y)
    
    if s_norm < 1e-12 or y_norm < 1e-12:
        return False
    
    cos_theta = torch.abs(torch.sum(s * y)) / (s_norm * y_norm)
    return cos_theta < tol

def _check_angle_condition(grad: torch.Tensor, prev_grad: torch.Tensor, tol: float) -> bool:
    """Check angle-based restart condition"""
    grad_norm = torch.norm(grad)
    prev_grad_norm = torch.norm(prev_grad)
    
    if grad_norm < 1e-12 or prev_grad_norm < 1e-12:
        return False
    
    cos_phi = torch.sum(grad * prev_grad) / (grad_norm * prev_grad_norm)
    return cos_phi < -tol

def _check_both_conditions(
    s: torch.Tensor,
    y: torch.Tensor,
    grad: torch.Tensor,
    prev_grad: torch.Tensor,
    tol: float
) -> bool:
    """Check both restart conditions"""
    restart1, restart2 = False, False
    
    s_norm = torch.norm(s)
    y_norm = torch.norm(y)
    grad_norm = torch.norm(grad)
    prev_grad_norm = torch.norm(prev_grad)
    
    if s_norm > 1e-12 and y_norm > 1e-12:
        cos_theta = torch.abs(torch.sum(s * y)) / (s_norm * y_norm)
        restart1 = cos_theta < tol
    
    if grad_norm > 1e-12 and prev_grad_norm > 1e-12:
        cos_phi = torch.sum(grad * prev_grad) / (grad_norm * prev_grad_norm)
        restart2 = cos_phi < -tol
    
    return restart1 or restart2