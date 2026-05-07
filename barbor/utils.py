import torch

def validate_arguments(
    lr: float,
    method: str,
    restart_condition: str,
    momentum: float,
    dampening: float
):
    """Validate optimizer arguments
    
    Args:
        lr: Learning rate
        method: Step size method
        restart_condition: Restart condition
        momentum: Momentum parameter
        dampening: Dampening parameter
        
    Raises:
        ValueError: If any argument is invalid
    """
    if lr <= 0.0:
        raise ValueError(f"Learning rate must be positive: {lr}")
    valid_methods = ['bb1', 'bb2', 'alternating']
    if method not in valid_methods:
        raise ValueError(f"Unsupported step size calculation method: {method}. "
                        f"Must be one of {valid_methods}")
    valid_restart_conditions = ['gradient', 'angle', 'both']
    if restart_condition not in valid_restart_conditions:
        raise ValueError(f"Unsupported restart condition: {restart_condition}. "
                        f"Must be one of {valid_restart_conditions}")
    if momentum < 0.0:
        raise ValueError(f"Momentum must be non-negative: {momentum}")
    if dampening < 0.0:
        raise ValueError(f"Dampening must be non-negative: {dampening}")

def compute_dot_products(s: torch.Tensor, y: torch.Tensor) -> tuple:
    """Compute dot products for BB step size calculation"""
    s_dot_s = torch.sum(s * s)
    s_dot_y = torch.sum(s * y)
    y_dot_y = torch.sum(y * y)
    return s_dot_s, s_dot_y, y_dot_y