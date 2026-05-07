import torch
from typing import Optional, Callable, Dict, List, Tuple
from .stepsize import compute_step_size
from .restart import check_restart_condition
from .momentum import apply_momentum
from .utils import validate_arguments, compute_dot_products

class barbor(torch.optim.Optimizer):
    """Barzilar-Borwein Gradient Descent Method
    
    Args:
        params: Parameters to optimize
        lr: Initial learning rate (default: 1.0)
        method: Step size calculation method, options: 'bb1', 'bb2', 'alternating' (default: 'alternating')
        gamma: Regularization parameter to prevent zero step size (default: 1e-8)
        safe_guard: Step size safety guard factor (default: 1e-8)
        min_step: Minimum step size (default: 1e-8)
        max_step: Maximum step size (default: 1e8)
        adaptive_restart: Whether to use adaptive restart (default: True)
        restart_condition: Restart condition, options: 'gradient', 'angle', 'both' (default: 'both')
        restart_tol: Restart condition tolerance (default: 0.9)
        momentum: Momentum parameter (default: 0.0)
        dampening: Momentum dampening (default: 0.0)
        nesterov: Whether to use Nesterov momentum (default: False)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1.0,
        method: str = 'alternating',
        gamma: float = 1e-8,
        safe_guard: float = 1e-8,
        min_step: float = 1e-8,
        max_step: float = 1e8,
        adaptive_restart: bool = True,
        restart_condition: str = 'both',
        restart_tol: float = 0.9,
        momentum: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False
    ):
        # Validate input arguments
        validate_arguments(lr, method, restart_condition, momentum, dampening)
        
        defaults = dict(
            lr=lr,
            method=method,
            gamma=gamma,
            safe_guard=safe_guard,
            min_step=min_step,
            max_step=max_step,
            adaptive_restart=adaptive_restart,
            restart_condition=restart_condition,
            restart_tol=restart_tol,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov
        )
        super().__init__(params, defaults)
        
        # Initialize state for each parameter group
        self._initialize_states()
    
    def _initialize_states(self):
        """Initialize optimizer states for all parameters"""
        for group in self.param_groups:
            for p in group['params']:
                self._initialize_param_state(p, group['lr'], group['momentum'])
    
    def _initialize_param_state(self, p, lr: float, momentum: float):
        """Initialize state for a single parameter"""
        state = self.state[p]
        state.setdefault('step', 0)
        state.setdefault('prev_param', torch.zeros_like(p))
        state.setdefault('prev_grad', torch.zeros_like(p))
        state.setdefault('alpha', torch.tensor(lr, device=p.device))
        state.setdefault('prev_alpha', torch.tensor(lr, device=p.device))
        
        if momentum > 0:
            state.setdefault('momentum_buffer', torch.zeros_like(p))
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        """Perform a single optimization step
        
        Args:
            closure: A callable that recomputes the loss and returns the loss
            
        Returns:
            Loss value (if closure is provided)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                self._update_parameter(p, group)
        
        return loss
    
    def _update_parameter(self, p, group: dict):
        """Update a single parameter"""
        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError('BarzilaiBorwein does not support sparse gradients')
        
        state = self.state[p]
        step = state['step']
        
        if step == 0:
            # First step: use initial learning rate
            new_alpha = torch.tensor(group['lr'], device=p.device)
            restart = False
        else:
            # Compute new step size
            s, y = self._compute_updates(p, state)
            restart = self._should_restart(s, y, grad, state, group)
            new_alpha = self._compute_new_step_size(s, y, state, group, restart)

        self._save_state(p, grad, state, new_alpha)
        # Update parameter with new step size
        self._apply_update(p, grad, state, new_alpha, group)
        
        state['step'] += 1
    
    def _compute_updates(self, p, state: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute parameter and gradient updates"""
        s = p.data - state['prev_param']
        y = p.grad - state['prev_grad']
        return s, y
    
    def _should_restart(self, s: torch.Tensor, y: torch.Tensor, 
                       grad: torch.Tensor, state: dict, group: dict) -> bool:
        """Check if restart condition is met"""
        if not group['adaptive_restart'] or state['step'] <= 1:
            return False
        
        return check_restart_condition(
            s, y, grad, state['prev_grad'],
            group['restart_condition'], group['restart_tol']
        )
    
    def _compute_new_step_size(self, s: torch.Tensor, y: torch.Tensor, 
                              state: dict, group: dict, restart: bool) -> torch.Tensor:
        """Compute new step size"""
        if restart:
            return torch.tensor(group['lr'], device=s.device)
        
        s_dot_s, s_dot_y, y_dot_y = compute_dot_products(s, y)
        
        new_alpha = compute_step_size(
            s_dot_s, s_dot_y, y_dot_y,
            state['step'], group['method'],
            group['gamma'], group['safe_guard'],
            s.device
        )
        
        # Clip step size
        return torch.clamp(new_alpha, group['min_step'], group['max_step'])
    
    def _save_state(self, p, grad: torch.Tensor, state: dict, alpha: torch.Tensor):
        """Save current state for next iteration"""
        state['prev_param'].copy_(p.data)
        state['prev_grad'].copy_(grad)
    
    def _apply_update(self, p, grad: torch.Tensor, state: dict, 
                     alpha: torch.Tensor, group: dict):
        """Apply parameter update"""
        state['prev_alpha'] = state['alpha']
        state['alpha'] = alpha
        
        if group['momentum'] > 0:
            apply_momentum(
                p, grad, alpha, state, 
                group['momentum'], group['dampening'], group['nesterov']
            )
        else:
            p.data.add_(grad, alpha=-alpha)
    
    def get_step_sizes(self) -> List[float]:
        """Get current step sizes for all parameters"""
        alphas = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'alpha' in state:
                    alpha = state['alpha']
                    alphas.append(alpha.item() if torch.is_tensor(alpha) else alpha)
        return alphas
    
    def reset_step_sizes(self, alpha: float = 1.0):
        """Reset step sizes for all parameters"""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                device = p.device
                state['alpha'] = torch.tensor(alpha, device=device)
                state['prev_alpha'] = torch.tensor(alpha, device=device)
    
    def get_gradient_history_info(self) -> List[Tuple[float, float, float]]:
        """Get gradient history information
        
        Returns:
            List of (s·s, s·y, y·y) values for each parameter
        """
        info = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'prev_grad' in state and p.grad is not None:
                    s, y = self._compute_updates(p, state)
                    s_dot_s, s_dot_y, y_dot_y = compute_dot_products(s, y)
                    info.append((s_dot_s.item(), s_dot_y.item(), y_dot_y.item()))
        return info
    
    def get_convergence_info(self) -> Dict[str, List[float]]:
        """Get convergence information"""
        info = {
            'step_sizes': [],
            'gradient_norms': [],
            'step_norms': []
        }
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'alpha' in state:
                    alpha = state['alpha']
                    info['step_sizes'].append(alpha.item() if torch.is_tensor(alpha) else alpha)
                if p.grad is not None:
                    grad_norm = torch.norm(p.grad).item()
                    info['gradient_norms'].append(grad_norm)
                if 'prev_param' in state:
                    step_norm = torch.norm(p.data - state['prev_param']).item()
                    info['step_norms'].append(step_norm)
        return info