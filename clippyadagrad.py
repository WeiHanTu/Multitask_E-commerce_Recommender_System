import torch
from torch import Tensor

from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _get_value,
						_default_to_fused_or_foreach, _differentiable_doc, _foreach_doc, _maximize_doc)
from typing import List, Optional

__all__ = ["ClippyAdagrad", "clippy_adagrad"]


class ClippyAdagrad(Optimizer):
	def __init__(
		self,
		params,
		lr=1e-2,
		lr_decay=0,
		weight_decay=0,
		initial_accumulator_value=0,
		eps=1e-10,
		foreach: Optional[bool] = None,
		*,
		maximize: bool = False,
		differentiable: bool = False,
  		lambda_rel: float = 0.5,
		lambda_abs: float = 1e-2,
	):
		if not 0.0 <= lr:
			raise ValueError(f"Invalid learning rate: {lr}")
		if not 0.0 <= lr_decay:
			raise ValueError(f"Invalid lr_decay value: {lr_decay}")
		if not 0.0 <= weight_decay:
			raise ValueError(f"Invalid weight_decay value: {weight_decay}")
		if not 0.0 <= lambda_rel:
					raise ValueError(f"Invalid lambda_rel value: {lambda_rel}")
		if not 0.0 <= lambda_abs:
					raise ValueError(f"Invalid lambda_abs value: {lambda_abs}")
		if not 0.0 <= initial_accumulator_value:
			raise ValueError(
				f"Invalid initial_accumulator_value value: {initial_accumulator_value}"
			)
		if not 0.0 <= eps:
			raise ValueError(f"Invalid epsilon value: {eps}")

		defaults = dict(
			lr=lr,
			lr_decay=lr_decay,
			eps=eps,
			weight_decay=weight_decay,
			initial_accumulator_value=initial_accumulator_value,
			foreach=foreach,
			maximize=maximize,
			differentiable=differentiable,
			lambda_rel=lambda_rel,
			lambda_abs=lambda_abs,
		)
		super().__init__(params, defaults)

		for group in self.param_groups:
			for p in group["params"]:
				state = self.state[p]
				state["step"] = torch.tensor(0.0)
				init_value = (
					complex(initial_accumulator_value, initial_accumulator_value)
					if torch.is_complex(p)
					else initial_accumulator_value
				)
				state["sum"] = torch.full_like(
					p, init_value, memory_format=torch.preserve_format
				)

	def __setstate__(self, state):
		super().__setstate__(state)
		for group in self.param_groups:
			group.setdefault("foreach", None)
			group.setdefault("maximize", False)
			group.setdefault("differentiable", False)

		state_values = list(self.state.values())
		step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
			state_values[0]["step"]
		)
		if not step_is_tensor:
			for s in state_values:
				s["step"] = torch.tensor(float(s["step"]))

	def share_memory(self):
		for group in self.param_groups:
			for p in group["params"]:
				state = self.state[p]
				state["sum"].share_memory_()

	def _init_group(self, group, params_with_grad, grads, state_sums, state_steps):
		has_sparse_grad = False
		for p in group["params"]:
			if p.grad is not None:
				if p.grad.is_sparse:
					has_sparse_grad = True
				params_with_grad.append(p)
				grads.append(p.grad)
				state = self.state[p]
				state_sums.append(state["sum"])
				state_steps.append(state["step"])

		return has_sparse_grad

	@_use_grad_for_differentiable
	def step(self, closure=None):
		"""Performs a single optimization step.

		Args:
			closure (Callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""
		loss = None

		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			params_with_grad = []
			grads = []
			state_sums = []
			state_steps = []

			has_sparse_grad = self._init_group(group, params_with_grad, grads, state_sums, state_steps)

			clippy_adagrad(
				params_with_grad,
				grads,
				state_sums,
				state_steps,
				lr=group["lr"],
				weight_decay=group["weight_decay"],
				lr_decay=group["lr_decay"],
				eps=group["eps"],
				has_sparse_grad=has_sparse_grad,
				foreach=group["foreach"],
				maximize=group["maximize"],
				differentiable=group["differentiable"],
    			lambda_rel=group["lambda_rel"],
				lambda_abs=group["lambda_abs"],
			)

		return loss


ClippyAdagrad.__doc__ = r"""Implements Adagrad algorithm with Clippy.
	""" + fr"""
	Args:
		params (iterable): iterable of parameters to optimize or dicts defining
			parameter groups
		lr (float, optional): learning rate (default: 1e-2)
		lr_decay (float, optional): learning rate decay (default: 0)
		weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
		eps (float, optional): term added to the denominator to improve
			numerical stability (default: 1e-10)
		{_foreach_doc}
		{_maximize_doc}
		{_differentiable_doc}

	.. _Adaptive Subgradient Methods for Online Learning and Stochastic
		Optimization: http://jmlr.org/papers/v12/duchi11a.html

	"""


def clippy_adagrad(
	params: List[Tensor],
	grads: List[Tensor],
	state_sums: List[Tensor],
	state_steps: List[Tensor],
	# kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
	# setting these as kwargs for now as functional API is compiled by torch/distributed/optim
	has_sparse_grad: bool = None,
	foreach: Optional[bool] = None,
	differentiable: bool = False,
	*,
	lr: float,
	weight_decay: float,
	lr_decay: float,
	eps: float,
	maximize: bool,
	lambda_rel: float,
	lambda_abs: float, 
):
	r"""Functional API that performs Adagrad algorithm computation with Clippy.

	See :class:`~torch.optim.ClippyAdagrad` for details.
	"""

	if not all(isinstance(t, torch.Tensor) for t in state_steps):
		raise RuntimeError(
			"API has changed, `state_steps` argument must contain a list of singleton tensors"
		)

	if foreach is None:
		_, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)

	if foreach and torch.jit.is_scripting():
		raise RuntimeError("torch.jit.script not supported with foreach optimizers")

	if foreach and not torch.jit.is_scripting():
		func = _multi_tensor_clippy_adagrad
	else:
		func = _single_tensor_clippy_adagrad

	func(
		params,
		grads,
		state_sums,
		state_steps,
		lr=lr,
		weight_decay=weight_decay,
		lr_decay=lr_decay,
		eps=eps,
		has_sparse_grad=has_sparse_grad,
		maximize=maximize,
		differentiable=differentiable,
		lambda_rel=lambda_rel,
		lambda_abs=lambda_abs,
	)


def _make_sparse(grad, grad_indices, values):
	size = grad.size()
	if grad_indices.numel() == 0 or values.numel() == 0:
		return torch.empty_like(grad)
	return torch.sparse_coo_tensor(grad_indices, values, size)


def _compute_clippy_factor(param,
						   grad,
						   std,
						   clr,
						   lambda_rel,
						   lambda_abs, 
         					foreach=True):
	#print(param)
	if foreach:
		#print('a')
		param = torch._foreach_abs(param)
		numerator = torch._foreach_mul(param, lambda_rel)
		numerator = torch._foreach_add(numerator, lambda_abs)
		r = torch._foreach_div(grad, std)
		clr = [-lr for lr in clr]
		denominator = torch._foreach_mul(torch._foreach_abs(r), clr)
		_res = torch._foreach_div(numerator, denominator)
		return [torch.min(torch.Tensor([1., torch.min(x)])) for x in _res]
	else:
		numerator = lambda_rel * torch.abs(param) + lambda_abs
		r = grad / std
		denominator = torch.abs(r) * clr
		return torch.min(torch.Tensor([1., torch.min(numerator / denominator)]))
	
	

def _single_tensor_clippy_adagrad(
	params: List[Tensor],
	grads: List[Tensor],
	state_sums: List[Tensor],
	state_steps: List[Tensor],
	*,
	lr: float,
	weight_decay: float,
	lr_decay: float,
	eps: float,
	has_sparse_grad: bool,
	maximize: bool,
	differentiable: bool,
	lambda_rel: float = 0.5,
	lambda_abs: float = 1e-2, 
):

	for (param, grad, state_sum, step_t) in zip(params, grads, state_sums, state_steps):
		# update step
		step_t += 1
		step = _get_value(step_t)
		grad = grad if not maximize else -grad

		if weight_decay != 0:
			if grad.is_sparse:
				raise RuntimeError(
					"weight_decay option is not compatible with sparse gradients"
				)
			grad = grad.add(param, alpha=weight_decay)

		clr = lr / (1 + (step - 1) * lr_decay)

		if grad.is_sparse:
			grad = grad.coalesce()  # the update is non-linear so indices must be unique
			grad_indices = grad._indices()
			grad_values = grad._values()

			state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
			std = state_sum.sparse_mask(grad)
			std_values = std._values().sqrt_().add_(eps)
			param.add_(
				_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr
			)
		else:
			is_complex = torch.is_complex(param)
			if is_complex:
				grad = torch.view_as_real(grad)
				state_sum = torch.view_as_real(state_sum)
				param = torch.view_as_real(param)
			state_sum.addcmul_(grad, grad, value=1)
			if differentiable:
				std = state_sum.sqrt() + eps
			else:
				std = state_sum.sqrt().add_(eps)
			clippy_factor = _compute_clippy_factor(param, grad, std, clr, lambda_rel=lambda_rel, 
                                          lambda_abs=lambda_abs, foreach=False)
			param.addcdiv_(grad, std, value=-clr*clippy_factor)
			if is_complex:
				param = torch.view_as_complex(param)
				state_sum = torch.view_as_complex(state_sum)


def _multi_tensor_clippy_adagrad(
	params: List[Tensor],
	grads: List[Tensor],
	state_sums: List[Tensor],
	state_steps: List[Tensor],
	*,
	lr: float,
	weight_decay: float,
	lr_decay: float,
	eps: float,
	has_sparse_grad: bool,
	maximize: bool,
	differentiable: bool,
 	lambda_rel: float,
	lambda_abs: float, 
):

	assert not differentiable, "_foreach ops don't support autograd"

	# Foreach functions will throw errors if given empty lists
	if len(params) == 0:
		return

	grouped_tensorlists = Optimizer._group_tensors_by_device_and_dtype([params, grads, state_sums, state_steps])
	for ((device_params, device_grads, device_state_sums, device_state_steps), _) in grouped_tensorlists.values():

		device_has_sparse_grad = any(grad.is_sparse for grad in device_grads)

		if device_has_sparse_grad:
			return _single_tensor_clippy_adagrad(
				device_params,
				device_grads,
				device_state_sums,
				device_state_steps,
				lr=lr,
				weight_decay=weight_decay,
				lr_decay=lr_decay,
				eps=eps,
				has_sparse_grad=True,
				maximize=False,
				differentiable=differentiable,
				lambda_rel=lambda_rel,
				lambda_abs=lambda_abs
			)

		if maximize:
			device_grads = torch._foreach_neg(device_grads)

		# Handle complex parameters
		device_grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads]
		device_state_sums = [
			torch.view_as_real(x) if torch.is_complex(x) else x for x in device_state_sums
		]
		device_params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_params]

		# Update steps
		torch._foreach_add_(device_state_steps, 1)

		if weight_decay != 0:
			# Re-use the intermediate memory (device_grads) already allocated for maximize
			if maximize:
				torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
			else:
				device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)

		minus_clr = [-lr / (1 + (_get_value(step) - 1) * lr_decay) for step in device_state_steps]

		torch._foreach_addcmul_(device_state_sums, device_grads, device_grads, value=1)

		std = torch._foreach_sqrt(device_state_sums)
		torch._foreach_add_(std, eps)
		
		clippy_factor = _compute_clippy_factor(device_params, device_grads, std, minus_clr, lambda_rel=lambda_rel,
		lambda_abs=lambda_abs)

		if weight_decay != 0 or maximize:
			# Again, re-use the intermediate memory (device_grads) already allocated
			torch._foreach_mul_(device_grads, minus_clr)
			torch._foreach_mul_(device_grads, clippy_factor)
			numerator = device_grads
		else:
			numerator = torch._foreach_mul(device_grads, minus_clr)
			numerator = torch._foreach_mul(numerator, clippy_factor)

		torch._foreach_addcdiv_(device_params, numerator, std)
