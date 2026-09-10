"""AdamW with CPU-resident master weights, moments and update ("ZeRO-offload"-style).

Purpose: train a model whose fp32 optimizer state does not fit on the GPU
(MoF-MoE is 385M params: weights + grads + AdamW moments alone are 6.2 GB).
The GPU keeps only the working copy of the weights and, during backward, one
layer's gradient at a time; gradients are streamed to pinned CPU buffers as soon
as each parameter's gradient has been accumulated and are freed on the GPU.

The update is the *same algorithm* as ``torch.optim.AdamW`` (it literally calls
``torch.optim.AdamW`` on the CPU master weights with the same hyperparameters),
so results match a GPU run up to floating-point rounding. Gradient accumulation
is supported (micro-batch gradients are summed into the CPU buffers). The
scheduler-facing ``param_groups`` live on this wrapper; their ``lr`` is copied
to the inner CPU optimizer at every step.
"""
from typing import Iterable, List

import torch
from torch.optim import Optimizer


class CPUOffloadAdamW(Optimizer):
    def __init__(self, params: Iterable[torch.Tensor], lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-2, pin_memory=True, num_threads=None):
        # num_threads: CPU intra-op threads for the update (and, being process-wide, for the
        # CPU-resident EMA update the workspace performs right after each step). Forked
        # dataloader workers stay safe because the datasets call threadpool_limits(1).
        self.num_threads = num_threads
        if num_threads:
            torch.set_num_threads(int(num_threads))
        params = list(params)
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._params: List[torch.Tensor] = [p for g in self.param_groups for p in g["params"]]
        # fp32 master copy + gradient accumulation buffers on the CPU
        self._master: List[torch.Tensor] = []
        self._grad_bufs: List[torch.Tensor] = []
        for p in self._params:
            m = p.detach().to("cpu", copy=True).float()
            b = torch.zeros_like(m)
            if pin_memory and torch.cuda.is_available():
                m, b = m.pin_memory(), b.pin_memory()
            self._master.append(m)
            self._grad_bufs.append(b)
        self._inner = torch.optim.AdamW(self._master, lr=lr, betas=tuple(betas), eps=eps,
                                        weight_decay=weight_decay, foreach=True)
        self._hooks = [p.register_post_accumulate_grad_hook(self._make_hook(i))
                       for i, p in enumerate(self._params)]

    # -- gradient streaming -------------------------------------------------
    def _make_hook(self, idx: int):
        buf = self._grad_bufs[idx]

        def hook(p: torch.Tensor):
            if p.grad is None:
                return
            # synchronous D2H copy of this parameter's gradient, accumulate on CPU, free on GPU
            buf.add_(p.grad.detach().to(buf.device, dtype=buf.dtype))
            p.grad = None

        return hook

    def zero_grad(self, set_to_none: bool = True):
        # Gradients never persist on the GPU; CPU buffers are cleared in step().
        for p in self._params:
            p.grad = None

    # -- update ---------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # any gradient that was not delivered through the hook (e.g. params on CPU in tests)
        for p, buf in zip(self._params, self._grad_bufs):
            if p.grad is not None:
                buf.add_(p.grad.detach().to(buf.device, dtype=buf.dtype))
                p.grad = None
        for g_out, g_in in zip(self.param_groups, self._inner.param_groups):
            g_in["lr"] = g_out["lr"]
            g_in["weight_decay"] = g_out["weight_decay"]
            g_in["betas"] = tuple(g_out["betas"])
            g_in["eps"] = g_out["eps"]
        for m, buf in zip(self._master, self._grad_bufs):
            m.grad = buf
        prev_threads = torch.get_num_threads()
        if self.num_threads:
            torch.set_num_threads(int(self.num_threads))
        try:
            self._inner.step()
        finally:
            if self.num_threads:
                torch.set_num_threads(prev_threads)
        for m, buf in zip(self._master, self._grad_bufs):
            m.grad = None
            buf.zero_()
        for p, m in zip(self._params, self._master):
            p.copy_(m, non_blocking=True)  # H2D from pinned memory; stream-ordered
        return loss

    # -- (de)serialization ------------------------------------------------------
    def state_dict(self):
        return {
            "inner": self._inner.state_dict(),
            "master": [m.detach().clone() for m in self._master],
            "param_groups": [{k: v for k, v in g.items() if k != "params"} for g in self.param_groups],
        }

    def load_state_dict(self, state_dict):
        self._inner.load_state_dict(state_dict["inner"])
        for m, saved in zip(self._master, state_dict["master"]):
            m.copy_(saved)
        for g, saved in zip(self.param_groups, state_dict["param_groups"]):
            g.update(saved)
        with torch.no_grad():
            for p, m in zip(self._params, self._master):
                p.copy_(m)

    @torch.no_grad()
    def sync_master_from_params(self):
        """Re-initialise the master copy from the (GPU) parameters, e.g. after loading a
        checkpoint that contains model weights but no optimizer state."""
        for p, m in zip(self._params, self._master):
            m.copy_(p.detach().to(m.device, dtype=m.dtype))
