"""Small reusable helpers for inspecting and capturing PyTorch module boundaries."""

from collections.abc import Iterable
from contextlib import AbstractContextManager

import torch


def first_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        values: Iterable = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise TypeError(f"expected tensor output, got {type(value).__name__}")
    for item in values:
        try:
            return first_tensor(item)
        except TypeError:
            pass
    raise TypeError("module value contains no tensor")


def tensor_signature(value) -> str:
    tensor = first_tensor(value).detach()
    return (
        f"[{'x'.join(map(str, tensor.shape))}; {str(tensor.dtype).removeprefix('torch.')}; "
        f"range: {tensor.min().item():.4f}..{tensor.max().item():.4f}; "
        f"absum: {tensor.abs().sum().item():.2f}]"
    )


class TensorCapture(AbstractContextManager):
    def __init__(self, destination: dict[str, torch.Tensor]):
        self.destination = destination
        self.handles = []

    def _store(self, name: str, value) -> None:
        if name in self.destination:
            raise RuntimeError(f"tensor was captured more than once: {name}")
        self.destination[name] = first_tensor(value).detach().float().clone().contiguous()

    def input(self, module: torch.nn.Module, name: str) -> None:
        self.handles.append(
            module.register_forward_pre_hook(
                lambda _module, inputs: self._store(name, inputs)
            )
        )

    def output(self, module: torch.nn.Module, name: str) -> None:
        self.handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, result: self._store(name, result)
            )
        )

    def __exit__(self, exception_type, exception, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
