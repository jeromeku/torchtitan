from torch._ops import HigherOrderOperator
from torch.utils._python_dispatch import TorchDispatchMode, transform_subclass, is_traceable_wrapper_subclass, return_and_correct_aliasing

import torch, functools
import contextlib

@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard

def _make(cls, data: torch.Tensor):
    return torch.Tensor._make_subclass(cls, data, require_grad=data.requires_grad)  # helper

class First(torch.Tensor):
    @staticmethod
    def __new__(cls, data): 
        self = torch.Tensor._make_subclass(cls, torch.empty_like(data))
        self._data = data
        return self

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print("First handles", func.__name__)
        # fall through to the next candidate
        # breakpoint()
        if isinstance(args[0], cls):
            new_args = (args[0]._data, *args[1:])
            return func(*new_args, **(kwargs or {}))

class Second(torch.Tensor):
    @staticmethod
    def __new__(cls, data: torch.Tensor): 
        self = torch.Tensor._make_wrapper_subclass(cls, data.size(), data.stride(), data.storage_offset())
        return self

    def __init__(self, data: torch.Tensor):
        self._tensor = data
        
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print("Second handles", func.__name__)
        with no_dispatch():
            return func(*args, **(kwargs or {}))

a = First(torch.eye(2))
b = Second(torch.eye(2))

print("a @ b  -->")
torch.matmul(a, b)       # First is left-most → First runs first

# print("\nb @ a  -->")
# torch.matmul(b, a)       # Second is left-most → Second runs first
