import contextlib
import inspect
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from transformers.models.qwen3_moe.configuration_qwen3_moe import (
    Qwen3MoeConfig as HFQwen3MoeConfig,
)


def safe_reset_parameters(mod: torch.nn.Module):
    if hasattr(mod, 'reset_parameters'):
        mod.reset_parameters()


def check_equal(a: any, b: any, label: str, should_raise=True, verbose=False):
    passed = True
    try:
        assert a == b, f"{label} mismatch: {a} != {b}"
    except AssertionError as e:
        if should_raise:
            raise AssertionError(e)
        else:
            print(f"{e.args[0]}")
            passed = False
    if passed and verbose:
        print(f"{label} matches")

    return passed


TEST_TOL = {torch.float32: (1e-5, 1e-5), torch.float16: (1e-3, 1e-3), torch.bfloat16: (1e-2, 1e-2)}


def get_kwargs(func: callable):
    """
    Get kwargs of a callable
    """

    sig = inspect.signature(func)
    kwargs = {
        name: param.default
        for name, param in sig.parameters.items()
        if param.kind == inspect.Parameter.KEYWORD_ONLY
        or param.default is not inspect.Parameter.empty
    }

    return kwargs


# pytest param ids
def key_value_format(x):
    return f"{x=}"


def print_allclose_violations(
    ref: torch.Tensor,
    test: torch.Tensor,
    label: str,
    atol: float,
    rtol: float,
    k: int = 10,
):
    """
    Report where torch.allclose(ref, test, rtol, atol) fails.

    """
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch: ref{tuple(ref.shape)} vs test{tuple(test.shape)}")

    diff = (ref - test).abs()
    tol = atol + rtol * test.abs()
    violations = diff > tol

    is_close = not violations.any().item()  # .item() => bool not Tensor
    worst_excess = 0.0 if is_close else (diff[violations] - tol[violations]).max().item()

    left_cond = "abs(ref - test)"
    right_cond = "atol + rtol * abs(test)"

    # --- summary statistics ---
    if not is_close:
        max_diff = diff.max()
        ref_min, ref_max = ref.min().item(), ref.max().item()
        test_min, test_max = test.min().item(), test.max().item()
        print(
            f"{label.upper()} not allclose with atol={atol:.1e} rtol={rtol:.1e}\n"
            f" max_abs_diff={max_diff.item():.6f}\n"
            f" ref[min={ref_min:.6f}, max={ref_max:.6f}] ‖"
            f" test[min={test_min:.6f}, max={test_max:.6f}]"
        )

    num_bad = violations.sum().item()
    print(f" {num_bad} / {ref.numel()} fails => {left_cond} > {right_cond}")
    # print(f" worst excess over tolerance = {worst_excess:.6g}")

    # severity = amount by which each violation exceeds its tolerance
    severity = (diff - tol).masked_fill(~violations, -1)
    k = min(k, num_bad)
    top_sev, flat_idx = torch.topk(severity.flatten(), k)
    unravel_idx = torch.unravel_index(flat_idx, ref.shape)
    indices = list(zip(*[i.tolist() for i in unravel_idx]))

    # print(f" top-{k} diffs:")
    print(
        f" {'idx':>15} | {'ref':^10} | {'test':^10} | {f"{left_cond}":^15} | {f"{right_cond}":^25}"
    )  # | {'excess':>12}")
    print(f" {'-' * (15 + 12 * 2 + 15 + 25 + 10)}")
    for idx, sev in zip(indices, top_sev.tolist()):
        r_val = ref[idx].item()
        t_val = test[idx].item()
        d_val = diff[idx].item()
        tol_val = tol[idx].item()
        print(
            f" {str(idx):>15} | {r_val:^10.6f} | {t_val:^10.6f} | "
            f"{d_val:^15.6f} | {tol_val:^25.6f}"  # | {sev:12.6f}"
        )


def sample_tensors(
    ref: torch.Tensor,
    test: torch.Tensor,
    num_samples: int = 10,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> None:
    if ref.shape != test.shape:
        raise ValueError("ref and test must have identical shapes")
    if ref.device != test.device:
        raise ValueError("ref and test must live on the same device")

    if generator is None:
        generator = torch.Generator(device=ref.device)
        if seed is not None:
            generator.manual_seed(seed)

    numel = ref.numel()
    flat_idx = torch.randint(0, numel, (num_samples,), generator=generator, device=ref.device)

    ref_vals = ref.flatten().index_select(0, flat_idx)
    test_vals = test.flatten().index_select(0, flat_idx)

    return ref_vals, test_vals


def pretty_tensor(t: torch.Tensor, precision=6):
    return "[ " + ", ".join([f"{v:.{precision}f}" for v in t.tolist()]) + " ]"


def check_tensors(
    ref: torch.Tensor,
    test: torch.Tensor,
    label: str,
    atol: float,
    rtol: float,
    verbose_error: bool = True,
    print_sample: bool = False,
    num_samples: int = 10,
):
    if ref.shape != test.shape:
        print(f"WARNING: {ref.shape=} != {test.shape=}, attempting to reshape")
        test = test.reshape_as(ref)

    passed = torch.allclose(ref, test, atol=atol, rtol=rtol)

    if not passed:
        if verbose_error:
            print_allclose_violations(ref, test, atol=atol, rtol=rtol, label=label)
            raise AssertionError(f"{label.upper()} mismatch!")
        else:
            torch.testing.assert_close(ref, test, atol=atol, rtol=rtol)

    if print_sample:
        ref_sample, test_sample = sample_tensors(ref, test, num_samples=num_samples)
        msg = (
            f"{label:<}\n"
            f"{'ref':>5}: {pretty_tensor(ref_sample)}\n"
            f"{'test':>5}: {pretty_tensor(test_sample)}"
        )
        print(msg)


def check_tensor_attributes(ref, test, label: str):
    assert ref.shape == test.shape, f"{label} shape mismatch: {ref.shape} != {test.shape}"
    assert ref.dtype == test.dtype, f"{label} dtype mismatch: {ref.dtype} != {test.dtype}"
    assert ref.device.type == test.device.type, (
        f"{label} device mismatch: {ref.device.type} != {test.device.type}"
    )


InitFn = dict[str, dict[str, callable]]


def all_parameters_on_device(module: torch.nn.Module, device: str):
    return all(param.device.type == device for param in module.parameters())


def has_parameters(module: torch.nn.Module):
    return len(module._parameters) > 0


def all_buffers_on_device(module: torch.nn.Module, device: str):
    return all(buffer.device.type == device for buffer in module.buffers())


def has_buffers(module: torch.nn.Module):
    return len(module._buffers) > 0


def _init_params_and_buffers(
    module: torch.nn.Module, dtype: torch.device, device: str = "cuda", init_fns: InitFn = None
):
    module_name = type(module).__name__

    # print(f"{type(module).__name__}: {has_parameters(module)} {has_buffers(module)}")

    if has_buffers(module) and not all_buffers_on_device(module, device):
        for name, buffer in module._buffers.items():
            module._buffers[name] = buffer.to(device)

    for name, buffer in module.named_buffers(recurse=False):
        assert buffer.device.type == device, (
            f"Buffer {name} device: {buffer.device.type} != {device}"
        )

    if has_parameters(module):
        if not all_parameters_on_device(module, device):
            if len(module._buffers) == 0:
                module.to_empty(device=device, recurse=False).to(dtype)
            else:
                for name, param in module._parameters.items():
                    new_param = torch.nn.Parameter(param.to(device, dtype), param.requires_grad)
                    module._parameters[name] = new_param

        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        elif module_name in init_fns:
            for param_name, fn in init_fns[module_name].items():
                p = getattr(module, param_name, None)
                assert p is not None, f"{module} does not have param {param_name}"
                fn(p)
        else:
            print(
                f"WARNING: {module_name} does not have `reset_parameters` and no initialization function provided for module's weights!"
            )

    for name, param in module.named_parameters(recurse=False):
        assert param.device.type == device, f"Param {name} device: {param.device.type} != {device}"
        assert param.dtype == dtype, f"Param {name} dtype: {param.dtype} != {dtype}"
        assert torch.count_nonzero(param) != 0, f"Param {name} all zeros"


def check_init_fns(model: torch.nn.Module, init_fns: dict[str, dict[str, callable]]):
    module_types = set([type(module).__name__ for module in model.modules()])
    for k in init_fns.keys():
        assert k in module_types, f"{k} not in {module_types}"

def initialize_model(
    model: torch.nn.Module, dtype: torch.dtype, device: str, init_fns: InitFn = None
):
    """
    Primarly for moving a model that was 'meta' initialized to the actual device, otherwise a simple module.to(...) will do.

    When a module is initialized on device 'meta', the params can be moved to device by calling to_empty(device) on the module
    followed by module.reset_parameters() to properly set the initial state of the params.

    However, if the module contains buffers, this wil result in all buffers being initialized to zeros, hence, the need to
    separate initialize params and buffers.

    For params, both device and dtype will be set, for buffers only device will be set.

    init_fns is a mapping from fully-qualified param name to an initialization function that takes a torch.tensor as an input.
    This is needed when a module does not define a reset_parameters method for initializing its params, such as RMSNorm.
    The scale of the RMSNorm after a call to empty_like will be all zeros, hence we have to manually init them to ones.

    """

    if init_fns is not None:
        check_init_fns(model, init_fns=init_fns)
    mod_fn = partial(_init_params_and_buffers, dtype=dtype, device=device, init_fns=init_fns)
    model.apply(mod_fn)
    return model


@torch.no_grad()
def to_dtype(
    model: torch.nn.Module,
    dtype: torch.dtype,
    include_grad: bool = True,
    include_buffers: bool = False,
):
    """
    Moves all model params to dtype, skipping buffers unless requested.
    """
    if include_buffers:
        return model.to(dtype)

    for name, param in model.named_parameters():
        if param.data.dtype != dtype:
            param.data = param.data.to(dtype, non_blocking=True)
            if include_grad and param.grad is not None:
                param.grad.data = param.grad.data.to(dtype, non_blocking=True)

    return model


@dataclass(kw_only=True)
class ModuleMapping:
    src_key: str
    dst_key: str
    src_submod_keys: list[str] = None
    dst_submod_keys: list[str] = None
    src_weights: list[str] = None
    dst_weights: list[str] = None

    def _check_submodule_weights(self, submod_keys: list[str], weights: list[str]):
        if len(weights) != len(submod_keys):
            if len(weights) == 1:
                weights = weights * len(self.src_submod_keys)
            else:
                raise ValueError(
                    f"len(src_weights) must be either 1 or equal to the number of submodule keys (only single weight supported per submodule): {len(weights)} != {len(submod_keys)}"
                )

    def __post_init__(self):
        if self.dst_weights is None:
            self.dst_weights = self.src_weights
        else:
            assert len(self.src_weights) == len(self.dst_weights), (
                f"must provide same number of src and dst weight keys: {len(self.src_weights)} != {len(self.dst_weights)}"
            )

        if self.src_submod_keys is not None:
            assert self.dst_submod_keys is not None, f"must provide both src and dst submodule keys"
            assert len(self.src_submod_keys) == len(self.dst_submod_keys), (
                f"must pass equal number of src and dst submodule keys"
            )
            self._check_submodule_weights(self.src_submod_keys, self.src_weights)
            self._check_submodule_weights(self.dst_submod_keys, self.dst_weights)

    def get_key_pair(self):
        return (self.src_key, self.dst_key)


@torch.no_grad()
def _copy_weight(src: torch.Tensor, dst: torch.Tensor, label: str):
    check_tensor_attributes(src, dst, label)
    dst.copy_(src)


def _copy_module(
    src_model: torch.nn.Module, dst_model: torch.nn.Module, model_mapping: dict[str, ModuleMapping]
):
    for name, module_mapping in model_mapping.items():
        # print(f"{name}: {module_mapping}")

        src_key, dst_key = module_mapping.get_key_pair()
        src_module = src_model.get_submodule(src_key)
        dst_module = dst_model.get_submodule(dst_key)

        if module_mapping.src_weights is not None:
            for s, d in zip(module_mapping.src_weights, module_mapping.dst_weights):
                src_submod_keys = (
                    module_mapping.src_submod_keys
                    if module_mapping.src_submod_keys is not None
                    else None
                )
                dst_submod_keys = (
                    module_mapping.dst_submod_keys
                    if module_mapping.dst_submod_keys is not None
                    else None
                )

                if src_submod_keys is not None:
                    for src_submod_key, dst_submod_key in zip(src_submod_keys, dst_submod_keys):
                        src_submodule = (
                            src_module.get_submodule(src_submod_key)
                            if src_submod_key is not None
                            else src_module
                        )
                        dst_submodule = (
                            dst_module.get_submodule(dst_submod_key)
                            if dst_submod_key is not None
                            else dst_module
                        )

                        fqn_src = (
                            f"{src_key}.{src_submod_key}.{s}"
                            if src_submod_key is not None
                            else f"{src_key}.{s}"
                        )
                        fqn_dst = (
                            f"{dst_key}.{dst_submod_key}.{d}"
                            if dst_submod_key is not None
                            else f"{dst_key}.{s}"
                        )

                        # print(f"copying from  -> {fqn_src} -> {fqn_dst}")

                        src_weight = getattr(src_submodule, s)
                        dst_weight = getattr(dst_submodule, d)
                        _copy_weight(src_weight, dst_weight, f"{fqn_src} -> {fqn_dst}")
                else:
                    fqn_src = f"{src_key}.{s}"
                    fqn_dst = f"{dst_key}.{s}"

                    # print(f"copying from  -> {fqn_src} -> {fqn_dst}")

                    src_weight = getattr(src_module, s)
                    dst_weight = getattr(dst_module, d)
                    _copy_weight(src_weight, dst_weight, f"{fqn_src} -> {fqn_dst}")


# credit: https://github.com/vllm-project/vllm/blob/d91278181d89686b73b2ec88c2db4d55c6c506cb/vllm/model_executor/model_loader/utils.py#L33
@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


# credit: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/model_loader/weight_utils.py#L675
def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 1234,
    exclude_pat: list[str] = None,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.

    We use per-parameter random seed, so that dummy weights are consistent,
    even if the model is partitioned across multiple devices. When the seed
    is fixed, the random values generated by this function only depends on
    the parameter's number of elements and its data type.
    """
    if isinstance(exclude_pat, str):
        exclude_pat = [exclude_pat]

    for name, param in model.state_dict().items():
        if exclude_pat is not None and any(pat in name for pat in exclude_pat):
            continue
        if torch.is_floating_point(param):
            generator = torch.Generator(device=param.data.device)
            generator.manual_seed(seed)
            if torch.finfo(param.data.dtype).bits < 16:
                # uniform_ doesn't support < 16-bit datatypes (FP8)
                dtype = param.data.dtype
                tmp_param = param.data.to(torch.float16)
                tmp_param = tmp_param.uniform_(low, high, generator=generator).to(dtype)
                param.data.copy_(tmp_param)
            else:
                param.uniform_(low, high, generator=generator)


