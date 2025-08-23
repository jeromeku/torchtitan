import torch

import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch._inductor.runtime.triton_compat import tl, triton

from torch.testing._internal.inductor_utils import requires_triton
from contextlib import contextmanager
import os
# So that tests are written in device-agnostic way
device = device_type = "cuda"
device_module = torch.get_device_module(device_type)

def _init_device(device) -> None:
    # TODO: relieve this (seems to hang if without)
    device_module.set_device(device)
    # NOTE: required for nvshmem allocation
    torch.empty(1, device=device)

@contextmanager
def dist_context(rank=None, world_size=None):

    rank = rank or int(os.environ.get("RANK", 0))
    world_size = world_size or int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(rank=rank, world_size=world_size)
    _init_device(rank)

    yield
    dist.destroy_process_group()


def test_nvshmem_all_to_all() -> None:

    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)

    dtype = torch.float
    numel_per_peer = 10
    numel = world_size * numel_per_peer
    inp = symm_mem.empty(numel, dtype=dtype, device=device).fill_(rank)
    out = symm_mem.empty(numel, dtype=dtype, device=device).fill_(-1)

    symm_mem.rendezvous(inp, group=group_name)
    symm_mem.rendezvous(out, group=group_name)
    torch.ops.symm_mem.nvshmem_all_to_all(inp, out, group_name)

    expected = torch.cat(
        [
            torch.empty(numel_per_peer, dtype=dtype, device=device).fill_(i)
            for i in range(world_size)
        ]
    )
    torch.testing.assert_close(out, expected)

def test_nvshmem_all_to_all_vdev() -> None:

    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)

    dtype = torch.float
    # Number of elements for a peer is random between [0, k)
    k = 10
    inp_splits = torch.randint(k, (world_size,), device=device)
    inp_numel = inp_splits.sum().item()
    # Exchange input splits to get output splits
    out_splits = torch.zeros_like(inp_splits)
    dist.all_to_all_single(out_splits, inp_splits)
    out_numel = out_splits.sum().item()

    # Max number of input elements (must be a constant across ranks for symmetric memory allocation)
    max_inp_numel = k * world_size
    # Max number of output elements (must be a constant across ranks for symmetric memory allocation)
    overflow_factor = world_size  # worst case: one rank receives all data
    max_out_numel = max_inp_numel * overflow_factor

    inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=device).fill_(
        rank
    )
    out = symm_mem.empty(max_out_numel, dtype=dtype, device=device).fill_(-1)
    in_out_splits = symm_mem.empty(
        (3, world_size), dtype=torch.int64, device=device
    )
    # Row 0 is input splits
    in_out_splits[0].copy_(inp_splits)

    torch.ops.symm_mem.nvshmem_all_to_all_vdev(inp, out, in_out_splits, group_name)

    # Check input splits (row 0) -- should not change
    torch.testing.assert_close(in_out_splits[0], inp_splits)

    # Check output splits (row 1)
    torch.testing.assert_close(in_out_splits[1], out_splits)

    # Check output offsets (row 2)
    out_offsets = torch.cumsum(out_splits, dim=0)  # inclusive scan
    # output offsets from `nvshmem_all_to_all_vdev` is exclusive scan
    assert in_out_splits[2][0] == 0
    
    torch.testing.assert_close(in_out_splits[2][1:], out_offsets[:-1])

    # Check data
    expected = torch.empty(out_numel, dtype=dtype, device=device)
    dist.all_to_all_single(
        expected, inp[:inp_numel], out_splits.tolist(), inp_splits.tolist()
    )
    torch.testing.assert_close(out[:out_numel], expected)

if __name__ == "__main__":
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    with dist_context(rank, world_size):
        test_nvshmem_all_to_all()