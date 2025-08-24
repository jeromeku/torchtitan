# hello_world_torchrun.py
import os, numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
from nvshmem.core.utils import get_size

import nvshmem.core as nv
import nvshmem
from cuda.core.experimental import Device, system, Stream  # pip install cuda-python
import time

class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python

def get_env():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world, local_rank

def set_device(local_rank: int) -> Device:
    # Prefer cuda-python Device if you have it; else fall back to torch.cuda
    # try:
    dev = Device(local_rank % system.num_devices)
    dev.set_current()
    return dev  # pass this to nv.init(device=...)

def create_arrays(rank: int, shape=(4,), dtype=torch.float32, impl="torch", stream: torch.cuda.Stream = None):
    
    if impl == "torch":
        # src_buf = nvshmem.core.buffer(get_size(shape, dtype))
        # src = torch.utils.dlpack.from_dlpack(src_buf)
        # src = src.view(dtype).view(shape).fill_(0)
        # dst_buf = nvshmem.core.buffer(get_size(shape, dtype))
        # dst = torch.utils.dlpack.from_dlpack(dst_buf).view(dtype).view(shape).fill_(rank + 1)
        assert stream
        with stream:
            src = nv.tensor(shape, dtype)
            dst = nv.tensor(shape, dtype)
            dst[:] = 0
            src[:] = rank + 1
        stream.synchronize()
    else:
        src = nv.array(shape, dtype="float32")
        dst = nv.array(shape, dtype="float32")
        dst[:] = 0
        src[:] = rank + 1

    return src, dst


def torchrun_uid_init():
    """
    Initialize NVSHMEM using UniqueID with `torchrun` as the launcher
    """
    rank, world_size, local_rank = get_env()
    # Set Torch device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # nvshmem4py requires a cuda.core Device at init time
    global dev
    dev = Device(device.index)
    dev.set_current()
    global stream
    # Get PyTorch's current stream
    pt_stream = torch.cuda.current_stream()

    stream = PyTorchStreamWrapper(pt_stream)
    if local_rank == 0:
        print(f"Stream: {stream} {stream.__cuda_stream__()}")

    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=local_rank,
        world_size=world_size,
        device_id=device
    )

    # Extract rank, nranks from process group
    num_ranks = dist.get_world_size()
    rank_id = dist.get_rank()

    # Create an empty uniqueid for all ranks
    uniqueid = nvshmem.core.get_unique_id(empty=True)
    if rank_id == 0:
        # Rank 0 gets a real uniqueid
        uniqueid = nvshmem.core.get_unique_id()
        broadcast_objects = [uniqueid]
    else:
        broadcast_objects = [None]

    # We use torch.distributed.broadcast_object_list to send the UID to all ranks
    dist.broadcast_object_list(broadcast_objects, src=0)
    dist.barrier()
    uid = broadcast_objects[0]
    nvshmem.core.init(device=dev, uid=uid, rank=rank_id, nranks=num_ranks, initializer_method="uid")
    
    time.sleep(local_rank)
    print(f"rank{rank}: Initialized with uid: {uid}")
    return uid
    
def main():
    uid = torchrun_uid_init()
    rank, world_size, local_rank = get_env()

    default_stream = torch.cuda.default_stream()
    stream_handle = default_stream.cuda_stream
    nv_stream = Stream.from_handle(stream_handle)

    # Test
    nv.barrier(nv.Teams.TEAM_WORLD, stream=nv_stream)
    nv_stream.sync()
    time.sleep(rank)
    print(f"[rank {rank + 1}/{world_size}] NVSHMEM init OK on device {rank}", flush=True)

    local_rank_per_node = local_rank
    impl = "torch"
    src, dst = create_arrays(local_rank, impl=impl, stream=default_stream)

    time.sleep(local_rank)
    # Print dst, src before
    print(f"rank{local_rank_per_node}: Dest BEFORE collective from PE {nvshmem.core.my_pe()}:", dst)
    print(f"rank{local_rank_per_node}: Src BEFORE collective from PE {nvshmem.core.my_pe()}:", src)

    # # Perform a sum reduction from arr_src to arr_dst across all PEs in TEAM_WORLD (an AllReduce)
    nv.reduce(nvshmem.core.Teams.TEAM_WORLD, dst, src, "sum", stream=nv_stream)
    nv_stream.sync()
    
    time.sleep(local_rank)

    # # Print dst, src after
    print(f"rank{local_rank_per_node}: Dest AFTER collective from PE {nvshmem.core.my_pe()}:", dst)
    print(f"rank{local_rank_per_node}: Src AFTER collective from PE {nvshmem.core.my_pe()}:", src)

    if impl == "torch":
        nvshmem.core.free_tensor(src)
        nvshmem.core.free_tensor(dst)
    else:        
        nvshmem.core.free_array(src)
        nvshmem.core.free_array(dst)

    # dev.sync()
    nv.finalize()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
