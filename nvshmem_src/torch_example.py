import torch.distributed as dist
import torch
import triton
import triton.language as tl
import nvshmem.core
import os
from cuda.core.experimental import Device, system, Stream
import time

###
#  Helper code from https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/pytorch_example.py
#  Used to extract PyTorch Stream into a cuda.core.Stream for NVSHMEM APIs
###

# Create a wrapper class that implements __cuda_stream__
# Example of using https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol
class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python

def torchrun_uid_init():
    """
    Initialize NVSHMEM using UniqueID with `torchrun` as the launcher
    """
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

    nvshmem.core.init(device=dev, uid=broadcast_objects[0], rank=rank_id, nranks=num_ranks, initializer_method="uid")
    time.sleep(local_rank)
    print(f"rank{rank}: Initialized with uid: {broadcast_objects[0]}")

if __name__ == '__main__':
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ["WORLD_SIZE"])

    torchrun_uid_init()

    import time
    import torch.distributed as dist
    
    n_elements = 4
    tensor_out = nvshmem.core.tensor((n_elements,), dtype=torch.float32)
    tensor_out.fill_(rank)
    time.sleep(rank)
    print(f"rank{rank}: BEFORE reduce {tensor_out.tolist()}")
    
    nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD, tensor_out, tensor_out, "sum", stream=stream)
    torch.cuda.synchronize()
    print(f"rank{rank}: AFTER reduce {tensor_out.tolist()}")
    
    nvshmem.core.free_tensor(tensor_out)
    nvshmem.core.finalize()
    dist.destroy_process_group()
