
from torch._C._distributed_c10d import _detect_dma_connectivity
from torch._C._autograd import DeviceType
import torch
import torch.distributed as dist

import time
dist.init_process_group()

def dist_print(*msg, delay_factor: float = 1.):
    delay = dist.get_rank() * delay_factor
    time.sleep(delay)
    print(f"rank{rank}:", *msg, flush=True)

rank = dist.get_rank()
connectivity = _detect_dma_connectivity(DeviceType.CUDA, "nvlink")
for prop in ["device_type", "connection_type", "matrix"]:
    dist_print(prop, getattr(connectivity, prop))

# self.assertEqual(connectivity.device_type, DeviceType.CUDA)
# self.assertEqual(connectivity.connection_type, "nvlink")
# self.assertEqual(len(connectivity.matrix), torch.cuda.device_count())
# for row in connectivity.matrix:
#     self.assertEqual(len(row), torch.cuda.device_count())
