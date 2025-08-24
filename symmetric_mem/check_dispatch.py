import torch
print(torch._C._dispatch_dump_table("symm_mem::nvshmem_all_to_all_vdev"))
print("CUDA kernel present? ", torch._C._dispatch_has_kernel_for_dispatch_key(
    "symm_mem::nvshmem_all_to_all", "CUDA"))
print("AutogradCUDA present? ", torch._C._dispatch_has_kernel_for_dispatch_key(
    "symm_mem::nvshmem_all_to_all", "AutogradCUDA"))
print("CompositeImplicitAutograd? ", torch._C._dispatch_has_kernel_for_dispatch_key(
    "symm_mem::nvshmem_all_to_all", "CompositeImplicitAutograd"))
