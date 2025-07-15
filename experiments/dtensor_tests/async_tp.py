#!/usr/bin/env python
# run with:  torchrun --nproc_per_node=4 demo_fusion_trace.py
import torch, torch.distributed as dist
from torch.distributed.tensor.parallel import (
        parallelize_module, ColwiseParallel)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
import torch._inductor.fx_passes.micro_pipeline_tp as micro_tp
from torch._inductor import compile_fx
from torch.fx import symbolic_trace
import torch
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
import torch._inductor.config as inductor_config

from torch._logging import set_logs
import logging
set_logs(dynamo=logging.DEBUG, inductor=logging.DEBUG, aot_graphs=True, graph_code_verbose=True, distributed=logging.DEBUG, fsdp=logging.DEBUG, dtensor=logging.DEBUG)
inductor_config.trace.enabled = True


class SimpleLinear(torch.nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.linear(x)
    
def print_stage(tag, gm):
    print(f"\n--- {tag} ---")
    gm.graph.print_tabular()
    print()

def print_params(m: torch.nn.Module):
    for name, param in m.named_parameters():
        print(f"{name}: {type(param)} {param.shape}")

USE_FAKE = True

def run_model_parallel(model: torch.nn.Module, inp: torch.tensor, mesh: DeviceMesh, use_local_output: bool = False, compile: bool = True, mode: str = "default"):
    model = parallelize_module(model, mesh,
           {"linear": ColwiseParallel(use_local_output=use_local_output)})

    if compile:
        model = torch.compile(model, mode=mode)
    
    out = model(inp)

    if isinstance(out, DTensor):
        print(f"{type(out)} {out.shape=} {out._local_tensor.shape=}")
    else:
        print(f"{type(out)} {out.shape=}")

def main():
    if USE_FAKE:
        world_size = 4

        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=world_size
        )
    else:
        import os

        world_size = os.getenv("WORLD_SIZE", 1)
        world_size = int(world_size)


    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (world_size,),
        mesh_dim_names=(
            "tp",
        ),
    )    
    
    rank = dist.get_rank()

    hidden_dim = 1024
    seqlen = 128
    compile = True
    mode = "default"

    model = SimpleLinear(hidden_dim)
    local_inp = torch.randn(seqlen, hidden_dim, device="cuda").chunk(world_size, dim=0)[rank]
    dist_inp = DTensor.from_local(local_inp, device_mesh=mesh, placements=[Shard(0)])
    
    print(f"{dist_inp._local_tensor.shape=} {dist_inp.shape=}")

    run_model_parallel(model, dist_inp, mesh=mesh, use_local_output=False, compile=compile, mode=mode)


if __name__ == "__main__":
    main()
