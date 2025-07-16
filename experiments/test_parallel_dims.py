from torchtitan.distributed.utils import init_fake_distributed

from torchtitan.distributed.parallel_dims import ParallelDims
import torch
from torchtitan.config_manager import ConfigManager
from torchtitan.protocols import train_spec as train_spec_module

CONFIG_FILE = "/home/jeromeku/torchtitan/llama_debug.toml"
WORLD_SIZE = 8
DP_SHARD = -1
EP = 4


def main():
    cfg_manager = ConfigManager()
    job_config = cfg_manager.parse_args(
        args=[
            f"--job.config_file={CONFIG_FILE}",
            f"--parallelism.world_size={WORLD_SIZE}",
            f"--parallelism.expert_parallel_degree={EP}",
            f"--parallelism.data_parallel_shard_degree={DP_SHARD}",
        ]
    )
    # print(config)

    parallelism_config = job_config.parallelism

    parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        world_size=WORLD_SIZE,
    )
    mesh = parallel_dims.build_mesh()

    # print(f"{mesh._dim_group_names=}")
    # print(f"{mesh.mesh_dim_names=}")
    # non_moe_dp_mesh_dim_names = ("dp_shard_cp",)
    # ep_dp_mesh_dim_names = ("dp_shard_mod_ep",)
    # non_moe_dp_mesh = mesh[non_moe_dp_mesh_dim_names]
    # ep_dp_mesh = mesh[ep_dp_mesh_dim_names]
    # print(f"{non_moe_dp_mesh_dim_names}: {non_moe_dp_mesh}")
    # print(f"{ep_dp_mesh_dim_names}: {ep_dp_mesh}")

    train_spec = train_spec_module.get_train_spec(job_config.model.name)
    tokenizer = (
        train_spec.build_tokenizer_fn(job_config)
        if train_spec.build_tokenizer_fn is not None
        else None
    )
    model_args = train_spec.model_args[job_config.model.flavor]
    # set the model args from training job configs
    model_args.update_from_config(job_config, tokenizer)
    print(model_args)

    with torch.device("meta"):
        model = train_spec.model_cls(model_args)

    print(model)


if __name__ == "__main__":
    init_fake_distributed(WORLD_SIZE)
    main()
