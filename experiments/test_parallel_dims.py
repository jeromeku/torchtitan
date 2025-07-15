from torchtitan.distributed.utils import init_fake_distributed

from torchtitan.distributed.parallel_dims import ParallelDims

from torchtitan.config_manager import ConfigManager

CONFIG_FILE = "/home/jeromeku/torchtitan/llama_debug.toml"
WORLD_SIZE = 8
DP_SHARD = -1
EP = 4


def main():
    cfg_manager = ConfigManager()
    config = cfg_manager.parse_args(
        args=[
            f"--job.config_file={CONFIG_FILE}",
            f"--parallelism.world_size={WORLD_SIZE}",
            f"--parallelism.expert_parallel_degree={EP}",
            f"--parallelism.data_parallel_shard_degree={DP_SHARD}",
        ]
    )
    print(config)

    parallelism_config = config.parallelism

    parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        world_size=WORLD_SIZE,
    )
    print(parallel_dims)


if __name__ == "__main__":
    init_fake_distributed(WORLD_SIZE)
    main()
