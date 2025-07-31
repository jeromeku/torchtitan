import pytest
from transformers.models.qwen3_moe.configuration_qwen3_moe import (
    Qwen3MoeConfig as HFQwen3MoeConfig,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel as HFQwen3MoeModel

from torchtitan.experiments.qwen3_moe.model.args import (
    Qwen3MoeConfig,
)

from torchtitan.experiments.qwen3_moe.model.model import Qwen3MoeModel
from torchtitan.tools.initialization import init_on_device
import torch

@pytest.fixture
def device_meta():
    with init_on_device("meta", include_buffers=False):
        yield


@pytest.fixture(
    params=[torch.float32, torch.float16, torch.bfloat16], ids=lambda d: str(d).split(".")[-1]
)
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param

@pytest.fixture
def hf_moe_config() -> HFQwen3MoeConfig:
    return HFQwen3MoeConfig(num_hidden_layers=1)


@pytest.fixture
def qwen3_moe_config(hf_moe_config: Qwen3MoeConfig) -> Qwen3MoeConfig:
    return Qwen3MoeConfig.from_hf(hf_moe_config)


"""
NOTE
Initialize larger modules on 'meta' device for faster testing.
 - Tests that use only submodules of the model then only need to initialize those submodules.

 See 
 - `torchtune.tests.conftest.py` the 'meta' device context.
 - `torchtune.testing.utils.init_module` for module initialization logic.
"""

@pytest.fixture
def hf_moe_model(hf_moe_config: HFQwen3MoeConfig, device_meta) -> HFQwen3MoeModel:
    return HFQwen3MoeModel(hf_moe_config)


@pytest.fixture
def tt_moe_model(qwen3_moe_config: Qwen3MoeConfig, device_meta):
    return Qwen3MoeModel(qwen3_moe_config)