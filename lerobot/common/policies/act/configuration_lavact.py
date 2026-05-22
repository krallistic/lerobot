from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.common.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("lavact")
@dataclass
class LAVACTConfig(ACTConfig):
    """Configuration for LAV-ACT with language conditioning."""

    voltron_model:str = "v-cond"
    film_hidden_dim:int = 512
    voltron_freeze:bool = True
