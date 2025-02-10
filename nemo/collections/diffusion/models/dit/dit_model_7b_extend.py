# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import Tensor
from typing import Dict, Literal, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.transformer_block import TransformerBlock

from megatron.core.models.dit.dit_layer_spec  import (
    AdaLN,
    get_dit_adaln_block_with_transformer_engine_spec as DiTLayerWithAdaLNspec,
)
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from megatron.core import parallel_state
from megatron.core.utils import make_sharded_tensor_for_checkpoint

import torch.distributed as dist
from torch.autograd import Function

from torch.distributed import ProcessGroup, get_process_group_ranks
import math

from torch import Tensor
from torch.distributed import ProcessGroup, all_gather, get_process_group_ranks, get_world_size

from nemo.collections.diffusion.models.dit.dit_model_7b import DiTCrossAttentionModel7B


class DiTCrossAttentionModel7BExtend(DiTCrossAttentionModel7B):
    def __init__(self, *args, in_channels=16 + 1, add_augment_sigma_embedding=False, **kwargs):
        self.add_augment_sigma_embedding = add_augment_sigma_embedding

        # extra channel for video condition mask
        super().__init__(*args, in_channels=in_channels, **kwargs)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        crossattn_emb: Tensor,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        pos_ids: Tensor = None,
        **kwargs,
    ) -> Tensor:
        assert 'condition_video_input_mask' in kwargs, 'condition_video_input_mask is required'
        condition_video_input_mask = kwargs['condition_video_input_mask']
        B, C, T, H, W = x.shape
        assert condition_video_input_mask.shape == (B, 1, T, H, W)
        
        x = torch.cat([x, condition_video_input_mask], dim=1)
        
        return super().forward(x, timesteps, crossattn_emb, inference_params, packed_seq_params, pos_ids, **kwargs)
