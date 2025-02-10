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

from dataclasses import dataclass, fields
import os
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from nemo.collections.diffusion.sampler.batch_ops import *
from nemo.collections.diffusion.sampler.conditioner import BaseVideoCondition, DataType, Edify4Condition
from nemo.collections.diffusion.sampler.context_parallel import split_inputs_cp, cat_outputs_cp
from nemo.collections.diffusion.sampler.res.res_sampler import COMMON_SOLVER_OPTIONS, RESSampler
from nemo.collections.diffusion.sampler.edm.edm_pipeline import EDMPipeline
from nemo.collections.diffusion.sampler.edm.edm import EDMSDE, EDMSampler, EDMScaling
from nemo.collections.diffusion.sampler.cosmos.cosmos_diffusion_pipeline import CosmosDiffusionPipeline

# key to check if the video data is normalized or image data is converted to video data
# to avoid apply normalization or augment image dimension multiple times
# It is due to we do not have normalization and augment image dimension in the dataloader and move it to the model
IS_PREPROCESSED_KEY = "is_preprocessed"

@dataclass
class BaseVideoCondition:
    crossattn_emb: torch.Tensor
    crossattn_mask: torch.Tensor
    data_type: DataType = DataType.VIDEO
    padding_mask: Optional[torch.Tensor] = None
    fps: Optional[torch.Tensor] = None
    num_frames: Optional[torch.Tensor] = None
    image_size: Optional[torch.Tensor] = None
    scalar_feature: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class VideoExtendCondition(BaseVideoCondition):
    video_cond_bool: Optional[torch.Tensor] = None  # whether or not it conditioned on video
    gt_latent: Optional[torch.Tensor] = None
    condition_video_indicator: Optional[torch.Tensor] = None  # 1 for condition region

    # condition_video_input_mask will concat to the input of network, along channel dim;
    # Will be concat with the input tensor
    condition_video_input_mask: Optional[torch.Tensor] = None
    # condition_video_augment_sigma: (B, T) tensor of sigma value for the conditional input augmentation, only valid when apply_corruption_to_condition_region is "noise_with_sigma" or "noise_with_sigma_fixed"
    condition_video_augment_sigma: Optional[torch.Tensor] = None

class CosmosDiffusionV2WPipeline(CosmosDiffusionPipeline):
    def __init__(self, max_num_latents_condition=2, p_mean_condition=-3, p_std_condition=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_latents_condition = max_num_latents_condition
        self.p_mean_condition=p_mean_condition
        self.p_std_condition=p_std_condition
        
        self.condition_sde = EDMSDE(p_mean=p_mean_condition, p_std=p_std_condition) 

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the diffusion model.

        This method is responsible for executing one iteration of the model's training. It involves:
        1. Adding noise to the input data using the SDE process.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed loss for the training step as a PyTorch Tensor.

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        x0_from_data_batch, x0, condition = self.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        sigma, epsilon = self.draw_training_sigma_and_epsilon(x0.size(), condition)
        
        # enable video conditioning
        condition.video_cond_bool = True
        
        # Sample the perturbation noise levels for the conditional latents
        sigma_condition = self.condition_sde.sample_t(x0.size(0)).to(**self.tensor_kwargs)
        
        # Sample the condition sequence length (1 to 2)
        condition_video_indicator = self.draw_condition_video_indicator(x0.size())
        
        # Add the condition_video_indicator and condition_video_input_mask to the condition object
        self.add_condition_video_indicator_and_video_input_mask(latent_state=x0, condition=condition, condition_video_indicator=condition_video_indicator)
        
        if parallel_state.is_pipeline_last_stage():
            output_batch, kendall_loss, pred_mse, edm_loss = self.compute_loss_with_epsilon_and_sigma(
                data_batch, x0_from_data_batch, x0, condition, epsilon, sigma, sigma_condition, condition_video_indicator
            )
            return output_batch, kendall_loss
        else:
            net_output =  self.compute_loss_with_epsilon_and_sigma(
                data_batch, x0_from_data_batch, x0, condition, epsilon, sigma
            )
            return net_output

    def draw_condition_video_indicator(self, x0_size: torch.Size):
        """
        Sample the number of frames to condition on.
        """
        B = x0_size[0]
        T = x0_size[2]
        condition_video_indicator = torch.zeros(B, 1, T, 1, 1, **self.tensor_kwargs)
        for i in range(B):
            num_latents_condition = torch.randint(1, self.max_num_latents_condition + 1, (1,)).item()
            condition_video_indicator[i, :, :num_latents_condition] = 1.
        
        return  condition_video_indicator.to(**self.tensor_kwargs)
    
    def add_condition_video_indicator_and_video_input_mask(
        self, latent_state: torch.Tensor, condition: VideoExtendCondition, condition_video_indicator: Union[int, None] = None
    ) -> VideoExtendCondition:
        T = latent_state.shape[2]

        condition.gt_latent = latent_state
        condition.condition_video_indicator = condition_video_indicator

        B, C, T, H, W = latent_state.shape
        # Create additional input_mask channel, this will be concatenated to the input of the network
        # See design doc section (Implementation detail A.1 and A.2) for visualization
        ones_padding = torch.ones((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        zeros_padding = torch.zeros((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        assert condition.video_cond_bool is not None, "video_cond_bool should be set"

        # The input mask indicate whether the input is conditional region or not
        if condition.video_cond_bool:  # Condition one given video frames
            condition.condition_video_input_mask = (
                condition_video_indicator * ones_padding + (1 - condition_video_indicator) * zeros_padding
            )
        else:  # Unconditional case, use for cfg
            condition.condition_video_input_mask = zeros_padding

        return condition
        
    def compute_loss_with_epsilon_and_sigma(
        self,
        data_batch: dict[str, torch.Tensor],
        x0_from_data_batch: torch.Tensor,
        x0: torch.Tensor,
        condition: Edify4Condition,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        sigma_condition: torch.Tensor,
        condition_video_indicator: torch.Tensor,
    ):
        # Get the mean and stand deviation of the marginal probability distribution.
        mean, std = self.sde.marginal_prob(x0, sigma)
        # Generate noisy observations with different noise levels depending on the video conditioning
        # Only noise the non image-conditioned latents
        xt = mean + batch_mul(std, epsilon) * (1 - condition_video_indicator)

        if parallel_state.is_pipeline_last_stage():
            # make prediction
            x0_pred, eps_pred, logvar = self.denoise(xt, sigma, condition, sigma_condition)
            # loss weights for different noise levels
            weights_per_sigma = self.get_per_sigma_loss_weights(sigma=sigma)
            # extra weight for each sample, for example, aesthetic weight, camera weight
            weights_per_sample = self.get_per_sample_weight(data_batch, x0.shape[0])
            loss_mask_per_sample = 1.0
            pred_mse = (x0 - x0_pred) ** 2 * loss_mask_per_sample
            edm_loss = batch_mul(pred_mse, weights_per_sigma * weights_per_sample)
            if len(edm_loss.shape) > 5:
                edm_loss = edm_loss.squeeze(0)
            b, c, t, h, w = edm_loss.shape
            if logvar is not None and self.loss_add_logvar:
                kendall_loss = batch_mul(edm_loss, torch.exp(-logvar).view(-1)).flatten(
                    start_dim=1
                ) + logvar.view(-1, 1)
            else:
                kendall_loss = edm_loss.flatten(start_dim=1)
            
            edm_loss = edm_loss * (1. - condition_video_indicator)
            pred_mse = pred_mse * (1. - condition_video_indicator)
            kendall_loss = rearrange(kendall_loss, "b (c t h w) -> b c t h w", b=b, c=c, t=t, h=h, w=w)
            kendall_loss = rearrange(kendall_loss * (1. - condition_video_indicator), "b c t h w -> b c (t h w)", b=b, c=c, t=t, h=h, w=w)
            output_batch = {
                "x0": x0,
                "xt": xt,
                "sigma": sigma,
                "weights_per_sigma": weights_per_sigma,
                "weights_per_sample": weights_per_sample,
                "loss_mask_per_sample": loss_mask_per_sample,
                "condition": condition,
                "model_pred": {"x0_pred": x0_pred, "eps_pred": eps_pred, "logvar": logvar},
                "mse_loss": pred_mse.mean(),
                "edm_loss": edm_loss.mean(),
            }
            return output_batch, kendall_loss, pred_mse, edm_loss
        else:
            # make prediction
            x0_pred = self.denoise(xt, sigma, condition)
            return x0_pred.contiguous()
            
    def denoise(
        self,
        noise_x: Tensor,
        sigma: Tensor,
        condition: VideoExtendCondition,
        condition_video_augment_sigma_in_inference: float = 0.001,
        seed: int = 1,
        is_sample: bool = False,
    ):
        """Denoises input tensor using conditional video generation.

        Args:
            noise_x (Tensor): Noisy input tensor.
            sigma (Tensor): Noise level.
            condition (VideoExtendCondition): Condition for denoising.
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
            seed (int): Random seed for reproducibility
        Returns:
            VideoDenoisePrediction containing:
            - x0: Denoised prediction
            - eps: Noise prediction
            - logvar: Log variance of noise prediction
            - xt: Input before c_in multiplication
            - x0_pred_replaced: x0 prediction with condition regions replaced by ground truth
        """

        assert (
            condition.gt_latent is not None
        ), f"find None gt_latent in condition, like[ly didn't call self.add_condition_video_indicator_and_video_input_mask when preparing the condition or this is a image batch but condition.data_type is wrong, get {noise_x.shape}"
        gt_latent = condition.gt_latent
        # cfg_video_cond_bool = self.conditioner.video_cond_bool #unneeded

        condition_latent = gt_latent

        # Augment the latent with different sigma value, and add the augment_sigma to the condition object if needed
        condition, augment_latent = self.augment_conditional_latent_frames(
            condition, condition_latent, condition_video_augment_sigma_in_inference, sigma, seed, is_sample
        )
        condition_video_indicator = condition.condition_video_indicator  # [B, 1, T, 1, 1]

        # Compose the model input with condition region (augment_latent) and generation region (noise_x)
        new_noise_xt = condition_video_indicator * augment_latent + (1 - condition_video_indicator) * noise_x
        # Call the abse model
        if not parallel_state.is_pipeline_last_stage():
            net_output = super().denoise(new_noise_xt, sigma, condition)
            return net_output
        else:
            x0_pred, eps_pred, logvar = super().denoise(new_noise_xt, sigma, condition)
            
        x0_pred_replaced = condition_video_indicator * gt_latent + (1 - condition_video_indicator) * x0_pred

        x0_pred = x0_pred_replaced
        return x0_pred, eps_pred, logvar
    
    def augment_conditional_latent_frames(
        self,
        condition: VideoExtendCondition,
        # cfg_video_cond_bool: VideoCondBoolConfig,
        gt_latent: Tensor,
        condition_video_augment_sigma_in_inference: float = 0.001,
        sigma: Tensor = None,
        seed: int = 1,
        is_sample: bool = False,
    ) -> Union[VideoExtendCondition, Tensor]:
        """Augments the conditional frames with noise during inference.

        Args:
            condition (VideoExtendCondition): condition object
                condition_video_indicator: binary tensor indicating the region is condition(value=1) or generation(value=0). Bx1xTx1x1 tensor.
                condition_video_input_mask: input mask for the network input, indicating the condition region. B,1,T,H,W tensor. will be concat with the input for the network.
            cfg_video_cond_bool (VideoCondBoolConfig): video condition bool config
            gt_latent (Tensor): ground truth latent tensor in shape B,C,T,H,W
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
            sigma (Tensor): noise level for the generation region
            seed (int): random seed for reproducibility
        Returns:
            VideoExtendCondition: updated condition object
                condition_video_augment_sigma: sigma for the condition region, feed to the network
            augment_latent (Tensor): augmented latent tensor in shape B,C,T,H,W

        """

        # Inference only, use fixed sigma for the condition region
        assert (
            condition_video_augment_sigma_in_inference is not None
        ), "condition_video_augment_sigma_in_inference should be provided"
        augment_sigma = condition_video_augment_sigma_in_inference

        if is_sample and augment_sigma >= sigma.flatten()[0]:
            # This is a inference trick! If the sampling sigma is smaller than the augment sigma, we will start denoising the condition region together.
            # This is achieved by setting all region as `generation`, i.e. value=0
            # log.debug("augment_sigma larger than sigma or other frame, remove condition")
            condition.condition_video_indicator = condition.condition_video_indicator * 0

        augment_sigma = torch.tensor([augment_sigma], **self.tensor_kwargs)

        # Now apply the augment_sigma to the gt_latent

        noise = torch.randn(
            *gt_latent.shape,
            dtype=torch.float32,
            device=self.tensor_kwargs["device"],
            generator=torch.Generator(device=self.tensor_kwargs['device']).manual_seed(seed),
        )

        augment_latent = gt_latent + noise * augment_sigma[:, None, None, None, None]

        _, _, c_in_augment, _ = self.scaling(sigma=augment_sigma)

        # Multiply the whole latent with c_in_augment
        augment_latent_cin = batch_mul(augment_latent, c_in_augment)

        # Since the whole latent will multiply with c_in later, we devide the value to cancel the effect
        _, _, c_in, _ = self.scaling(sigma=sigma)
        augment_latent_cin = batch_mul(augment_latent_cin, 1 / c_in)

        return condition, augment_latent_cin    

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor):
        """
        Args:
            sigma (tensor): noise level

        Returns:
            loss weights per sigma noise level
        """
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def get_x0_fn_from_batch_with_condition_latent(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        seed: int = 1,
    ) -> Callable:
        """Creates denoising function for conditional video generation.

        Args:
            data_batch: Input data dictionary
            guidance: Classifier-free guidance scale
            is_negative_prompt: Whether to use negative prompting
            condition_latent: Conditioning frames tensor (B,C,T,H,W)
            num_condition_t: Number of frames to condition on
            condition_video_augment_sigma_in_inference: Noise level for condition augmentation
            add_input_frames_guidance: Whether to apply guidance to input frames
            seed: Random seed for reproducibility

        Returns:
            Function that takes noisy input and noise level and returns denoised prediction
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t
        )

        uncondition.video_cond_bool = False if add_input_frames_guidance else True
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, uncondition, num_condition_t
        )

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(
                noise_x,
                sigma,
                condition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                seed=seed,
                is_sample=True
            ).x0_pred_replaced
            uncond_x0 = self.denoise(
                noise_x,
                sigma,
                uncondition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                seed=seed,
                is_sample=True
            ).x0_pred_replaced

            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab"
    ) -> Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        """

        is_image_batch = self.is_image_batch(data_batch)
        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            if is_image_batch:
                state_shape = (self.state_shape[0], 1, *self.state_shape[2:])  # C,T,H,W

        cp_enabled = parallel_state.get_context_parallel_world_size() > 1

        if self._noise_generator is None:
            self._initialize_generators()

        x0_fn = self.get_x0_fn_from_batch_with_condition_latent(data_batch, guidance, is_negative_prompt=is_negative_prompt)
        
        state_shape = list(state_shape)
        
        np.random.seed(self.seed)
        x_sigma_max = (
            torch.from_numpy(np.random.randn(1, *state_shape).astype(np.float32)).to(
                dtype=torch.float32, device=self.tensor_kwargs["device"]
            )
            * self.sde.sigma_max
        )

        if cp_enabled:
            cp_group = parallel_state.get_context_parallel_group()
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=cp_group)

        if self.sampler_type == "EDM":
            samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=self.sde.sigma_max)
        elif self.sampler_type == "RES":
            samples = self.sampler(x0_fn, x_sigma_max, sigma_max=self.sde.sigma_max, num_steps=num_steps, solver_option=solver_option)

        if cp_enabled:
            cp_group = parallel_state.get_context_parallel_group()
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=cp_group)

        return samples
    