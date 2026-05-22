#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""LAV-ACT: Language-Augmented Visual Action Chunking with Transformers

Simplified implementation that always uses language conditioning with Voltron v-cond
and FiLM modulation of ResNet features.
"""

import einops
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict

try:
    from voltron import load as load_voltron

    VOLTRON_AVAILABLE = True
except ImportError:
    VOLTRON_AVAILABLE = False
    print("Warning: Voltron not available. Install with: pip install voltron-robotics")
    raise Exception

from lerobot.common.policies.act.modeling_act import ACTPolicy, ACT
from lerobot.common.policies.act.configuration_lavact import LAVACTConfig
from lerobot.common.constants import ACTION




class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""

    def __init__(self, conditioning_dim: int, feature_dim: int, hidden_dim: int = 512):
        super().__init__()

        self.gamma_net = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.beta_net = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        # Initialize to identity transformation
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, features: Tensor, conditioning: Tensor) -> Tensor:
        """Apply FiLM conditioning: gamma * features + beta"""
        gamma = self.gamma_net(conditioning).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_net(conditioning).unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta


class VoltronEncoder(nn.Module):
    """Voltron v-cond encoder for language and vision."""

    def __init__(self, device: str = "cuda", freeze: bool = True):
        super().__init__()

        if not VOLTRON_AVAILABLE:
            raise ImportError("Voltron is required. Install with: pip install voltron-robotics")

        self.voltron_model, self.preprocess = load_voltron(
            "v-cond", device=device, freeze=freeze
        )

        # Voltron v-cond typically outputs 384 or 512 dim embeddings
        self.embedding_dim = 384
        self.device = device

    def forward(self, images: list[Tensor], language: list[str]) -> Tensor:
        """Encode images and language into multimodal embeddings."""
        batch_size = images[0].shape[0]
        embeddings = []

        # Process each sample in the batch
        for i in range(batch_size):
            # Use first image for v-cond (single frame)
            img = images[0][i]  # (C, H, W)
            lang = [language[i]]

            # Apply Voltron's preprocessing and add batch dimension
            img_preprocessed = self.preprocess(img)[None, ...].to(self.device)  # (1, C, H, W)

            # Get multimodal embedding from Voltron
            with torch.no_grad() if self.voltron_model.training == False else torch.enable_grad():
                embedding = self.voltron_model(img_preprocessed, lang, mode="multimodal")

            # Pool if needed (embedding might be (1, seq_len, dim))
            if embedding.dim() > 2:
                embedding = embedding.mean(dim=1)  # Pool sequence dimension

            embeddings.append(embedding)

        return torch.cat(embeddings, dim=0)  # (B, embedding_dim)


class LAVACT(ACT):
    """Language-Augmented Visual Action Chunking Transformer."""

    def __init__(self, config: LAVACTConfig):
        super().__init__(config)
        self.config = config

        # Initialize Voltron encoder
        self.voltron_encoder = VoltronEncoder(
            device="cuda" if torch.cuda.is_available() else "cpu",
            freeze=config.voltron_freeze
        )

        # FiLM conditioning for ResNet features
        if config.image_features:
            # Get ResNet output channels (typically 2048 for ResNet-50)
            backbone_channels = 512

            self.film_layer = FiLMLayer(
                conditioning_dim=self.voltron_encoder.embedding_dim,
                feature_dim=backbone_channels,
                hidden_dim=config.film_hidden_dim
            )

    def forward(self, batch: Dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """Forward pass with language conditioning."""

        if self.config.use_vae and self.training:
            assert "action" in batch, "Actions required for VAE training"

        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        else:
            batch_size = batch["observation.environment_state"].shape[0]

        # Get language tasks
        language_tasks = batch["task"]
        if isinstance(language_tasks, Tensor):
            # Convert tensor to list of strings if needed
            language_tasks = [str(task) for task in language_tasks.tolist()]

        # VAE encoder processing (same as parent)
        if self.config.use_vae and "action" in batch:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )

            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"])
                robot_state_embed = robot_state_embed.unsqueeze(1)
                vae_encoder_input = [cls_embed, robot_state_embed, self.vae_encoder_action_input_proj(batch["action"])]
            else:
                vae_encoder_input = [cls_embed, self.vae_encoder_action_input_proj(batch["action"])]

            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch["observation.state"].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]

            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim:]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch["observation.state"].device
            )


        # Prepare encoder tokens
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))

        if self.config.env_state_feature:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch["observation.environment_state"])
            )

        # Process images with FiLM conditioning
        if self.config.image_features:
            all_cam_features = []
            all_cam_pos_embeds = []

            for img in batch["observation.images"]:
                # Extract ResNet features
                cam_features = self.backbone(img)["feature_map"]

                # Generate multimodal embeddings with Voltron
                voltron_embeddings = self.voltron_encoder(
                    batch["observation.images"],
                    language_tasks
                )


                # Apply FiLM conditioning with Voltron embeddings
                cam_features = self.film_layer(cam_features, voltron_embeddings)

                # Standard ACT processing
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Reshape for transformer: (B, C, H, W) -> (H*W, B, C)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)

            encoder_in_tokens.extend(torch.cat(all_cam_features, axis=0))
            encoder_in_pos_embed.extend(torch.cat(all_cam_pos_embeds, axis=0))

        # Process through transformer
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )

        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class LAVACTPolicy(ACTPolicy):
    """Language-Augmented Visual Action Chunking Transformer Policy."""

    config_class = LAVACTConfig
    name = "lav_act"

    def __init__(
            self,
            config: LAVACTConfig,
            dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
        self.model = LAVACT(config)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Forward pass expecting task language in batch."""
        assert "task" in batch, "Language task must be provided in batch['task']"
        return super().forward(batch)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Action selection with language task."""
        assert "task" in batch, "Language task must be provided in batch['task']"
        return super().select_action(batch)

    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict action chunk with language task."""
        assert "task" in batch, "Language task must be provided in batch['task']"

        self.eval()
        batch = self.normalize_inputs(batch)

        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions