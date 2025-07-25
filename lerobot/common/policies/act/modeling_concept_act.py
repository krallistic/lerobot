from lerobot.common.policies.act.modeling_act import ACTPolicy, ACT, ACTDecoder, ACTEncoder, get_activation_fn
from lerobot.common.policies.act.configuration_concept_act import ConceptACTConfig
import torch
from torch import Tensor, nn
import einops
import torch.nn.functional as F  # noqa: N812
from typing import Tuple
import math

import torch
import torch.nn as nn
from typing import Callable


class RBFLayer(nn.Module):
    """
    Defines a Radial Basis Function Layer that handles arbitrary batch dimensions

    An RBF network output is given by:
    y(x) = sum_{i=1}^N w_i * phi(eps_i * ||x - c_i||)

    Parameters
    ----------
        in_features_dim: int
            Dimensionality of the input features
        num_kernels: int
            Number of RBF kernels to use
        out_features_dim: int
            Dimensionality of the output features
        rbf_type: str
            Type of RBF kernel function. Options: 'gaussian', 'multiquadric',
            'inverse_quadratic', 'thin_plate_spline'
        norm_function: Callable[[torch.Tensor], torch.Tensor], optional
            Norm function for distance computation (default: L2 norm)
        normalization: bool, optional
            If True, normalizes RBF outputs to sum to 1 (default: True)
    """

    # Available RBF functions
    RBF_FUNCTIONS = {
        'gaussian': lambda r: torch.exp(-r ** 2),
        'multiquadric': lambda r: 1.0 / torch.sqrt(1 + r ** 2),
        'inverse_quadratic': lambda r: 1.0 / (1 + r ** 2),
        'thin_plate_spline': lambda r: torch.where(r > 0, r ** 2 * torch.log(r + 1e-10), torch.zeros_like(r))
    }

    def __init__(self,
                 in_features_dim: int,
                 num_kernels: int,
                 out_features_dim: int,
                 rbf_type: str = 'gaussian',
                 norm_function: Callable[[torch.Tensor], torch.Tensor] = None,
                 normalization: bool = True):
        super(RBFLayer, self).__init__()

        self.in_features_dim = in_features_dim
        self.num_kernels = num_kernels
        self.out_features_dim = out_features_dim
        self.rbf_type = rbf_type
        self.normalization = normalization

        # Set RBF function based on string parameter
        if rbf_type not in self.RBF_FUNCTIONS:
            raise ValueError(f"Unknown RBF type: {rbf_type}. Available options: {list(self.RBF_FUNCTIONS.keys())}")
        self.radial_function = self.RBF_FUNCTIONS[rbf_type]

        # Default to L2 norm if none provided
        if norm_function is None:
            self.norm_function = lambda x: torch.norm(x, dim=-1)
        else:
            self.norm_function = norm_function

        # Initialize parameters
        self.kernels_centers = nn.Parameter(
            torch.zeros(num_kernels, in_features_dim, dtype=torch.float32))

        self.log_shapes = nn.Parameter(
            torch.zeros(num_kernels, dtype=torch.float32))

        self.weights = nn.Parameter(
            torch.zeros(out_features_dim, num_kernels, dtype=torch.float32))

        self.reset()

    def reset(self,
              upper_bound_kernels: float = 1.0,
              std_shapes: float = 0.1,
              gain_weights: float = 1.0) -> None:
        """
        Resets all parameters to random initial values

        Parameters
        ----------
            upper_bound_kernels: float
                Range for uniform initialization of kernel centers [-x, x]
            std_shapes: float
                Standard deviation for normal initialization of log-shape parameters
            gain_weights: float
                Gain for Xavier uniform initialization of weights
        """
        nn.init.uniform_(self.kernels_centers,
                         a=-upper_bound_kernels,
                         b=upper_bound_kernels)
        nn.init.normal_(self.log_shapes, mean=0.0, std=std_shapes)
        nn.init.xavier_uniform_(self.weights, gain=gain_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass supporting arbitrary leading batch dimensions

        Parameters
        ----------
            input: torch.Tensor
                Input tensor of shape (..., in_features_dim) where ...
                represents any number of leading dimensions

        Returns
        -------
            out: torch.Tensor
                Output tensor of shape (..., out_features_dim)
        """
        # Store original shape for later reshaping
        original_shape = input.shape[:-1]  # All dims except last
        feature_dim = input.shape[-1]

        assert feature_dim == self.in_features_dim, \
            f"Expected input feature dim {self.in_features_dim}, got {feature_dim}"

        # Flatten leading dimensions: (..., in_features) -> (N, in_features)
        input_flat = input.view(-1, self.in_features_dim)
        batch_size = input_flat.size(0)

        # Compute differences from centers
        # input_flat: (N, in_features_dim)
        # kernels_centers: (num_kernels, in_features_dim)
        input_expanded = input_flat.unsqueeze(1)  # (N, 1, in_features_dim)
        centers_expanded = self.kernels_centers.unsqueeze(0)  # (1, num_kernels, in_features_dim)

        diff = input_expanded - centers_expanded  # (N, num_kernels, in_features_dim)

        # Apply norm function: (N, num_kernels, in_features_dim) -> (N, num_kernels)
        r = self.norm_function(diff)

        # Apply shape parameters: (N, num_kernels)
        shapes = self.log_shapes.exp()  # (num_kernels,)
        eps_r = shapes.unsqueeze(0) * r  # (1, num_kernels) * (N, num_kernels)

        # Apply radial basis function: (N, num_kernels)
        rbfs = self.radial_function(eps_r)

        # Apply normalization if requested
        if self.normalization:
            # Prevent division by zero
            rbf_sums = rbfs.sum(dim=-1, keepdim=True) + 1e-9  # (N, 1)
            rbfs = rbfs / rbf_sums  # (N, num_kernels)

        # Linear combination with weights
        # weights: (out_features_dim, num_kernels)
        # rbfs: (N, num_kernels)
        out_flat = torch.matmul(rbfs, self.weights.t())  # (N, out_features_dim)

        # Reshape back to original leading dimensions
        output_shape = original_shape + (self.out_features_dim,)
        out = out_flat.view(output_shape)

        return out

    @property
    def get_kernels_centers(self):
        """Returns the centers of the kernels"""
        return self.kernels_centers.detach()

    @property
    def get_weights(self):
        """Returns the linear combination weights"""
        return self.weights.detach()

    @property
    def get_shapes(self):
        """Returns the shape parameters"""
        return self.log_shapes.detach().exp()

class RBFActivation(nn.Module):
    def __init__(self, input_dim, n_centers=None, rbf_type='gaussian'):
        super().__init__()
        # Default: use input_dim // 4 centers (more reasonable compression)
        if n_centers is None:
            n_centers = max(1, input_dim // 4)

        self.input_dim = input_dim
        self.n_centers = n_centers
        self.centers = nn.Parameter(torch.randn(n_centers, input_dim))
        self.gamma = nn.Parameter(torch.ones(n_centers) * 1.5)  # Even wider
        self.rbf_type = rbf_type

    def forward(self, x):
        # x can be any shape (..., input_dim)
        # We want output (..., n_centers)

        original_shape = x.shape[:-1]  # All dimensions except last
        feature_dim = x.shape[-1]

        # Flatten all leading dimensions
        x_flat = x.view(-1, feature_dim)  # (N, input_dim) where N = prod(leading_dims)

        # Expand for distance computation
        x_expanded = x_flat.unsqueeze(1)  # (N, 1, input_dim)
        centers_expanded = self.centers.unsqueeze(0)  # (1, n_centers, input_dim)

        # Compute distances
        distances = torch.norm(x_expanded - centers_expanded, dim=-1)  # (N, n_centers)

        # Apply RBF kernel
        if self.rbf_type == 'gaussian':
            activations = torch.exp(-self.gamma * distances ** 2)
        elif self.rbf_type == 'multiquadric':
            activations = 1.0 / torch.sqrt(1 + self.gamma * distances ** 2)

        # Reshape back to original leading dimensions + n_centers
        output_shape = original_shape + (self.n_centers,)
        activations = activations.view(output_shape)

        return activations  # (..., n_centers)


class ConceptACTPolicy(ACTPolicy):
    """ACT Policy with concept learning capabilities.
    
    This class extends the standard ACTPolicy by adding concept learning,
    which can help the model develop better representations of the task.
    """
    
    type = "concept_act"  # This identifies the policy type in the factory

    def __init__(
        self,
        config: ConceptACTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        # Make sure concept learning is enabled in config
        config.use_concept_learning = True
        
        # Initialize parent class
        super().__init__(config=config, dataset_stats=dataset_stats)
        self.config = config
        # Replace the standard ACT model with our ConceptACT model
        self.model = ConceptACT(config)
        #self.model = torch.compile(model=ConceptACT(config))

        # Set up concept normalization if concept learning is enabled
        if "concept" in dataset_stats:
            self.normalize_concepts = nn.ModuleDict({
                "concept": self.normalize_inputs.normalization_modules["observation.concept"]
            })
            self.unnormalize_concepts = nn.ModuleDict({
                "concept": self.unnormalize_outputs.unnormalization_modules["observation.concept"]
            })

        self.reset()

    def get_optim_params(self) -> list[dict]:
        rbf_params = []
        concept_params = []
        other_params = []

        # Define your module types
        CONCEPT_MODULE_TYPES = (
            ConceptACTEncoderLayer,
            # Add your actual concept module classes here
        )

        RBF_MODULE_TYPES = (RBFLayer,)

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if isinstance(module, RBF_MODULE_TYPES):
                    rbf_params.append(param)
                elif isinstance(module, CONCEPT_MODULE_TYPES):
                    concept_params.append(param)
                else:
                    other_params.append(param)

        return [
            {'params': other_params, 'lr': self.config.optimizer_lr},
            {'params': concept_params, 'lr': self.config.optimizer_lr * 10.0},
            {'params': rbf_params, 'lr': self.config.optimizer_lr * 10000.0}
        ]


    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        batch = self.normalize_targets(batch)
        
        # Call the model with different outputs based on whether concept learning is enabled
        if self.config.use_concept_learning:
            actions_hat, (mu_hat, log_sigma_x2_hat), concepts_hat = self.model(batch)
        else:
            actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
            
        # Calculate action reconstruction loss - common for all configurations
        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()
        
        # Initialize loss dictionary with L1 loss
        loss_dict = {"l1_loss": l1_loss.item()}
        total_loss = l1_loss
        
        # Add concept loss if concept learning is enabled
        if self.config.use_concept_learning and self.config.concept_method == "prediction_head":
            concept_loss = 0.0

            # Calculate cross-entropy loss for each concept type defined in config
            for concept_type in self.config.concept_types.keys():
                if concept_type in batch and concept_type in concepts_hat:
                    # Get the predicted logits and true one-hot encoding
                    logits = concepts_hat[concept_type]
                    targets = batch[concept_type]

                    # Calculate cross-entropy loss (assuming targets are one-hot encoded)
                    # For one-hot targets, we need to convert to class indices for cross-entropy
                    if targets.dim() > 1 and targets.shape[-1] > 1:  # One-hot encoded
                        targets = targets.argmax(dim=-1)

                    # Add cross-entropy loss for this concept type
                    ce_loss = F.cross_entropy(logits, targets)
                    concept_loss += ce_loss
                    loss_dict[f"{concept_type}_loss"] = ce_loss.item()

            # Add the total concept loss to the overall loss
            loss_dict["concept_loss"] = concept_loss.item()
            total_loss = total_loss + concept_loss * self.config.concept_weight

        elif self.config.use_concept_learning and self.config.concept_method == "transformer_ce":
            # TODO add NORM loss
            start = 0
            concept_loss = 0.0

            for concept_class in self.config.concept_types.keys():
                targets = batch[concept_class]

                current_concept_class_size = targets.shape[-1]
                current_concept_class_hat = concepts_hat[:, start:start+current_concept_class_size]

                # Calculate cross-entropy loss (assuming targets are one-hot encoded)
                # For one-hot targets, we need to convert to class indices for cross-entropy
                if targets.dim() > 1 and targets.shape[-1] > 1:  # One-hot encoded
                    targets = targets.argmax(dim=-1)
                ce_loss = F.cross_entropy(current_concept_class_hat,  targets)
                concept_loss += ce_loss

                start = start + current_concept_class_size

            loss_dict["concept_loss"] = concept_loss.item()
            total_loss = total_loss + concept_loss * self.config.concept_weight
        elif self.config.use_concept_learning and self.config.concept_method == "transformer_bce":
            # Concatenate all targets into a single multi-label tensor
            all_targets = torch.concatenate([batch[key] for key, value in sorted(self.config.concept_types.items())], dim=-1).float()


            # Calculate pos_weight dynamically
            number_of_positive = len(self.config.concept_types.keys())
            number_of_negative = all_targets.shape[-1] - number_of_positive
            pos_weight_value = number_of_negative / number_of_positive
            pos_weight = torch.tensor([pos_weight_value] * all_targets.shape[-1], device=all_targets.device)

            # DEBUG: Let's see what the raw logits look like
            #with torch.no_grad():
            #    print(f"Raw logits stats: mean={concepts_hat.mean().item():.3f}, std={concepts_hat.std().item():.3f}")
            #    print(f"Raw logits range: min={concepts_hat.min().item():.3f}, max={concepts_hat.max().item():.3f}")
            #    print(f"Target sum per batch: {all_targets.sum(dim=1)}")  # Should be [3, 3, 3, ...] for each batch

            # Apply BCE loss with positive weighting
            concept_loss = F.binary_cross_entropy_with_logits(
                concepts_hat,
                all_targets.float(),
                pos_weight=pos_weight
            )

            # Add detailed logging
            with torch.no_grad():
                probs = torch.sigmoid(concepts_hat)
                predictions = (probs > 0.5).float()

                # Calculate metrics
                true_positives = (predictions * all_targets).sum()
                predicted_positives = predictions.sum()
                actual_positives = all_targets.sum()
                total_elements = all_targets.numel()
                correct_predictions = (predictions == all_targets).float().sum()

                # Avoid division by zero
                precision = (true_positives / predicted_positives).item() if predicted_positives > 0 else 0.0
                recall = (true_positives / actual_positives).item() if actual_positives > 0 else 0.0
                f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                accuracy = (correct_predictions / total_elements).item()

                # Probability statistics
                mean_prob_positive = probs[all_targets == 1].mean().item() if (all_targets == 1).any() else 0.0
                mean_prob_negative = probs[all_targets == 0].mean().item() if (all_targets == 0).any() else 0.0

                # Update loss dictionary with detailed metrics
                loss_dict.update({
                    "concept_loss": concept_loss.item(),
                    "mean_concept_hat": concepts_hat.mean().item(),
                    "mean_concept_target": all_targets.mean().item(),
                    "sum_targets_per_batch": all_targets.sum().item(),
                    "concept_precision": precision,
                    "concept_recall": recall,
                    "concept_f1": f1_score,
                    "concept_accuracy": accuracy,
                    "mean_prob_positive": mean_prob_positive,
                    "mean_prob_negative": mean_prob_negative,
                    "num_predicted_positive": predicted_positives.item(),
                    "num_actual_positive": actual_positives.item(),
                    "num_true_positive": true_positives.item(),
                    "pos_weight_used": pos_weight_value
                })
            total_loss = total_loss + concept_loss * self.config.concept_weight
        elif self.config.use_concept_learning and self.config.concept_method == "transformer":
            # For transformer method, keep the existing MSE loss implementation
            # TODO add NORM loss

            target_concepts = torch.concatenate([batch[key] for key, value in sorted(self.config.concept_types.items())], dim=-1).float()
            # Calculate Frobenius norm (equivalent to MSE loss over the matrices)
            # ||A - H||^2_F
            concept_loss = F.mse_loss(concepts_hat, target_concepts, reduction='mean')
            loss_dict["concept_loss"] = concept_loss.item()
            total_loss = total_loss + concept_loss * self.config.concept_weight
        elif self.config.use_concept_learning:
            # Add zero concept loss if concept learning is enabled but no concepts are provided
            loss_dict["concept_loss"] = 0.0
        
        # Add KL divergence loss if VAE is enabled
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal)
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            total_loss = total_loss + mean_kld * self.config.kl_weight

        return total_loss, loss_dict

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        # Concept learning has no impact on action selection, as we only use the action outputs
        if self.config.use_concept_learning:
            actions, _, _ = self.model(batch)
        else:
            actions, _ = self.model(batch)

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            actions = actions[0]  # (batch_size, chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = actions[:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()




class ConceptACT(ACT):
    """Action Chunking Transformer with concept learning capabilities.
    
    This class extends the ACT model to include concept learning in two ways:
    1. Using prediction heads on the encoder output for each concept type ("prediction_head" method)
    2. Using a separate transformer for concept learning ("transformer" method)
    """
    def __init__(self, config: ConceptACTConfig):
        super().__init__(config)
        self.config = config
        self.encoder = ConceptACTEncoder(config)
        



    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None], dict[str, Tensor] | None]:
        """A forward pass through the Action Chunking Transformer with concept learning.

        Args:
            batch: Dictionary containing observation data, and optionally action data during training.

        Returns:
            actions: (B, chunk_size, action_dim) batch of action sequences
            latent_params: Tuple containing the latent PDF's parameters (mean, log(σ²)) 
                           both as (B, L) tensors where L is the latent dimension.
            concepts: Dictionary of predicted concepts with one-hot encodings for each concept type if concept learning is enabled and prediction head mode,
                      (B, num_concepts) if Concept Transformer is enables
                     or None if concept learning is disabled.
        """
        # -------------------- 1. Process inputs and prepare batch --------------------
        if self.config.use_vae and self.training:
            assert "action" in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        # Determine batch size from available inputs
        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        else:
            batch_size = batch["observation.environment_state"].shape[0]

        # -------------------- 2. VAE Encoding (if applicable) --------------------
        # Process the VAE encoding path if using VAE and we have actions (typically during training)
        if self.config.use_vae and "action" in batch:
            # Prepare VAE encoder inputs
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            
            # Add robot state if available
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
                vae_encoder_input = [cls_embed, robot_state_embed, self.vae_encoder_action_input_proj(batch["action"])]
            else:
                vae_encoder_input = [cls_embed, self.vae_encoder_action_input_proj(batch["action"])]
            
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)
            
            # Prepare positional embeddings and padding mask
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch["observation.state"].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
            
            # Forward pass through VAE encoder
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            
            # Get latent distribution parameters
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]  # 2log(sigma)
            
            # Sample latent with reparameterization trick
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, use a zero latent
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch["observation.state"].device
            )

        # -------------------- 3. Prepare transformer encoder inputs --------------------
        # Prepare token sequence for the encoder: [latent, (robot_state), (env_state), (image_features)]
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        
        # Add robot state token if available
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
            
        # Add environment state token if available
        if self.config.env_state_feature:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch["observation.environment_state"])
            )

        # Add camera features and positional embeddings if available
        if self.config.image_features:
            all_cam_features = []
            all_cam_pos_embeds = []

            for img in batch["observation.images"]:
                # Extract features from backbone
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Reshape to sequence format
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)

            # Add all camera features to the token sequence
            encoder_in_tokens.extend(torch.cat(all_cam_features, axis=0))
            encoder_in_pos_embed.extend(torch.cat(all_cam_pos_embeds, axis=0))

        # Stack tokens along the sequence dimension
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)


        # -------------------- 4. Forward pass through transformer --------------------

        if self.config.use_concept_learning:
            encoder_out, concepts = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        else:
            encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        # 5 was concept prediction, now in 4

        # -------------------- 6. Action prediction --------------------
        # Prepare decoder input (zeros initialized)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        
        # Run decoder with cross-attention to encoder output
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move to batch-first format and predict actions
        decoder_out = decoder_out.transpose(0, 1)  # (B, S, C)
        actions = self.action_head(decoder_out)  # (B, S, action_dim)

        # Return appropriate outputs based on configuration
        if self.config.use_concept_learning:
            return actions, (mu, log_sigma_x2), concepts
        else:
            return actions, (mu, log_sigma_x2)

class ConceptACTEncoderLayer(nn.Module):
    """
    Encoder layer that incorporates concept learning via a parallel pathway.
    Uses the entire input sequence as query for concept attention.
    """
    def __init__(self, config):
        super().__init__()
        # Standard self-attention as in ACTEncoderLayer
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads * 2, dropout=config.dropout)

        # Concept learning components

        self.n_concepts = sum([value for key, value in config.concept_types.items()])

        # Learnable concepts
        self.concepts = nn.Parameter(torch.zeros(1, self.n_concepts, config.dim_model))
        nn.init.trunc_normal_(self.concepts, std=1.0 / math.sqrt(config.dim_model))
        self.number_of_concept_heads = config.n_heads * 2
        # Cross-attention for concept pathway
        self.concept_attn = nn.MultiheadAttention(
            config.dim_model, self.number_of_concept_heads, dropout=config.dropout,
        )

        # Projection layer after concatenation
        self.concept_proj = nn.Linear(config.dim_model * 2, config.dim_model)
        if config.use_rbf_head_selection:
            self.head_weight_predictor = nn.Sequential(
                nn.Linear(config.dim_model, self.number_of_concept_heads * 4),
                #RBFActivation(self.number_of_concept_heads * 4, n_centers=self.number_of_concept_heads * 2),
                nn.SELU(),
                nn.Linear(self.number_of_concept_heads * 4, self.number_of_concept_heads * 2),
                nn.LayerNorm(self.number_of_concept_heads * 2),
                RBFLayer(in_features_dim=self.number_of_concept_heads * 2, num_kernels=self.number_of_concept_heads, out_features_dim=self.number_of_concept_heads),
                #RBFActivation(self.number_of_concept_heads * 2, n_centers=self.number_of_concept_heads),
                #nn.Linear(self.number_of_concept_heads, self.number_of_concept_heads),
            )
        else:
            self.head_weight_predictor = nn.Sequential(
                nn.Linear(config.dim_model, self.number_of_concept_heads * 4),
                nn.SELU(),
                nn.Linear(self.number_of_concept_heads * 4, self.number_of_concept_heads * 2),
                nn.SELU(),
                nn.Linear(self.number_of_concept_heads * 2, self.number_of_concept_heads),
                nn.Softmax(dim=-1)
            )

        # Standard feed-forward network as in ACTEncoderLayer
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # Layer normalization and dropout layers
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        # From original ACTEncoderLayer
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm
    def forward(self, x, pos_embed=None, key_padding_mask=None):
        """
        Forward pass with concept learning pathway.

        Args:
            x: Input tensor (seq_len, batch_size, dim_model)
            pos_embed: Positional embeddings
            key_padding_mask: Padding mask for attention

        Returns:
            x: Output tensor (seq_len, batch_size, dim_model)
            concept_attn: Attention scores over concepts (optional, if n_concepts > 0)
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)

        # Self-attention pathway (same as original)
        q = k = x if pos_embed is None else x + pos_embed
        x_self_attn = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]

        # If using concepts, add the concept pathway
        # Get batch size from x
        seq_len, batch_size, dim = x.shape

        # Expand concepts to match batch size
        concepts = self.concepts.expand(batch_size, -1, -1).transpose(0, 1)  # (n_concepts, batch_size, dim)

        # Cross-attention between the entire input sequence and concepts
        # Using the entire x as query
        x_concept, concept_attn = self.concept_attn(
            query=q,  # Use the same query as in self-attention (with pos_embed if provided)
            key=concepts,
            value=concepts,
            average_attn_weights=False,
        ) # return shape: `(N, \text{num\_heads}, L, S)

        concept_attn = concept_attn.mean(-2) # Mean over Seq Lenght

        def gumbel_top_k(logits, k, tau=1.0, hard=False):
            """Gumbel softmax with top-k masking"""
            # Get top-k indices
            top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)

            # Create mask for top-k elements
            mask = torch.zeros_like(logits)
            mask.scatter_(-1, top_k_indices, 1.0)

            # Mask out non-top-k elements
            masked_logits = logits.masked_fill(mask == 0, float('-inf'))

            # Apply gumbel_softmax only to top-k elements
            return F.gumbel_softmax(masked_logits, tau=tau, hard=hard, dim=-1)
        # Use mean pooling across sequence dimension to get per-batch representation
        #input_repr = x.mean(0)  # (batch_size, dim_model)
        # First token as
        #input_repr = x[0, :, :].squeeze()
        input_repr = x_concept
        head_logits = self.head_weight_predictor(input_repr).mean(0)  # (batch_size, n_heads)
        head_weights = gumbel_top_k(head_logits, k=3, tau=0.1, hard=False)

        # Apply dynamic weights
        head_weights = head_weights.unsqueeze(-1) # (batch_size, n_heads, 1,)
        concept_attn = (concept_attn * head_weights).sum(1)

        # Concatenate the self-attention output with concept output along feature dimension
        x_combined = torch.cat([x_self_attn, x_concept], dim=-1)

        # Project back to original dimension
        x = self.activation(self.concept_proj(x_combined))


        # Add residual connection and normalization
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # Feed-forward network (same as original)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)

        if self.n_concepts > 0:
            return x, concept_attn
        else:
            return x

from lerobot.common.policies.act.modeling_act import ACTEncoderLayer
class ConceptACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ConceptACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        assert is_vae_encoder is False # Should not be used as a VAE Encode

        self.config = config
        self.is_vae_encoder = is_vae_encoder
        self.use_concepts = getattr(config, "n_concepts", 0) > 0

        # Determine number of layers

        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers - 1)])

        # Add components for concept learning if enabled
        if config.use_concept_learning:
            if config.concept_method == "prediction_head":


                # We append an IdentyLayer so we can share more code with the ConceptTransformer Approach.
                self.layers.append(ACTEncoderLayer(config))

                self.layers.append(nn.Identity())

                # Create prediction heads for each concept type defined in config
                self.concept_heads = nn.ModuleDict({
                    concept_name: nn.Sequential(
                        nn.Linear(config.dim_model, config.concept_dim),
                        nn.ReLU(),  # Add non-linearity between layers
                        nn.Dropout(p=0.2),
                        nn.Linear(config.concept_dim, num_classes)
                    )
                    for concept_name, num_classes in sorted(config.concept_types.items())
                })
            elif config.concept_method == "transformer" or config.concept_method == "transformer_ce" or config.concept_method == "transformer_bce":
                # Create a transformer-based concept learning module
                if num_layers > 1:
                    # Add concept layer as the final layer
                    self.layers.append(ConceptACTEncoderLayer(config))
                else:
                    # If only one layer, make it a concept layer
                    self.layers = nn.ModuleList([ConceptACTEncoderLayer(config)])
            else:
                raise "Unknown concept method"


        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
            self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        concept_attn = None
        concepts = None
        # Process through all layers except potentially the last concept layer
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        if self.config.concept_method == "prediction_head":
            # Use first token from encoder output to predict concepts
            concept_token = x[0]  # Shape: (batch_size, dim_model)

            # Create dictionary of predictions for each concept type
            concepts_dict = {}
            for concept_name, head in self.concept_heads.items():
                # Each head outputs logits for its concept type
                concepts_dict[concept_name] = head(concept_token)

            #concepts = torch.concatenate([value for key, value in sorted(concepts_dict)], dim=-1)
            concepts = concepts_dict
        elif self.config.concept_method == "transformer" or self.config.concept_method == "transformer_ce" or self.config.concept_method == "transformer_bce":
            # Process the last layer, which is a concept layer
            last_layer = self.layers[-1]
            x, concepts = last_layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
            #if self.config.n_heads > 1:
            #    concepts = concepts.mean(1) # Dim 1 should be the Head dimensions
        x = self.norm(x)

        if concepts is not None:
            return x, concepts
        return x
