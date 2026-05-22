
from lerobot.common.policies.act.modeling_act import ACTPolicy, ACT, ACTDecoder, ACTEncoder, get_activation_fn
from lerobot.common.policies.act.configuration_concept_act import ConceptACTConfig

import torch.nn.functional as F  # noqa: N812
import math

import torch
import torch.nn as nn
from typing import Callable

class ConceptACTEncoderLayer(nn.Module):
    """
    Encoder layer that incorporates concept learning via a parallel pathway.
    Uses the entire input sequence as query for concept attention.
    """
    def __init__(self, config: ConceptACTConfig):
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


class ClassAwareConceptACTEncoderLayer(nn.Module):
    """
    Encoder layer with class-specific concept learning.
    Creates separate concept embeddings and attention heads for each concept class,
    but returns the same output format as ConceptACTEncoderLayer for compatibility.
    """

    def __init__(self, config):
        super().__init__()

        # Standard self-attention as in ACTEncoderLayer
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads * 2, dropout=config.dropout)

        # Class-specific concept learning components
        self.concept_types = config.concept_types
        self.concept_classes = list(sorted(self.concept_types.keys()))  # Ensure consistent ordering
        self.number_of_concept_heads = 1 #config.n_heads * 2
        self.total_concepts = sum(self.concept_types.values())

        # Create separate concept embeddings for each class
        self.class_concepts = nn.ParameterDict()
        self.class_attentions = nn.ModuleDict()

        for class_name, num_concepts in self.concept_types.items():
            # Learnable concept embeddings for this class
            concepts = nn.Parameter(torch.zeros(1, num_concepts, config.dim_model))
            nn.init.trunc_normal_(concepts, std=1.0 / math.sqrt(config.dim_model))
            self.class_concepts[class_name] = concepts

            # Class-specific attention head
            self.class_attentions[class_name] = nn.MultiheadAttention(
                config.dim_model, self.number_of_concept_heads, dropout=config.dropout
            )

        # Combine all class outputs into a unified representation
        self.concept_combiner = nn.Sequential(
            nn.Linear(config.dim_model * len(self.concept_classes), config.dim_model),
            nn.LayerNorm(config.dim_model),
            nn.ReLU()
        )

        # Main projection layer after concatenating with self-attention
        self.concept_proj = nn.Linear(config.dim_model * 2, config.dim_model)

        # Head weight predictor for dynamic attention weighting
        if config.use_rbf_head_selection:
            self.head_weight_predictor = nn.ModuleDict({
                class_name: nn.Sequential(
                    nn.Linear(config.dim_model, self.number_of_concept_heads * 4),
                    nn.SELU(),
                    nn.Linear(self.number_of_concept_heads * 4, self.number_of_concept_heads * 2),
                    nn.LayerNorm(self.number_of_concept_heads * 2),
                    RBFLayer(
                        in_features_dim=self.number_of_concept_heads * 2,
                        num_kernels=self.number_of_concept_heads,
                        out_features_dim=self.number_of_concept_heads
                    ),
                )
                for class_name in self.concept_classes
            })
        else:
            self.head_weight_predictor = nn.ModuleDict({
                class_name: nn.Sequential(
                    nn.Linear(config.dim_model, self.number_of_concept_heads * 4),
                    nn.SELU(),
                    nn.Linear(self.number_of_concept_heads * 4, self.number_of_concept_heads * 2),
                    nn.SELU(),
                    nn.Linear(self.number_of_concept_heads * 2, self.number_of_concept_heads),
                    nn.Softmax(dim=-1)
                )
                for class_name in self.concept_classes
            })

        # Standard feed-forward network
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
        Forward pass with class-specific concept learning.

        Args:
            x: Input tensor (seq_len, batch_size, dim_model)
            pos_embed: Positional embeddings
            key_padding_mask: Padding mask for attention

        Returns:
            x: Output tensor (seq_len, batch_size, dim_model)
            concept_attn: Concatenated attention scores (batch_size, total_concepts)
                         Same format as ConceptACTEncoderLayer for compatibility
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)

        # Self-attention pathway
        q = k = x if pos_embed is None else x + pos_embed
        x_self_attn = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]

        # Class-specific concept processing
        seq_len, batch_size, dim = x.shape
        class_outputs = []
        class_attentions = []

        def gumbel_top_k(logits, k, tau=1.0, hard=False):
            """Gumbel softmax with top-k masking"""
            top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
            mask = torch.zeros_like(logits)
            mask.scatter_(-1, top_k_indices, 1.0)
            masked_logits = logits.masked_fill(mask == 0, float('-inf'))
            return F.gumbel_softmax(masked_logits, tau=tau, hard=hard, dim=-1)

        # Process each concept class in sorted order for consistent concatenation
        for class_name in self.concept_classes:
            # Get class-specific concept embeddings
            class_concepts = self.class_concepts[class_name].expand(batch_size, -1, -1).transpose(0, 1)

            # Class-specific cross-attention
            x_class_concept, class_attn = self.class_attentions[class_name](
                query=q,
                key=class_concepts,
                value=class_concepts,
                average_attn_weights=False,
            )

            # Average attention across sequence length
            class_attn = class_attn.mean(-2)  # (batch_size, n_heads, n_concepts_for_class)

            # Dynamic head weighting
            head_logits = self.head_weight_predictor[class_name](x_class_concept).mean(0)
            #head_weights = gumbel_top_k(head_logits, k=3, tau=0.1, hard=False)
            #head_weights = head_weights.unsqueeze(-1)  # (batch_size, n_heads, 1)

            # Apply head weights and get final attention for this class
            #weighted_attn = (class_attn * head_weights).sum(1)  # (batch_size, n_concepts_for_class)
            weighted_attn = class_attn.squeeze()
            class_attentions.append(weighted_attn)

            # Pool concept features across sequence for this class
            class_features = x_class_concept.mean(0)  # (batch_size, dim_model)
            class_outputs.append(class_features)

        # Concatenate attention weights from all classes in the same order as targets
        # This matches the concatenated one-hot target format: [color, shape, dropoff]
        concept_attn = torch.cat(class_attentions, dim=-1)  # (batch_size, total_concepts)

        # Combine all class outputs for the main pathway
        if class_outputs:
            combined_concepts = torch.cat(class_outputs, dim=-1)  # (batch_size, dim_model * n_classes)
            concept_repr = self.concept_combiner(combined_concepts)  # (batch_size, dim_model)

            # Expand to match sequence dimension
            concept_repr_expanded = concept_repr.unsqueeze(0).expand(seq_len, -1, -1)
        else:
            concept_repr_expanded = torch.zeros_like(x_self_attn)

        # Combine self-attention with concept representation
        x_combined = torch.cat([x_self_attn, concept_repr_expanded], dim=-1)
        x = self.activation(self.concept_proj(x_combined))

        # Add residual connection and normalization
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # Feed-forward network
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)

        return x, concept_attn

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
