# Concept Learning in ACT (Action Chunking Transformer)

This module implements two approaches for incorporating concept learning into the ACT (Action Chunking Transformer) policy.

## Overview

Concept learning allows the model to predict high-level concepts from visual observations alongside the primary task of action prediction. These concepts can represent attributes like color, shape, position, or any other task-relevant information.

The concept learning component provides several benefits:
- Improved interpretability by explicitly modeling task-relevant concepts
- Potential for better performance through multi-task learning
- Easier debugging and understanding of model behavior

## Implementation Options

The implementation supports two approaches for concept learning:

### 1. Prediction Head Approach

This approach adds a simple linear projection from the encoder output to the concept space:

```
Encoder Output → Linear Layer → Concepts
```

The prediction head is applied to the first token (latent token) from the encoder output, similar to how BERT uses the CLS token for classification tasks.

### 2. Transformer Approach

This approach uses a dedicated transformer decoder to attend to relevant parts of the encoder output:

```
Encoder Output → Concept Transformer Decoder → Concepts
```

The concept transformer uses a learnable query token (similar to DETR's object queries) to attend to relevant features in the encoder output.

## Configuration

The concept learning can be configured with the following parameters in `ACTConfig`:

```python
# Enable concept learning
use_concept_learning: bool = False

# Select the concept learning method
concept_method: str = "prediction_head"  # Options: "prediction_head", "transformer"

# Set the dimension of the concept space
concept_dim: int = 32

# Set the weight for the concept loss in the total loss
concept_weight: float = 1.0
```

## Usage

To use concept learning, provide the concepts as part of the observation batch with the key "observation.concept". The model will predict these concepts during training and inference.

### Training

During training, the loss function includes a term for concept prediction, weighted by `concept_weight`:

```
total_loss = action_loss + concept_weight * concept_loss
```

If using VAE, the KL divergence loss is also included:

```
total_loss = action_loss + kl_weight * kl_loss + concept_weight * concept_loss
```

### Inference

During inference, the model can predict concepts alongside actions, which can be useful for interpretability or downstream tasks. 