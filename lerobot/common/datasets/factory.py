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
import logging
from pprint import pformat

import torch

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
        cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def select_episodes_for_percentage(total_episodes: int, percent: float, seed: int) -> list[int] | None:
    """
    Select episode indices based on percentage.

    Args:
        total_episodes: Total number of episodes available
        percent: Percentage of episodes to use (0-1)
        seed: Random seed for reproducibility

    Returns:
        episode_indices: List of episode indices to keep, or None to use all episodes
    """
    if percent >= 1.0:
        return None  # Use all episodes

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Calculate number of episodes to keep (round to nearest integer)
    num_episodes_to_keep = round(total_episodes * percent)

    # Ensure we keep at least 1 episode
    num_episodes_to_keep = max(1, num_episodes_to_keep)

    # Randomly select episode indices
    all_indices = torch.arange(total_episodes)
    perm = torch.randperm(total_episodes)
    selected_indices = all_indices[perm[:num_episodes_to_keep]]

    # Return as sorted list
    return sorted(selected_indices.tolist())


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )


    use_episode_filtering = cfg.dataset_percent < 1.0

    # Convert single repo_id to list for consistent handling
    repo_ids = cfg.dataset.repo_id if isinstance(cfg.dataset.repo_id, list) else [cfg.dataset.repo_id]

    # When using episode filtering, always use MultiLeRobotDataset
    if use_episode_filtering or len(repo_ids) > 1:
        logging.info(f"Creating MultiLeRobotDataset with {len(repo_ids)} dataset(s)")

        if use_episode_filtering:
            logging.info(f"Applying episode filtering with dataset_percent={cfg.dataset_percent}")

        # Prepare episodes dict for each dataset if using percentage filtering
        episodes_dict = {}
        if use_episode_filtering:
            for i, repo_id in enumerate(repo_ids):
                # Get metadata to determine total episodes
                ds_meta = LeRobotDatasetMetadata(
                    repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
                )
                total_episodes = ds_meta.total_episodes

                # Select episodes based on percentage
                # Use different seed offset for each dataset to ensure different episodes are selected
                selected_episodes = select_episodes_for_percentage(
                    total_episodes, cfg.dataset_percent, cfg.seed + i
                )

                if selected_episodes is not None:
                    episodes_dict[repo_id] = selected_episodes
                    logging.info(
                        f"Dataset '{repo_id}': selected {len(selected_episodes)}/{total_episodes} "
                        f"episodes ({len(selected_episodes) / total_episodes * 100:.1f}%)"
                    )

        # Use first dataset's metadata for delta_timestamps
        ds_meta = LeRobotDatasetMetadata(
            repo_ids[0], root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

        # Create multi-dataset with episode filtering if applicable
        dataset = MultiLeRobotDataset(
            repo_ids=repo_ids,
            root=cfg.dataset.root,
            episodes=episodes_dict if episodes_dict else None,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        dataset.meta = ds_meta

        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

        if use_episode_filtering:
            # Log total frames/episodes after filtering
            total_filtered_episodes = sum(len(ds.episodes) if ds.episodes else ds.meta.total_episodes
                                          for ds in dataset._datasets)
            total_original_episodes = sum(ds.meta.total_episodes for ds in dataset._datasets)
            logging.info(
                f"After episode filtering: {total_filtered_episodes}/{total_original_episodes} "
                f"total episodes ({total_filtered_episodes / total_original_episodes * 100:.1f}%), "
                f"{dataset.num_frames} total frames"
            )

        if cfg.dataset.use_imagenet_stats:
            for dataset_idx in range(len(dataset._datasets)):
                for key in dataset._datasets[dataset_idx].meta.camera_keys:
                    for stats_type, stats in IMAGENET_STATS.items():
                        dataset._datasets[dataset_idx].meta.stats[key][stats_type] = torch.tensor(
                            stats, dtype=torch.float32
                        )

    else:
        # Single dataset without episode filtering (original behavior)
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )
        if cfg.dataset.use_imagenet_stats:
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset