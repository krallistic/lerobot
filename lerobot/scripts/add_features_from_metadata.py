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

"""
Add new features to an existing LeRobot dataset by extracting information from its metadata.
The enhanced dataset is then saved under a new repo ID.

Example usage:
    python add_features_from_metadata.py \
        --source-repo-id lerobot/pusht \
        --target-repo-id user/pusht_enhanced
"""

import argparse
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Optional, Any, Union

import torch
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME


def extract_features_from_metadata(dataset: LeRobotDataset) -> Dict[int, Dict[str, Any]]:
    """Extract additional features from dataset metadata.
    
    This is where you implement your specific metadata parsing.
    Replace this with your actual metadata extraction code.
    
    Args:
        dataset: LeRobotDataset object
        
    Returns:
        dict: Dictionary mapping from episode_index to a dict of new features
    """
    # Example implementation - replace with your actual metadata extraction
    new_features = {}
    
    # Get information from dataset metadata
    for episode_index in range(dataset.meta.total_episodes):
        # Get episode metadata
        episode_data = dataset.meta.episodes[episode_index]
        episode_stats = dataset.meta.episodes_stats[episode_index]
        
        # Extract interesting information
        # Example: task information, success rates, or anything from the metadata
        tasks = episode_data.get("tasks", [])
        length = episode_data.get("length", 0)
        
        # Create a dictionary of new features for this episode
        new_features[episode_index] = {
            "num_tasks": len(tasks),
            "episode_length": length,
            # Add more features as needed
        }
    
    return new_features


def create_enhanced_dataset(
    source_dataset: LeRobotDataset, 
    target_repo_id: str, 
    new_features_config: Dict[str, Dict[str, Any]], 
    root: Optional[Union[str, Path]] = None
) -> LeRobotDataset:
    """Create a new dataset with additional features.
    
    Args:
        source_dataset: Source LeRobotDataset
        target_repo_id: Repository ID for the target dataset
        new_features_config: Configuration for new features to add
        root: Root directory for the new dataset
        
    Returns:
        LeRobotDataset: The newly created dataset
    """
    # Create a new dataset with the same basic configuration but enhanced features
    features = {**source_dataset.meta.features}
    
    # Add new feature definitions to the features dictionary
    for feature_name, feature_config in new_features_config.items():
        features[feature_name] = feature_config
    
    # Create the new dataset
    enhanced_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=source_dataset.meta.fps,
        root=root,
        robot_type=source_dataset.meta.robot_type,
        features=features,
        use_videos=len(source_dataset.meta.video_keys) > 0,
    )
    
    return enhanced_dataset


def copy_and_enhance_episodes(
    source_dataset: LeRobotDataset, 
    enhanced_dataset: LeRobotDataset, 
    new_features: Dict[int, Dict[str, Any]]
) -> None:
    """Copy episodes from source dataset to enhanced dataset, adding new features.
    
    Args:
        source_dataset: Source LeRobotDataset
        enhanced_dataset: Target enhanced LeRobotDataset
        new_features: Dictionary mapping from episode_index to a dict of new features
    """
    for episode_index in range(source_dataset.meta.total_episodes):
        logging.info(f"Processing episode {episode_index}")
        
        # Get episode data
        episode_indices = source_dataset.meta.episodes[episode_index]
        from_idx = source_dataset.hf_dataset[episode_indices["from"]:episode_indices["to"]]
        
        # Process each frame in the episode
        for i in tqdm.tqdm(range(len(from_idx))):
            frame = from_idx[i]
            
            # Add new features to the frame
            episode_features = new_features[episode_index]
            for feature_name, feature_value in episode_features.items():
                frame[feature_name] = feature_value
            
            # Add the enhanced frame to the new dataset
            enhanced_dataset.add_frame(frame)
        
        # Save the episode
        enhanced_dataset.save_episode()


def main():
    parser = argparse.ArgumentParser(description="Add new features to a LeRobot dataset.")
    parser.add_argument(
        "--source-repo-id",
        type=str,
        required=True,
        help="Source repository ID of the dataset to enhance",
    )
    parser.add_argument(
        "--target-repo-id",
        type=str,
        required=True,
        help="Target repository ID for the enhanced dataset",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for storing the datasets (defaults to HF_LEROBOT_HOME)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the enhanced dataset to HuggingFace Hub",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load the source dataset
    logging.info(f"Loading source dataset from {args.source_repo_id}")
    source_dataset = LeRobotDataset(args.source_repo_id, root=args.root)
    
    # Extract new features from metadata
    logging.info("Extracting features from metadata")
    new_features = extract_features_from_metadata(source_dataset)
    
    # Define configurations for the new features
    # Replace this with your actual feature configurations
    new_features_config = {
        "num_tasks": {
            "dtype": "int64",
            "shape": [1],
            "names": ["num_tasks"],
        },
        "episode_length": {
            "dtype": "int64", 
            "shape": [1],
            "names": ["episode_length"],
        },
        # Add more feature configurations as needed
    }
    
    # Create a new dataset with enhanced features
    logging.info(f"Creating enhanced dataset at {args.target_repo_id}")
    enhanced_dataset = create_enhanced_dataset(
        source_dataset, 
        args.target_repo_id, 
        new_features_config,
        args.root
    )
    
    # Copy and enhance episodes
    logging.info("Copying and enhancing episodes")
    copy_and_enhance_episodes(source_dataset, enhanced_dataset, new_features)
    
    # Push to Hub if requested
    if args.push:
        logging.info(f"Pushing enhanced dataset to {args.target_repo_id}")
        enhanced_dataset.push_to_hub()
    
    logging.info("Done!")


if __name__ == "__main__":
    main() 