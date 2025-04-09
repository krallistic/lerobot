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
import glob

import numpy as np
import torch
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME


def create_feature_configurations() -> Dict[str, Dict[str, Any]]:
    """Define the feature configurations for all one-hot encoded features.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of feature configurations
    """
    # Define possible values for each feature (based on cases.sh)
    all_colors = ["red", "green", "yellow", "blue"]
    all_shapes = ["cube", "rectangle", "cylinder"]
    all_locations = ["1", "2", "3", "4", "5"]
    all_dropoffs = ["A", "B"]
    
    # Create the configurations dictionary
    feature_configs = {
        "concept_color": {
            "dtype": "int64",
            "shape": (len(all_colors),),
            "names": [f"concept_color_{color}" for color in all_colors],
            "values": all_colors,
        },
        "concept_shape": {
            "dtype": "int64",
            "shape": tuple([len(all_shapes)]),
            "names": [f"concept_shape_{shape}" for shape in all_shapes],
            "values": all_shapes,
        },
        "concept_location": {
            "dtype": "int64",
            "shape": tuple([len(all_locations)]),
            "names": [f"concept_location_{location}" for location in all_locations],
            "values": all_locations,
        },
        "concept_dropoff": {
            "dtype": "int64",
            "shape": tuple([len(all_dropoffs)]),
            "names": [f"concept_dropoff_{dropoff}" for dropoff in all_dropoffs],
            "values": all_dropoffs,
        },
    }
    
    return feature_configs


def extract_features_from_metadata(
    dataset: LeRobotDataset, 
    feature_configs: Dict[str, Dict[str, Any]]
) -> Dict[int, Dict[str, Any]]:
    """Extract additional features from dataset metadata.
    
    Extract color, shape, pickup location, and dropoff information from task descriptions,
    and create one-hot encodings for each feature.
    
    Args:
        dataset: LeRobotDataset object
        feature_configs: Dictionary of feature configurations
        
    Returns:
        dict: Dictionary mapping from episode_index to a dict of new features
    """
    # Initialize dictionary for new features
    new_features = {}
    
    # Get information from dataset metadata
    for episode_index in range(dataset.meta.total_episodes):
        # Get episode metadata
        episode_data = dataset.meta.episodes[episode_index]
        
        # Initialize feature vectors with zeros
        feature_vectors = {
            feature_name: np.array([0] * len(config["values"]))
            for feature_name, config in feature_configs.items()
        }
        
        # Extract task information
        tasks = episode_data.get("tasks", [])
        
        if tasks:
            # Parse the task description (Expected format: "Grasping Color: $color Shape: $shape Location: $location Dropoff: $dropoff")
            task_description = tasks[0]  # Assuming one task per episode
            
            # Extract color
            if "Color:" in task_description:
                color_part = task_description.split("Color:")[1].split("Shape:")[0].strip()
                values = feature_configs["concept_color"]["values"]
                if color_part in values:
                    feature_vectors["concept_color"][values.index(color_part)] = 1
            
            # Extract shape
            if "Shape:" in task_description:
                shape_part = task_description.split("Shape:")[1].split("Location:")[0].strip()
                values = feature_configs["concept_shape"]["values"]
                if shape_part in values:
                    feature_vectors["concept_shape"][values.index(shape_part)] = 1
            
            # Extract location (pickup)
            if "Location:" in task_description:
                location_part = task_description.split("Location:")[1].split("Dropoff:")[0].strip()
                values = feature_configs["concept_location"]["values"]
                if location_part in values:
                    feature_vectors["concept_location"][values.index(location_part)] = 1
            
            # Extract dropoff
            if "Dropoff:" in task_description:
                dropoff_part = task_description.split("Dropoff:")[1].strip()
                values = feature_configs["concept_dropoff"]["values"]
                if dropoff_part in values:
                    feature_vectors["concept_dropoff"][values.index(dropoff_part)] = 1
        
        # Create a dictionary of new features for this episode
        new_features[episode_index] = feature_vectors
    
    return new_features


def create_enhanced_dataset(
    source_dataset: LeRobotDataset, 
    target_repo_id: str, 
    feature_configs: Dict[str, Dict[str, Any]], 
    root: Optional[Union[str, Path]] = None
) -> LeRobotDataset:
    """Create a new dataset with additional features.
    
    Args:
        source_dataset: Source LeRobotDataset
        target_repo_id: Repository ID for the target dataset
        feature_configs: Dictionary of feature configurations
        root: Root directory for the new dataset
        
    Returns:
        LeRobotDataset: The newly created dataset
    """
    # Create a new dataset with the same basic configuration but enhanced features
    features = {**source_dataset.meta.features}
    
    # Add new feature definitions to the features dictionary
    for feature_name, feature_config in feature_configs.items():
        # Remove the "values" key as it's not needed for dataset creation
        config_for_dataset = {k: v for k, v in feature_config.items() if k != "values"}
        features[feature_name] = config_for_dataset
    print(features)
    # Create the new dataset
    enhanced_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=source_dataset.meta.fps,
        root=root,
        robot_type=source_dataset.meta.robot_type,
        features=features,
        use_videos=True,
        image_writer_threads=4,

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

    enhanced_dataset.start_image_writer(4)
    print(enhanced_dataset.meta.camera_keys)
    for episode_index in tqdm.tqdm(range(source_dataset.meta.total_episodes)):
        logging.info(f"Processing episode {episode_index}")
        
        # Get episode data
        episode_indices = source_dataset.meta.episodes[episode_index]
        #print(source_dataset.episodes)
        #print(source_dataset.episode_data_index)
        #print(episode_indices)
        #print(source_dataset.hf_dataset)
        #print(source_dataset.image_transforms)
        #print(enhanced_dataset.image_transforms)
        from_to = list(range(source_dataset.episode_data_index['from'][episode_index], source_dataset.episode_data_index["to"][episode_index]))
        #from_idx = source_dataset.hf_dataset[episode_indices["from"]:episode_indices["to"]]
        #from_idx = source_dataset.hf_dataset.filter(lambda example: example['episode_index'] == episode_index)

        # Process each frame in the episode
        for i in from_to:
            frame = source_dataset[i]
            #c, h, w = frame['observation.images.wrist'].shape
            #print("Before:",frame['observation.images.wrist'].shape)
            #frame['observation.images.wrist'] = frame['observation.images.wrist'].reshape(h, w, c)
            #print("After:", frame['observation.images.wrist'].shape)
            #c, h, w = frame['observation.images.webcam'].shape
            #frame['observation.images.webcam'] = frame['observation.images.webcam'].reshape(h, w, c)

            episode_features = new_features[episode_index]
            for feature_name, feature_value in episode_features.items():
                frame[feature_name] = feature_value


            del frame['index']
            del frame['frame_index']
            del frame['task_index']
            del frame['episode_index']
            del frame['timestamp']

            #frame['timestamp'] = np.array(frame['timestamp'])
            # Add the enhanced frame to the new dataset
            enhanced_dataset.add_frame(frame)
        
        # Save the episode
        enhanced_dataset.save_episode()
    enhanced_dataset.stop_image_writer()


def find_datasets_by_prefix(prefix: str, root: Optional[Union[str, Path]] = None) -> List[str]:
    """Find all dataset repository IDs matching a given prefix.
    
    Args:
        prefix: The prefix to search for
        root: Root directory for datasets (defaults to HF_LEROBOT_HOME)
        
    Returns:
        List[str]: List of repository IDs matching the prefix
    """
    if root is None:
        root = HF_LEROBOT_HOME
    
    # Get the username
    username = None
    try:
        import subprocess
        result = subprocess.run(["huggingface-cli", "whoami"], capture_output=True, text=True)
        username = result.stdout.strip().split("\n")[0]
    except Exception as e:
        logging.warning(f"Failed to get username with huggingface-cli: {e}")
        logging.warning("Will search for datasets without username prefix")
    
    # Create the search pattern
    if username:
        search_pattern = f"{username}/so100_{prefix}*"
    else:
        search_pattern = f"*/{prefix}*"
    
    # Find matching directories
    root_path = Path(root)
    matching_dirs = glob.glob(str(root_path / search_pattern))
    
    # Extract repo IDs from paths
    repo_ids = []
    for dir_path in matching_dirs:
        # Get the relative path after the root
        rel_path = Path(dir_path).relative_to(root_path)
        repo_id = str(rel_path)
        repo_ids.append(repo_id)
    
    return repo_ids


def main():
    parser = argparse.ArgumentParser(description="Add new features to LeRobot datasets.")
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix for finding datasets to enhance (e.g., 'individual_cases_simple_')",
    )
    parser.add_argument(
        "--target-suffix",
        type=str,
        default="_enhanced",
        help="Suffix to add to the original repo ID for the enhanced datasets",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for storing the datasets (defaults to HF_LEROBOT_HOME)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Find datasets matching the prefix
    logging.info(f"Searching for datasets with prefix: {args.prefix}")
    repo_ids = find_datasets_by_prefix(args.prefix, args.root)
    
    if not repo_ids:
        logging.error(f"No datasets found matching prefix: {args.prefix}")
        return
    
    logging.info(f"Found {len(repo_ids)} datasets to process")
    
    # Create feature configurations
    logging.info("Creating feature configurations")
    feature_configs = create_feature_configurations()
    
    # Process each dataset
    for source_repo_id in repo_ids:
        # Generate target repo ID by adding suffix
        target_repo_id = source_repo_id.replace(args.prefix, f"{args.prefix}{args.target_suffix}")
        
        logging.info(f"Processing dataset: {source_repo_id} -> {target_repo_id}")
        
        try:
            # Load the source dataset
            logging.info(f"Loading source dataset from {source_repo_id}")
            source_dataset = LeRobotDataset(source_repo_id, root=args.root)
            
            # Extract new features from metadata
            logging.info("Extracting features from metadata")
            new_features = extract_features_from_metadata(source_dataset, feature_configs)
            
            # Create a new dataset with enhanced features
            logging.info(f"Creating enhanced dataset at {target_repo_id}")
            enhanced_dataset = create_enhanced_dataset(
                source_dataset, 
                target_repo_id, 
                feature_configs,
                args.root
            )

            from torchvision.transforms import ToPILImage, v2
            transforms = v2.Compose(
                [
                    ToPILImage()
                ]
            )
            source_dataset.image_transforms = transforms
            
            # Copy and enhance episodes
            logging.info("Copying and enhancing episodes")
            copy_and_enhance_episodes(source_dataset, enhanced_dataset, new_features)
        
            
            logging.info(f"Successfully processed {source_repo_id}")
            
        except Exception as e:
            logging.error(f"Error processing {source_repo_id}: {e}")
            raise e

    logging.info("All datasets processed!")


if __name__ == "__main__":
    main() 