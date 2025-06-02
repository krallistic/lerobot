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
Evaluate a trained policy on a real robot with user feedback.

Usage:
```
python cact_scripts/eval.py \
    --policy.path=outputs/train/concept_act_so100_transformer_ce_lower_concept_weight_seed42/checkpoints/020000/pretrained_model \
    --robot.type=so100 \
    --eval.n_episodes=10 \
    --device=cuda
```
"""

import json
import logging
import os
import pynput
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm

from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    log_control_info,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig


@dataclass
class TestCase:
    color: str
    shape: str
    location: str
    expected_dropoff: str


@dataclass
class EvalResults:
    cases: List[TestCase] = field(default_factory=list)
    scores: List[int] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_result(self, test_case: TestCase, score: int):
        self.cases.append(test_case)
        self.scores.append(score)

    def set_metadata(self, metadata: Dict):
        self.metadata = metadata

    @property
    def success_rate(self) -> float:
        """Calculate success rate (score 3 only)"""
        if not self.scores:
            return 0.0
        return sum(1 for score in self.scores if score == 3) / len(self.scores) * 100

    @property
    def average_score(self) -> float:
        """Calculate average score"""
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    @property
    def score_distribution(self) -> Dict[int, int]:
        """Get distribution of scores"""
        distribution = {0: 0, 1: 0, 2: 0, 3: 0}
        for score in self.scores:
            distribution[score] += 1
        return distribution

    def to_dict(self) -> Dict:
        results = []
        for i, (case, score) in enumerate(zip(self.cases, self.scores)):
            results.append({
                "episode": i,
                "color": case.color,
                "shape": case.shape,
                "location": case.location,
                "expected_dropoff": case.expected_dropoff,
                "score": score
            })

        score_dist = self.score_distribution

        return {
            "metadata": self.metadata,
            "results": results,
            "aggregated": {
                "success_rate": self.success_rate,
                "average_score": self.average_score,
                "total_episodes": len(self.cases),
                "successful_episodes": sum(1 for score in self.scores if score == 3),
                "score_distribution": {
                    "score_0_failed_pick": score_dist[0],
                    "score_1_failed_place": score_dist[1],
                    "score_2_wrong_sorting": score_dist[2],
                    "score_3_complete_success": score_dist[3]
                }
            }
        }


def determine_dropoff(color: str, shape: str, location: str) -> str:
    """
    Determine the correct dropoff location based on the rule:
    Object goes to Drop-off Location A if:
    (The object is red) OR (The object is a rectangle AND NOT yellow)
    """
    # Check if the object is red
    if color == "red":
        return "A"

    # Check if the object is a rectangle AND not yellow
    if shape == "rectangle" and color != "yellow":
        return "A"

    # Otherwise, go to Location B
    return "B"


def get_test_cases() -> List[TestCase]:
    """Return the list of test cases to evaluate."""
    test_cases = []
    tests = [('red', 'cube'), ('yellow', 'rectangle')]
    for tc in tests:
        for location in ["1", "2", "3", "4", "5"]:
            test_cases.append(TestCase(color=tc[0], shape=tc[1], location=location,
                                       expected_dropoff=determine_dropoff(tc[0], tc[1], location)))

    return test_cases


@safe_disconnect
def run_evaluation(
        robot: Robot,
        policy: PreTrainedPolicy,
        test_cases: List[TestCase],
        fps: int = 30,
        warmup_time_s: int = 5,
        episode_time_s: int = 30,
        display_cameras: bool = True,
        play_sounds: bool = True,
) -> EvalResults:
    """
    Run evaluation on the real robot using the trained policy.

    Args:
        robot: The robot instance
        policy: The trained policy to evaluate
        test_cases: List of test cases to evaluate
        fps: Control frequency
        warmup_time_s: Warmup time in seconds
        episode_time_s: Maximum time per episode in seconds
        display_cameras: Whether to display camera feeds
        play_sounds: Whether to play audio cues

    Returns:
        Evaluation results
    """
    if not robot.is_connected:
        robot.connect()

    # Initialize keyboard listener with added support for numerical scoring
    listener, events = extended_init_keyboard_listener()
    results = EvalResults()

    log_say("Starting evaluation", play_sounds)

    # Execute a warmup period to ensure everything is working properly
    log_say("Warmup period", play_sounds)
    warmup_record(robot, events, False, warmup_time_s, display_cameras, fps)

    # Evaluate each test case
    for i, test_case in enumerate(test_cases):
        if events.get("stop_recording", False):
            break

        policy.reset()

        # Announce the current test case
        log_say(f"Test case {i + 1}: {test_case.color} {test_case.shape} at location {test_case.location}", play_sounds)
        print(
            f"\nRunning test case {i + 1}/{len(test_cases)}: {test_case.color} {test_case.shape} at location {test_case.location}")
        print(f"Expected dropoff: {test_case.expected_dropoff}")
        input("Press Enter to continue...")

        # Reset events for this episode
        for key in ["score_0", "score_1", "score_2", "score_3"]:
            events[key] = False

        # Run the episode
        log_say("Starting episode", play_sounds)

        # Use the control_loop function from control_utils to run the policy
        control_loop(
            robot=robot,
            control_time_s=episode_time_s,
            display_cameras=display_cameras,
            events=events,
            policy=policy,
            fps=fps,
            teleoperate=False,
        )

        log_say("Episode complete. Please rate the performance.", play_sounds)

        # Display scoring instructions
        print(colored("\n" + "=" * 60, "yellow", attrs=["bold"]))
        print(colored("Please rate the robot's performance:", "yellow", attrs=["bold"]))
        print(colored("0 - Pick attempt failed (object not successfully grasped)", "red"))
        print(colored("1 - Successful pick but failed place (object dropped/misplaced)", "yellow"))
        print(colored("2 - Successful pick and place but incorrect sorting", "blue"))
        print(colored("3 - Complete success (correct pick, place, and sorting)", "green"))
        print(colored("=" * 60, "yellow", attrs=["bold"]))
        print(colored("Press the number key (0-3) corresponding to the performance level", "white", attrs=["bold"]))

        # Wait for user input
        score = None
        while score is None:
            time.sleep(0.1)
            if events.get("stop_recording", False):
                break
            elif events.get("rerecord_episode", False):
                break

            # Check for score input
            for s in range(4):
                if events.get(f"score_{s}", False):
                    score = s
                    break

        if score is not None:
            # Record the result
            results.add_result(test_case, score)

            score_descriptions = {
                0: "Failed pick attempt",
                1: "Successful pick, failed place",
                2: "Correct pick/place, wrong sorting",
                3: "Complete success"
            }

            score_colors = {0: "red", 1: "yellow", 2: "blue", 3: "green"}

            log_say(f"Score {score} recorded", play_sounds)
            print(colored(f"Score {score} recorded: {score_descriptions[score]}",
                          score_colors[score], attrs=["bold"]))

        # Reset events for next episode
        for key in ["score_0", "score_1", "score_2", "score_3"]:
            events[key] = False
        events["exit_early"] = False

        # Reset environment before next episode

        log_say("Reset environment for next test case", play_sounds)
        # Allow teleoperation during reset to position the robot correctly
        warmup_record(robot, events, True, warmup_time_s=warmup_time_s, display_cameras=display_cameras, fps=fps)
        print("Robot reset\n")

    # Stop and clean up
    log_say("Evaluation complete", play_sounds)
    stop_recording(robot, listener, display_cameras)

    return results


def on_key_press(key, events):
    """Handle key presses for user feedback and control."""
    try:
        # Handle number keys for scoring
        if hasattr(key, 'char') and key.char and key.char.isdigit():
            score = int(key.char)
            if 0 <= score <= 3:
                events[f"score_{score}"] = True
                return

        # Handle special keys
        if key == pynput.keyboard.Key.left:
            events["rerecord_episode"] = True
        elif key == pynput.keyboard.Key.right:
            events["exit_early"] = True
        elif key == pynput.keyboard.Key.esc:
            events["stop_recording"] = True
    except AttributeError:
        # Handle special keys that don't have char attribute
        if key == pynput.keyboard.Key.left:
            events["rerecord_episode"] = True
        elif key == pynput.keyboard.Key.right:
            events["exit_early"] = True
        elif key == pynput.keyboard.Key.esc:
            events["stop_recording"] = True


def extended_init_keyboard_listener():
    """Initialize the keyboard listener with numerical scoring."""
    events = {
        "stop_recording": False,
        "rerecord_episode": False,
        "exit_early": False,
        "score_0": False,
        "score_1": False,
        "score_2": False,
        "score_3": False,
    }

    listener = pynput.keyboard.Listener(
        on_press=lambda key: on_key_press(key, events)
    )
    listener.start()

    return listener, events


from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.default import EvalConfig
from lerobot.scripts.control_robot import ControlPipelineConfig
from lerobot.common.robot_devices.robots.configs import RobotConfig
import datetime as dt


@dataclass
class EvalPipelineConfig:
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch
    # (useful for debugging). This argument is mutually exclusive with `--config`.
    n_episodes: int = 50
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    output_filename: str | None = None  # New parameter for specifying the output filename
    job_name: str | None = None
    seed: int | None = 1000
    robot: RobotConfig = None
    control: ControlPipelineConfig = None
    dataset: DatasetConfig = None

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        else:
            raise Exception("No pretrained path was provided")

        if not self.job_name:
            self.job_name = f"eval_{self.policy.type}"

        # Only create timestamped directory if output_dir is not provided
        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval") / eval_dir

        # Ensure output_dir is a Path object
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Set default filename if not provided
        if not self.output_filename:
            self.output_filename = "eval_results.json"

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    """Main function for policy evaluation on the real robot."""
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    # Set random seed for reproducibility
    set_seed(cfg.seed)

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    # Initialize robot
    logging.info("Creating robot.")
    robot = make_robot_from_config(cfg.robot)

    # Loading TrainingDataset Meta:
    logging.info("Creating dataset")
    from lerobot.common.datasets.factory import make_dataset

    dataset = make_dataset(cfg)

    # Load policy
    logging.info("Loading policy.")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        env_cfg=None,
    )
    policy.eval()

    # Get test cases
    test_cases = get_test_cases()
    logging.info(f"Found {len(test_cases)} test cases")

    # Override keyboard listener to handle scoring keys
    global init_keyboard_listener
    init_keyboard_listener = extended_init_keyboard_listener

    # Collect metadata
    metadata = {
        "evaluation_config": {
            "policy_path": str(cfg.policy.pretrained_path) if hasattr(cfg.policy, 'pretrained_path') else None,
            "policy_type": cfg.policy.type if cfg.policy else None,
            "dataset_repo_id": cfg.dataset.repo_id if cfg.dataset else None,
            "job_name": cfg.job_name,
            "seed": cfg.seed,
            "robot_type": cfg.robot.type if cfg.robot else None,
            "device": str(device),
            "n_episodes_requested": cfg.n_episodes,
            "evaluation_date": dt.datetime.now().isoformat(),
        },
        "policy_config": asdict(cfg.policy) if cfg.policy else None,
        "dataset_config": asdict(cfg.dataset) if cfg.dataset else None,
        "robot_config": asdict(cfg.robot) if cfg.robot else None,
        "scoring_system": {
            "0": "Pick attempt failed (object not successfully grasped)",
            "1": "Successful pick but failed place attempt (object dropped or misplaced)",
            "2": "Successful pick and place but incorrect sorting (object placed in wrong collection area)",
            "3": "Complete success (correct pick, place, and sorting according to the rule)"
        }
    }

    # Run evaluation
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        results = run_evaluation(
            robot=robot,
            policy=policy,
            test_cases=test_cases,
            fps=30,
            warmup_time_s=2,
            episode_time_s=30,
            display_cameras=True,
            play_sounds=False,
        )

    # Set metadata in results
    results.set_metadata(metadata)

    # Print results
    print("\n" + "=" * 60)
    print(colored("Evaluation Results:", "blue", attrs=["bold"]))
    print(f"Total Episodes: {len(results.scores)}")
    print(f"Average Score: {results.average_score:.2f}")
    print(f"Success Rate (Score 3): {results.success_rate:.2f}%")
    print(f"Successful Episodes: {sum(1 for score in results.scores if score == 3)}")
    print("\nScore Distribution:")
    score_dist = results.score_distribution
    print(f"  Score 0 (Failed Pick): {score_dist[0]} episodes")
    print(f"  Score 1 (Failed Place): {score_dist[1]} episodes")
    print(f"  Score 2 (Wrong Sorting): {score_dist[2]} episodes")
    print(f"  Score 3 (Complete Success): {score_dist[3]} episodes")
    print("=" * 60)

    # Save results using the specified filename
    results_dict = results.to_dict()
    results_file = output_dir / cfg.output_filename
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    logging.info(f"Results saved to {results_file}")

    return results


if __name__ == "__main__":
    init_logging()
    eval_main()