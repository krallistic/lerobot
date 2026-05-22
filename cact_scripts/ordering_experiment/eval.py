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

import json
import logging
import os
import pynput
import sys
import time
import random
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
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.act.modeling_lavact import LAVACTPolicy

from lerobot.common.utils.control_utils import control_loop
from lerobot.record import record_loop

from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.common.utils.robot_utils import busy_wait, safe_disconnect
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
import datetime as dt

from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features


@dataclass
class OrderingTestCase:
    """Test case for ordering task with three objects"""
    objects: List[Tuple[str, str]]  # List of (color, shape) tuples in correct order
    initial_positions: List[str]  # Random positions where objects start (fixed per test case)
    test_case_name: str  # Human-readable name

    def __post_init__(self):
        assert len(self.objects) == 3, "Ordering task requires exactly 3 objects"
        assert len(self.initial_positions) == 3, "Must specify 3 initial positions"


@dataclass
class EvalResults:
    cases: List[OrderingTestCase] = field(default_factory=list)
    scores: List[int] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_result(self, test_case: OrderingTestCase, score: int):
        self.cases.append(test_case)
        self.scores.append(score)

    def set_metadata(self, metadata: Dict):
        self.metadata = metadata

    @property
    def success_rate(self) -> float:
        """Calculate success rate (score 6 only)"""
        if not self.scores:
            return 0.0
        return sum(1 for score in self.scores if score == 6) / len(self.scores) * 100

    @property
    def average_score(self) -> float:
        """Calculate average score"""
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    @property
    def score_distribution(self) -> Dict[int, int]:
        """Get distribution of scores"""
        distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for score in self.scores:
            distribution[score] += 1
        return distribution

    def to_dict(self) -> Dict:
        results = []
        for i, (case, score) in enumerate(zip(self.cases, self.scores)):
            results.append({
                "episode": i,
                "test_case_name": case.test_case_name,
                "correct_order": [f"{color}_{shape}" for color, shape in case.objects],
                "initial_positions": case.initial_positions,
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
                "perfect_episodes": sum(1 for score in self.scores if score == 6),
                "score_distribution": {
                    "score_0_complete_failure": score_dist[0],
                    "score_1_one_success": score_dist[1],
                    "score_2_two_successes": score_dist[2],
                    "score_3_three_successes": score_dist[3],
                    "score_4_four_successes": score_dist[4],
                    "score_5_five_successes": score_dist[5],
                    "score_6_perfect_ordering": score_dist[6]
                }
            }
        }


def parse_object_string(obj_str: str) -> Tuple[str, str]:
    """Parse 'color_shape' string into (color, shape) tuple"""
    parts = obj_str.split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid object string: {obj_str}. Expected format: 'color_shape'")
    return parts[0], parts[1]


def get_test_cases() -> List[OrderingTestCase]:
    """Return the list of test cases for the ordering task."""
    # Test case definitions
    test_case_strings = [
        "green_rectangle green_cube red_cylinder",
        "red_rectangle green_cube blue_cylinder",
        "red_rectangle red_cube yellow_cube",
        "red_rectangle red_cube red_cylinder",
        "green_rectangle red_cube yellow_cube",
        "red_rectangle green_rectangle blue_cylinder",
    ]

    # Available positions for object placement
    available_positions = ["2", "3", "4",]

    test_cases = []

    # Set seed for reproducible random positioning
    random.seed(42)

    for i, case_str in enumerate(test_case_strings):
        # Parse objects in correct order
        object_strings = case_str.split()
        objects = [parse_object_string(obj_str) for obj_str in object_strings]

        # Generate random but fixed initial positions for this test case
        initial_positions = random.sample(available_positions, 3)

        # Create human-readable test case name
        test_case_name = f"case_{i + 1}_" + "_".join(object_strings)

        test_cases.append(OrderingTestCase(
            objects=objects,
            initial_positions=initial_positions,
            test_case_name=test_case_name
        ))

    return test_cases


def init_keyboard_listener():
    """Initialize the keyboard listener with numerical scoring for 0-6."""
    events = {
        "stop_recording": False,
        "rerecord_episode": False,
        "exit_early": False,
        "score_0": False,
        "score_1": False,
        "score_2": False,
        "score_3": False,
        "score_4": False,
        "score_5": False,
        "score_6": False,
    }

    def on_key_press(key, events):
        """Handle key presses for user feedback and control."""
        try:
            # Handle number keys for scoring (0-6)
            if hasattr(key, 'char') and key.char and key.char.isdigit():
                score = int(key.char)
                if 0 <= score <= 6:
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

    listener = pynput.keyboard.Listener(
        on_press=lambda key: on_key_press(key, events)
    )
    listener.start()

    return listener, events


def stop_recording(robot, listener, teleop):
    """Stop recording and clean up resources."""
    if listener:
        listener.stop()

    if robot.is_connected:
        robot.disconnect()

    if teleop.is_connected:
        teleop.disconnect()


@safe_disconnect
def run_evaluation(
        robot: Robot,
        teleop: Teleoperator,
        policy: PreTrainedPolicy,
        test_cases: List[OrderingTestCase],
        fps: int = 30,
        warmup_time_s: int = 5,
        episode_time_s: int = 60,  # Increased time for three sequential placements
        display_cameras: bool = True,
        play_sounds: bool = True,
        generic_task: bool = False,
) -> EvalResults:
    """
    Run evaluation on the real robot using the trained policy for the ordering task.

    Args:
        robot: The robot instance
        policy: The trained policy to evaluate
        test_cases: List of ordering test cases to evaluate
        fps: Control frequency
        warmup_time_s: Warmup time in seconds
        episode_time_s: Maximum time per episode in seconds (increased for 3 objects)
        display_cameras: Whether to display camera feeds
        play_sounds: Whether to play audio cues

    Returns:
        Evaluation results
    """
    if not robot.is_connected:
        robot.connect()

    if not teleop.is_connected:
        teleop.connect()

    # Initialize keyboard listener with added support for 0-6 scoring
    listener, events = init_keyboard_listener()
    results = EvalResults()

    log_say("Starting ordering task evaluation", play_sounds)

    # Execute a warmup period to ensure everything is working properly
    log_say("Warmup period", play_sounds)

    action_features = hw_to_dataset_features(robot.action_features, "action", True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
    dataset_features = {**action_features, **obs_features}

    control_loop(
        robot=robot,
        teleop=teleop,
        control_time_s=warmup_time_s,
        display_cameras=display_cameras,
        events=events,
        policy=None,
        fps=fps,
    )

    # Evaluate each test case
    for i, test_case in enumerate(test_cases):
        if events.get("stop_recording", False):
            break

        policy.reset()

        # Announce the current test case with colored output
        def get_termcolor_for_object_color(color: str) -> str:
            """Map object colors to termcolor color names"""
            color_map = {
                "red": "red",
                "green": "green",
                "blue": "blue",
                "yellow": "yellow",
                "cyan": "cyan",
                "magenta": "magenta"
            }
            return color_map.get(color.lower(), "white")

        # Create colored correct order string
        colored_order_parts = []
        for color, shape in test_case.objects:
            colored_text = colored(f"{color} {shape}", get_termcolor_for_object_color(color), attrs=["bold"])
            colored_order_parts.append(colored_text)
        correct_order_str = " -> ".join(colored_order_parts)

        # Create colored positions string (color positions by the object color that starts there)
        colored_position_parts = []
        for i_pos, pos in enumerate(test_case.initial_positions):
            # The i-th position contains the i-th object from the objects list
            object_color = test_case.objects[i_pos][0]  # Get color of object at this position
            colored_pos = colored(f"pos {pos}", get_termcolor_for_object_color(object_color), attrs=["bold"])
            colored_position_parts.append(colored_pos)
        positions_str = ", ".join(colored_position_parts)

        log_say(f"Test case {i + 1}: Ordering task", play_sounds)
        print(f"\nRunning test case {i + 1}/{len(test_cases)}: {test_case.test_case_name}")
        print(f"Correct order: {correct_order_str}")
        print(f"Initial positions: {positions_str}")
        print("Task: Pick up all three objects and place them sequentially in the collection area in the correct order")

        single_task = None
        if isinstance(policy, SmolVLAPolicy) or isinstance(policy, LAVACTPolicy):
            single_task = f"Pick up objects in this order: {correct_order_str}"
            if generic_task:
                single_task = "Pick up the objects in the correct order"
            print("SmolVLA Policy, setting task to:", single_task)

        input("Press Enter to continue...")

        # Reset events for this episode
        for key in [f"score_{j}" for j in range(7)]:
            events[key] = False

        # Run the episode
        log_say("Starting ordering episode", play_sounds)

        # Use the control_loop function from control_utils to run the policy
        control_loop(
            robot=robot,
            teleop=None,
            dataset_features=dataset_features,
            control_time_s=episode_time_s,
            display_cameras=display_cameras,
            events=events,
            policy=policy,
            fps=fps,
            single_task=single_task
        )

        log_say("Episode complete. Please rate the performance.", play_sounds)

        # Display scoring instructions for ordering task
        print(colored("\n" + "=" * 80, "yellow", attrs=["bold"]))
        print(colored("Please rate the robot's performance for the ordering task:", "yellow", attrs=["bold"]))
        print(colored("6 - Perfect: All 3 objects picked and placed in correct order", "green"))
        print(colored("5 - Very good: All 3 objects placed, 1 ordering mistake", "cyan"))
        print(colored("4 - Good: All 3 objects placed, 2 ordering mistakes", "blue"))
        print(colored("3 - Fair: 2 objects successfully placed in sequence", "blue"))
        print(colored("2 - Poor: 1 object successfully placed", "yellow"))
        print(colored("1 - Very poor: Some pick attempts but no successful placements", "yellow"))
        print(colored("0 - Complete failure: No successful pick attempts", "red"))
        print(colored("=" * 80, "yellow", attrs=["bold"]))
        print(colored("Press the number key (0-6) corresponding to the performance level", "white", attrs=["bold"]))

        # Wait for user input
        score = None
        while score is None:
            time.sleep(0.1)
            if events.get("stop_recording", False):
                break
            elif events.get("rerecord_episode", False):
                break

            # Check for score input (0-6)
            for s in range(7):
                if events.get(f"score_{s}", False):
                    score = s
                    break

        if score is not None:
            # Record the result
            results.add_result(test_case, score)

            score_descriptions = {
                0: "Complete failure",
                1: "Some attempts, no successful placements",
                2: "One object successfully placed",
                3: "Two objects successfully placed in sequence",
                4: "All objects placed, 2 ordering mistakes",
                5: "All objects placed, 1 ordering mistake",
                6: "Perfect ordering"
            }

            score_colors = {0: "red", 1: "red", 2: "yellow", 3: "blue", 4: "blue", 5: "cyan", 6: "green"}

            log_say(f"Score {score} recorded", play_sounds)
            print(colored(f"Score {score} recorded: {score_descriptions[score]}",
                          score_colors[score], attrs=["bold"]))

        # Reset events for next episode
        for key in [f"score_{j}" for j in range(7)]:
            events[key] = False
        events["exit_early"] = False

        # Reset environment before next episode
        log_say("Reset environment for next test case", play_sounds)
        # Allow teleoperation during reset to position the robot correctly
        control_loop(
            robot=robot,
            teleop=teleop,
            control_time_s=warmup_time_s,
            display_cameras=display_cameras,
            events=events,
            policy=None,
            fps=fps,
        )
        print("Robot reset\n")

    # Stop and clean up
    log_say("Ordering task evaluation complete", play_sounds)
    stop_recording(robot, listener, teleop)

    return results


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
    teleop: TeleoperatorConfig = None
    dataset: DatasetConfig = None
    dataset_percent: int = 1.0  # Ugly fix
    generic_task: bool = False

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
            self.job_name = f"eval_{self.policy.type}_ordering"

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
            self.output_filename = "eval_results_ordering.json"

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    """Main function for policy evaluation on the real robot for ordering task."""
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

    logging.info("Creating teleop.")
    teleop = make_teleoperator_from_config(cfg.teleop)

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

    # Get test cases for ordering task
    test_cases = get_test_cases()
    logging.info(f"Found {len(test_cases)} ordering test cases")

    policy_seed = cfg.job_name.split("_")[-1]
    policy_percent = cfg.job_name.split("_")[-3] if len(cfg.job_name.split("_")) >= 4 else "unknown"

    # Collect metadata
    metadata = {
        "evaluation_config": {
            "task_type": "ordering",
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
        "teleop_config": asdict(cfg.teleop) if cfg.teleop else None,
        'policy_seed': policy_seed,
        'policy_percent': policy_percent,
        "task_description": {
            "name": "Sequential Ordering Task",
            "description": "Pick and place three objects sequentially into a single collection area according to priority rules",
            "objects_per_episode": 3,
            "max_score": 6
        },
        "scoring_system": {
            "0": "Complete failure (no successful pick attempts)",
            "1": "Some pick attempts but no successful placements",
            "2": "One object successfully placed",
            "3": "Two objects successfully placed in sequence",
            "4": "All three objects placed, but 2 ordering mistakes",
            "5": "All three objects placed, but 1 ordering mistake",
            "6": "Perfect execution (all objects picked and placed in correct order)"
        }
    }

    # Run evaluation
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        results = run_evaluation(
            robot=robot,
            teleop=teleop,
            policy=policy,
            test_cases=test_cases,
            fps=30,
            warmup_time_s=2,
            episode_time_s=60,  # Increased for sequential placement of 3 objects
            display_cameras=True,
            play_sounds=False,
            generic_task=cfg.generic_task,
        )

    # Set metadata in results
    results.set_metadata(metadata)

    # Print results
    print("\n" + "=" * 60)
    print(colored("Ordering Task Evaluation Results:", "blue", attrs=["bold"]))
    print(f"Total Episodes: {len(results.scores)}")
    print(f"Average Score: {results.average_score:.2f}/6.0")
    print(f"Success Rate (Score 6): {results.success_rate:.2f}%")
    print(f"Perfect Episodes: {sum(1 for score in results.scores if score == 6)}")
    print("\nScore Distribution:")
    score_dist = results.score_distribution
    print(f"  Score 0 (Complete Failure): {score_dist[0]} episodes")
    print(f"  Score 1 (No Placements): {score_dist[1]} episodes")
    print(f"  Score 2 (1 Object Placed): {score_dist[2]} episodes")
    print(f"  Score 3 (2 Objects Placed): {score_dist[3]} episodes")
    print(f"  Score 4 (All Placed, 2 Mistakes): {score_dist[4]} episodes")
    print(f"  Score 5 (All Placed, 1 Mistake): {score_dist[5]} episodes")
    print(f"  Score 6 (Perfect Ordering): {score_dist[6]} episodes")
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