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
    successes: List[bool] = field(default_factory=list)
    
    def add_result(self, test_case: TestCase, success: bool):
        self.cases.append(test_case)
        self.successes.append(success)
    
    @property
    def success_rate(self) -> float:
        if not self.successes:
            return 0.0
        return sum(self.successes) / len(self.successes) * 100
    
    def to_dict(self) -> Dict:
        results = []
        for i, (case, success) in enumerate(zip(self.cases, self.successes)):
            results.append({
                "episode": i,
                "color": case.color,
                "shape": case.shape,
                "location": case.location, 
                "expected_dropoff": case.expected_dropoff,
                "success": success
            })
        
        return {
            "results": results,
            "aggregated": {
                "success_rate": self.success_rate,
                "total_episodes": len(self.cases),
                "successful_episodes": sum(self.successes)
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
    
    # Hard-coded test cases from cases.sh
    test_cases.append(TestCase(color="red", shape="cube", location="1", 
                              expected_dropoff=determine_dropoff("red", "cube", "1")))
    test_cases.append(TestCase(color="yellow", shape="rectangle", location="2", 
                              expected_dropoff=determine_dropoff("yellow", "rectangle", "2")))
    
    # Additional test cases can be defined here
    for location in ["3", "4", "5"]:
        test_cases.append(TestCase(color="red", shape="cube", location=location, 
                                  expected_dropoff=determine_dropoff("red", "cube", location)))
        test_cases.append(TestCase(color="yellow", shape="rectangle", location=location, 
                                  expected_dropoff=determine_dropoff("yellow", "rectangle", location)))
    
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
    
    # Initialize keyboard listener with added support for arrow up/down feedback
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
        log_say(f"Test case {i+1}: {test_case.color} {test_case.shape} at location {test_case.location}", play_sounds)
        print(f"\nRunning test case {i+1}/{len(test_cases)}: {test_case.color} {test_case.shape} at location {test_case.location}")
        print(f"Expected dropoff: {test_case.expected_dropoff}")
        print("Press UP arrow for success or DOWN arrow for failure after the episode")
        
        # Reset events for this episode
        events["arrow_up"] = False
        events["arrow_down"] = False
        
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
        
        log_say("Episode complete. Was it successful?", play_sounds)
        
        # Wait for user feedback (arrow up = success, arrow down = failure)
        print(colored("Was the task completed successfully?", "yellow", attrs=["bold"]))
        print(colored("Press UP arrow for YES, DOWN arrow for NO", "yellow", attrs=["bold"]))
        
        # Wait for user input
        while not (events.get("arrow_up", False) or events.get("arrow_down", False)):
            time.sleep(0.1)
            if events.get("stop_recording", False):
                break
            elif events.get("rerecord_episode", False):
                break   
        
        # Record the result
        success = events.get("arrow_up", False)
        results.add_result(test_case, success)
        
        if success:
            log_say("Success recorded", play_sounds)
            print(colored("Success recorded!", "green", attrs=["bold"]))
        else:
            log_say("Failure recorded", play_sounds)
            print(colored("Failure recorded.", "red", attrs=["bold"]))
            
        # Reset events for next episode
        events["arrow_up"] = False
        events["arrow_down"] = False
        events["exit_early"] = False
        
        # Reset environment before next episode
        log_say("Reset environment for next test case", play_sounds)
        print("\nPlease reset the environment for the next test case...")
        # Allow teleoperation during reset to position the robot correctly
        warmup_record(robot, events, True, warmup_time_s=warmup_time_s, display_cameras=display_cameras, fps=fps)
    
    # Stop and clean up
    log_say("Evaluation complete", play_sounds)
    stop_recording(robot, listener, display_cameras)
    
    return results


def on_arrow_press(key, events):
    """Handle arrow key presses for user feedback."""
    if key == pynput.keyboard.Key.up:
        events["arrow_up"] = True
    elif key == pynput.keyboard.Key.down:
        events["arrow_down"] = True
    elif key == pynput.keyboard.Key.left:
        events["rerecord_episode"] = True
    elif key == pynput.keyboard.Key.right:
        events["exit_early"] = True
    elif key == pynput.keyboard.Key.esc:
        events["stop_recording"] = True


def extended_init_keyboard_listener():
    """Initialize the keyboard listener with arrow key handling."""
    events = {
        "stop_recording": False,
        "rerecord_episode": False,
        "exit_early": False,
        "arrow_up": False,
        "arrow_down": False,
    }
    
    listener = pynput.keyboard.Listener(
        on_press=lambda key: on_arrow_press(key, events)
    )
    listener.start()
    
    return listener, events


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
    
    # Load policy
    logging.info("Loading policy.")
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy.eval()
    
    # Get test cases
    test_cases = get_test_cases()
    logging.info(f"Found {len(test_cases)} test cases")
    
    # Override keyboard listener to handle arrow keys
    global init_keyboard_listener
    init_keyboard_listener = extended_init_keyboard_listener
    
    # Run evaluation
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        results = run_evaluation(
            robot=robot,
            policy=policy,
            test_cases=test_cases[:cfg.eval.n_episodes],
            fps=30,
            warmup_time_s=5,
            episode_time_s=30,
            display_cameras=True,
            play_sounds=True,
        )
    
    # Print results
    print("\n" + "="*50)
    print(colored("Evaluation Results:", "blue", attrs=["bold"]))
    print(f"Total Episodes: {len(results.successes)}")
    print(f"Successful Episodes: {sum(results.successes)}")
    print(f"Success Rate: {results.success_rate:.2f}%")
    print("="*50)
    
    # Save results
    results_dict = results.to_dict()
    results_file = output_dir / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    logging.info(f"Results saved to {results_file}")
    
    return results


if __name__ == "__main__":
    init_logging()
    eval_main() 