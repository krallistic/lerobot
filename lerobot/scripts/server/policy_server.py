"""LeRobot AsyncInference gRPC policy server.

Serves a pretrained LeRobot policy so that leisaac's LeRobotServicePolicyClient
can drive evaluation from a separate container over gRPC.

Protocol:
  1. leisaac client calls Ready()  -> server replies Empty
  2. leisaac client calls SendPolicyInstructions(RemotePolicyConfig)
     -> server loads policy from checkpoint, replies Empty
  3. Per chunk: client calls SendObservations(stream Observation)
     -> server runs inference, stores action list, replies Empty
  4. Client calls GetActions() -> server returns pickled list[TimedAction]

Usage:
  python -m lerobot.scripts.server.policy_server --port 5555 --device cuda
"""
import argparse
import logging
import pickle
import time
from concurrent import futures
from multiprocessing import Event

import grpc
import numpy as np
import torch

from lerobot.common.policies.factory import get_policy_class
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.scripts.server.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.scripts.server.services_pb2 import Actions, Empty
from lerobot.scripts.server.services_pb2_grpc import (
    AsyncInferenceServicer,
    add_AsyncInferenceServicer_to_server,
)
from lerobot.scripts.server.transport import receive_bytes_in_chunks

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MAX_MSG = 256 * 1024 * 1024  # 256 MB — camera images can be large


def _grpc_options():
    return [
        ("grpc.max_receive_message_length", MAX_MSG),
        ("grpc.max_send_message_length", MAX_MSG),
    ]


def _build_obs(raw_obs: dict, config: RemotePolicyConfig) -> dict:
    """Convert the raw observation dict (sent by leisaac client) to the lerobot
    batch format expected by policy.select_action().

    Images:  raw uint8 (H,W,3)  ->  float (1,3,H,W) in [0,1]
    State:   per-joint floats    ->  float (1,D) tensor
    """
    obs = {}
    for key, feature in config.lerobot_features.items():
        if feature["dtype"] == "image":
            camera_name = key.replace("observation.images.", "")
            img = raw_obs[camera_name]  # (H, W, 3) uint8
            t = torch.from_numpy(np.asarray(img, dtype=np.float32)) / 255.0
            obs[key] = t.permute(2, 0, 1).unsqueeze(0).to(config.device)  # (1,3,H,W)
        elif key == "observation.state":
            state = np.array([raw_obs[n] for n in feature["names"]], dtype=np.float32)
            obs[key] = torch.from_numpy(state).unsqueeze(0).to(config.device)  # (1,D)
    return obs


class LeRobotPolicyServicer(AsyncInferenceServicer):
    def __init__(self):
        self._policy: PreTrainedPolicy | None = None
        self._config: RemotePolicyConfig | None = None
        self._latest_actions: list[TimedAction] | None = None
        self._shutdown = Event()

    # ------------------------------------------------------------------
    def Ready(self, request, context):
        log.info("[server] Ready")
        return Empty()

    # ------------------------------------------------------------------
    def SendPolicyInstructions(self, request, context):
        config: RemotePolicyConfig = pickle.loads(request.data)
        log.info(
            f"[server] Loading policy type={config.policy_type!r} "
            f"from {config.pretrained_name_or_path!r} on {config.device}"
        )
        policy_cfg = PreTrainedConfig.from_pretrained(config.pretrained_name_or_path)
        policy_cls = get_policy_class(policy_cfg.type)
        policy = policy_cls.from_pretrained(config.pretrained_name_or_path)
        policy = policy.to(config.device)
        policy.eval()
        self._policy = policy
        self._config = config
        self._latest_actions = None
        log.info("[server] Policy ready")
        return Empty()

    # ------------------------------------------------------------------
    def SendObservations(self, request_iterator, context):
        if self._policy is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Call SendPolicyInstructions before SendObservations")
            return Empty()

        obs_bytes = receive_bytes_in_chunks(request_iterator, queue=None, shutdown_event=self._shutdown)
        timed_obs: TimedObservation = pickle.loads(obs_bytes)
        raw_obs = timed_obs.observation

        obs = _build_obs(raw_obs, self._config)

        # Reset the policy's internal action queue so a fresh forward pass
        # is triggered for every observation chunk received.
        self._policy.reset()
        n = self._config.actions_per_chunk
        actions: list[TimedAction] = []
        with torch.no_grad():
            for i in range(n):
                action = self._policy.select_action(obs)  # (1, action_dim)
                actions.append(
                    TimedAction(
                        timestamp=time.time(),
                        timestep=timed_obs.timestep + i,
                        action=action.cpu().numpy()[0],
                    )
                )
        self._latest_actions = actions
        log.debug(f"[server] Computed {n} actions for obs timestep {timed_obs.timestep}")
        return Empty()

    # ------------------------------------------------------------------
    def GetActions(self, request, context):
        if self._latest_actions is None:
            log.warning("[server] GetActions called before any inference — returning empty")
            return Actions(data=b"")
        return Actions(data=pickle.dumps(self._latest_actions))


def serve(port: int):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=_grpc_options(),
    )
    add_AsyncInferenceServicer_to_server(LeRobotPolicyServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info(f"[server] Listening on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LeRobot gRPC policy server")
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()
    serve(args.port)
