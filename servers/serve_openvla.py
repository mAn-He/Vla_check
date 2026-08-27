"""OpenVLA-OFT policy server. Runs inside the openvla-oft venv with the
openvla-oft checkout importable (set OFT_DIR or --oft-dir).

Model loading and inference reuse openvla-oft's own official helpers rather
than reimplementing them (all verified against
experiments/robot/libero/run_libero_eval.py, experiments/robot/robot_utils.py
and experiments/robot/openvla_utils.py in github.com/moojink/openvla-oft):

  - GenerateConfig, initialize_model()  (loads VLA + L1-regression action head
    + proprio projector (proprio_dim=8 for LIBERO) + processor, and resolves
    unnorm_key with the "_no_noops" fallback)
  - get_image_resize_size(cfg) -> 224 for the openvla family
  - resize_image_for_policy(img, 224)
  - robot_utils.get_action(...) -> action chunk (list of 7-dim np arrays)
  - process_action per action: normalize_gripper_action(binarize=True) then
    invert_gripper_action, producing env-ready actions

The observation this server receives already matches what OFT's own
prepare_observation() builds from LIBERO obs: 180°-rotated full/wrist images
and the 8-dim [eef_pos, eef axis-angle, gripper_qpos] state.

Default checkpoint: moojink/openvla-7b-oft-finetuned-libero-<suite>
(one checkpoint per suite — pass --checkpoint per eval run; there is also the
combined moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10).

Usage:
    OFT_DIR=third_party/openvla-oft python servers/serve_openvla.py \
        --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
        --task-suite libero_spatial --port 8700
"""

import argparse
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from base_server import PolicyAdapter, run


class OpenVLAOFTAdapter(PolicyAdapter):
    def __init__(self, checkpoint: str, task_suite: str, oft_dir: str, center_crop: bool = True) -> None:
        self._checkpoint = checkpoint
        self._task_suite = task_suite
        self._oft_dir = pathlib.Path(oft_dir).resolve()
        self._center_crop = center_crop
        self._model = None

    def load_model(self) -> None:
        if not (self._oft_dir / "experiments").exists():
            raise FileNotFoundError(
                f"openvla-oft checkout not found at {self._oft_dir} (run setup/40_openvla.sh, or set OFT_DIR)"
            )
        sys.path.insert(0, str(self._oft_dir))
        os.chdir(self._oft_dir)  # official eval scripts assume repo-root cwd

        from experiments.robot.libero.run_libero_eval import (
            GenerateConfig,
            initialize_model,
            process_action,
        )
        from experiments.robot.openvla_utils import resize_image_for_policy
        from experiments.robot.robot_utils import get_action, get_image_resize_size

        cfg = GenerateConfig(
            pretrained_checkpoint=self._checkpoint,
            task_suite_name=self._task_suite,
            center_crop=self._center_crop,
            # use_l1_regression=True, num_images_in_input=2, use_proprio=True
            # are already the GenerateConfig defaults (OFT recipe).
        )
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        self._cfg = cfg
        self._model = model
        self._action_head = action_head
        self._proprio_projector = proprio_projector
        self._noisy_action_projector = noisy_action_projector
        self._processor = processor
        self._resize_size = get_image_resize_size(cfg)
        self._resize_image_for_policy = resize_image_for_policy
        self._get_action = get_action
        self._process_action = process_action

    def predict(self, obs: dict) -> dict:
        img = np.asarray(obs["observation/image"], dtype=np.uint8)
        wrist = np.asarray(obs["observation/wrist_image"], dtype=np.uint8)
        state = np.asarray(obs["observation/state"], dtype=np.float64)
        prompt = str(obs["prompt"])

        observation = {
            "full_image": self._resize_image_for_policy(img, self._resize_size),
            "wrist_image": self._resize_image_for_policy(wrist, self._resize_size),
            "state": state,
            "task_description": prompt,
        }
        actions = self._get_action(
            self._cfg,
            self._model,
            observation,
            prompt,
            processor=self._processor,
            action_head=self._action_head,
            proprio_projector=self._proprio_projector,
            noisy_action_projector=self._noisy_action_projector,
            use_film=self._cfg.use_film,
        )
        chunk = np.stack([self._process_action(np.asarray(a), self._cfg.model_family) for a in actions])
        return {"actions": chunk}

    def metadata(self) -> dict:
        return {"model": f"openvla_oft::{self._checkpoint}", "ready": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument("--checkpoint", default="moojink/openvla-7b-oft-finetuned-libero-spatial")
    parser.add_argument("--task-suite", default="libero_spatial",
                        help="used by OFT to resolve unnorm_key (with _no_noops fallback)")
    parser.add_argument("--oft-dir", default=os.environ.get("OFT_DIR", "third_party/openvla-oft"))
    parser.add_argument("--no-center-crop", action="store_true",
                        help="disable center crop (OFT checkpoints are trained with crop aug; keep default)")
    args = parser.parse_args()
    adapter = OpenVLAOFTAdapter(
        checkpoint=args.checkpoint,
        task_suite=args.task_suite,
        oft_dir=args.oft_dir,
        center_crop=not args.no_center_crop,
    )
    run(adapter, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
