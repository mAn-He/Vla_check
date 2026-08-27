"""Main evaluation loop: LIBERO env <-> policy server (openpi websocket protocol).

The episode loop mirrors openpi's official examples/libero/main.py step by step
(dummy-action settling period, action-chunk deque, done == success) so results
are comparable with published numbers; openvla-oft's run_libero_eval.py follows
the same structure with the same per-suite max_steps.

Key behaviors required by the project spec:
  - one JSONL line per episode (schema in README/DOC_TRACE), flushed + fsynced
    immediately; failed and crashed episodes are recorded too
  - resume is the DEFAULT: already-recorded (model, suite, task_id, episode_idx)
    tuples are skipped; pass --fresh to ignore existing results
  - ETA printed after the first 3 fresh episodes
  - abort with a loud warning if the first 20 episodes are all failures
    (guards against checkpoint/config mismatch silently producing 0%)
  - --limit-episodes N caps episodes per task; --dry-run lists the plan without
    loading LIBERO envs or contacting any server

Examples:
    python client/run_eval.py --model random --suite libero_spatial --limit-episodes 2 --dry-run
    python client/run_eval.py --model random --suite libero_spatial --limit-episodes 2
    python client/run_eval.py --model pi05_libero --suite libero_10
"""

import argparse
import collections
import json
import os
import pathlib
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import libero_env

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

# Defaults if configs/eval.yaml is absent. resize_with_pad=224 for openpi
# replicates its official client preprocessing; None sends raw 256px images
# (GR00T/OFT servers do their own model-specific resizing, as their official
# harnesses do). replan_steps=None consumes the whole returned chunk (OFT's
# recommended num_open_loop_steps == full chunk); openpi's official client
# replans every 5 steps.
DEFAULT_MODELS = {
    "random": {"host": "localhost", "port": 8000, "resize_with_pad": 224, "replan_steps": 5,
               "model_source": "servers/serve_random.py"},
    "pi05_libero": {"host": "localhost", "port": 8000, "resize_with_pad": 224, "replan_steps": 5,
                    "model_source": "gs://openpi-assets/checkpoints/pi05_libero"},
    "groot_n17_libero": {"host": "localhost", "port": 8600, "resize_with_pad": None, "replan_steps": None,
                         "model_source": "hf:nvidia/GR00T-N1.7-LIBERO"},
    "openvla_oft_libero": {"host": "localhost", "port": 8700, "resize_with_pad": None, "replan_steps": None,
                           "model_source": "hf:moojink/openvla-7b-oft-finetuned-libero-<suite>"},
}


def load_model_config(name: str, config_path: pathlib.Path) -> dict:
    cfg = dict(DEFAULT_MODELS.get(name, {"host": "localhost", "port": 8000,
                                         "resize_with_pad": None, "replan_steps": None,
                                         "model_source": "unknown"}))
    if config_path.exists():
        import yaml

        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg.update((file_cfg.get("models") or {}).get(name) or {})
    return cfg


def result_path(results_dir: pathlib.Path, model: str, suite: str) -> pathlib.Path:
    return results_dir / f"{model}__{suite}.jsonl"


def load_completed(results_dir: pathlib.Path) -> set:
    done = set()
    for path in sorted(results_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add((rec["model"], rec["task_suite"], rec["task_id"], rec["episode_idx"]))
                except (json.JSONDecodeError, KeyError):
                    print(f"[warn] unparseable line in {path}, ignoring it for resume", file=sys.stderr)
    return done


def run_episode(env, task_description, client, model_cfg, max_steps, seed):
    """One episode; returns (success, num_steps, error). Mirrors openpi
    examples/libero/main.py."""
    from image_utils import convert_to_uint8, resize_with_pad

    resize = model_cfg.get("resize_with_pad")
    replan = model_cfg.get("replan_steps")
    action_plan = collections.deque()
    t = 0
    steps_executed = 0
    done = False
    try:
        while t < max_steps + libero_env.NUM_STEPS_WAIT:
            if t < libero_env.NUM_STEPS_WAIT:
                # let objects settle after set_init_state (openpi/OFT convention)
                obs, reward, done, info = env.step(libero_env.DUMMY_ACTION)
                t += 1
                continue

            pieces = libero_env.extract(obs)
            img, wrist = pieces["image"], pieces["wrist_image"]
            if resize:
                img = convert_to_uint8(resize_with_pad(img, resize, resize))
                wrist = convert_to_uint8(resize_with_pad(wrist, resize, resize))

            if not action_plan:
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "observation/state": pieces["state"],
                    "prompt": task_description,
                }
                action_chunk = np.asarray(client.infer(element)["actions"])
                if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
                    raise ValueError(f"server returned action chunk of shape {action_chunk.shape}, expected (T, 7)")
                keep = len(action_chunk) if replan is None else replan
                if len(action_chunk) < keep:
                    raise ValueError(f"chunk length {len(action_chunk)} < replan_steps {keep}")
                action_plan.extend(action_chunk[:keep])

            action = action_plan.popleft()
            obs, reward, done, info = env.step(np.asarray(action, dtype=np.float64).tolist())
            steps_executed += 1
            if done:
                # done within the step budget is the success signal used by
                # openpi, openvla-oft, and LIBERO's own metric.py
                return True, steps_executed, None
            t += 1
        return False, steps_executed, None
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        summary = tb.strip().splitlines()[-1][:500]
        return False, steps_executed, summary


def dry_run(args, episodes_per_task):
    """Print the plan without importing LIBERO envs or contacting servers."""
    print(f"[dry-run] model={args.model} suites={args.suites}")
    total = 0
    for suite in args.suites:
        try:
            from libero.libero import benchmark  # full name list if available

            suite_obj = benchmark.get_benchmark_dict()[suite]()
            names = [suite_obj.get_task(i).language for i in range(suite_obj.n_tasks)]
            n = suite_obj.n_tasks
        except ImportError:
            n = libero_env.STATIC_TASK_COUNTS[suite]
            names = None
            print(f"  (LIBERO not importable here; task count from the static suite map)")
        print(f"  {suite}: {n} tasks x {episodes_per_task} episodes = {n * episodes_per_task}")
        if names:
            for i, name in enumerate(names):
                print(f"    [{i:2d}] {name}")
        total += n * episodes_per_task
    print(f"[dry-run] total episodes: {total}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model name (key in configs/eval.yaml)")
    parser.add_argument("--suite", dest="suites", action="append", choices=list(libero_env.DEFAULT_MAX_STEPS),
                        help="task suite; repeatable. Default: the 4 standard eval suites")
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="episodes per task (50 = official openpi/OFT setting)")
    parser.add_argument("--limit-episodes", type=int, default=None, help="cap episodes per task (cost guardrail)")
    parser.add_argument("--seed", type=int, default=7,
                        help="env seed (7 = openpi/OFT default; note OFT seeds the env itself with 0)")
    parser.add_argument("--host", default=None, help="override server host")
    parser.add_argument("--port", type=int, default=None, help="override server port")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "eval.yaml"))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--fresh", action="store_true",
                        help="ignore existing results (default is RESUME: completed episodes are skipped)")
    parser.add_argument("--dry-run", action="store_true", help="print task/episode plan only")
    args = parser.parse_args()
    args.suites = args.suites or list(SUITES)

    episodes_per_task = args.num_episodes if args.limit_episodes is None else min(args.num_episodes, args.limit_episodes)

    if args.dry_run:
        dry_run(args, episodes_per_task)
        return

    model_cfg = load_model_config(args.model, pathlib.Path(args.config))
    if args.host:
        model_cfg["host"] = args.host
    if args.port:
        model_cfg["port"] = args.port

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    completed = set() if args.fresh else load_completed(results_dir)
    if completed:
        print(f"[resume] {len(completed)} episodes already recorded; skipping them (use --fresh to redo)")

    # GL backend must be picked before libero/robosuite import (inside LiberoSuite).
    from policy_client import WebsocketClientPolicy

    client = WebsocketClientPolicy(model_cfg["host"], model_cfg["port"])
    print(f"[server] metadata: {client.get_server_metadata()}")

    np.random.seed(args.seed)
    fresh_done = 0
    fresh_successes = 0
    fresh_started_at = time.time()
    planned = 0
    plans = []
    for suite_name in args.suites:
        suite = libero_env.LiberoSuite(suite_name)
        for task_id in range(suite.n_tasks):
            for ep in range(episodes_per_task):
                planned += 1
                plans.append((suite, suite_name, task_id, ep))

    remaining = [p for p in plans if (args.model, p[1], p[2], p[3]) not in completed]
    print(f"[plan] {len(remaining)} episodes to run ({planned} planned, {planned - len(remaining)} already done)")

    current_env = None
    current_key = None
    try:
        for suite, suite_name, task_id, episode_idx in remaining:
            if current_key != (suite_name, task_id):
                if current_env is not None:
                    current_env.close()
                current_env, task_description = suite.make_env(task_id, seed=args.seed)
                init_states = suite.init_states(task_id)
                current_key = (suite_name, task_id)

            max_steps = libero_env.DEFAULT_MAX_STEPS[suite_name]
            current_env.reset()
            obs = current_env.set_init_state(init_states[episode_idx])
            del obs  # first obs comes from the settling steps inside run_episode

            t0 = time.time()
            success, num_steps, error = run_episode(
                current_env, task_description, client, model_cfg, max_steps, args.seed
            )
            wall = time.time() - t0

            record = {
                "model": args.model,
                "model_source": model_cfg.get("model_source", "unknown"),
                "task_suite": suite_name,
                "task_id": task_id,
                "task_name": task_description,
                "episode_idx": episode_idx,
                "seed": args.seed,
                "success": bool(success),
                "num_steps": num_steps,
                "max_steps": max_steps,
                "wall_time_sec": round(wall, 2),
                "error": error,
            }
            out_path = result_path(results_dir, args.model, suite_name)
            with open(out_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

            fresh_done += 1
            fresh_successes += int(success)
            print(f"[{fresh_done}/{len(remaining)}] {suite_name}/task{task_id} ep{episode_idx} "
                  f"success={success} steps={num_steps} {wall:.1f}s")

            if fresh_done == 3:
                per_ep = (time.time() - fresh_started_at) / 3
                eta_h = per_ep * (len(remaining) - 3) / 3600
                print(f"[eta] ~{per_ep:.1f}s/episode -> ~{eta_h:.1f}h for the remaining "
                      f"{len(remaining) - 3} episodes. Ctrl-C now if that is not what you want; "
                      f"resume is the default on restart.")
            if fresh_done == 20 and fresh_successes == 0:
                print(
                    "\n[ABORT] 0/20 successes. This usually means a checkpoint/config mismatch "
                    "(e.g. wrong unnorm_key, wrong checkpoint for this repo version — see openpi "
                    "issue #849-style failures), not a bad model. Fix the setup, then rerun "
                    "(completed episodes are skipped automatically).",
                    file=sys.stderr,
                )
                sys.exit(2)
    finally:
        if current_env is not None:
            current_env.close()

    print(f"[done] {fresh_done} episodes, {fresh_successes} successes "
          f"({100.0 * fresh_successes / max(fresh_done, 1):.1f}% of the fresh ones)")


if __name__ == "__main__":
    main()
