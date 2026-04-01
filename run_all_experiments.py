import argparse
import copy
import json
import os
from typing import Dict, List

from trainer import run_training_once


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run all baseline/main experiments across multiple seeds")
    parser.add_argument("--configs_dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    args = parser.parse_args()

    config_files = [
        "exp_joint_only.json",
        "exp_visual_only.json",
        "exp_early_fusion.json",
        "exp_late_fusion.json",
        "exp_stepwise_fusion.json",
    ]

    all_summaries: List[Dict] = []
    for cfg_name in config_files:
        cfg_path = os.path.join(args.configs_dir, cfg_name)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config file: {cfg_path}")

        base_cfg = load_config(cfg_path)
        for seed in args.seeds:
            run_cfg = copy.deepcopy(base_cfg)
            run_cfg["seed"] = seed
            summary = run_training_once(run_cfg)
            all_summaries.append(summary)

    out_manifest = os.path.join(base_cfg.get("experiments_root", os.path.join(os.path.dirname(__file__), "experiments")), "all_runs_manifest.json")
    os.makedirs(os.path.dirname(out_manifest), exist_ok=True)
    with open(out_manifest, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"Completed {len(all_summaries)} runs")
    print(f"Manifest: {out_manifest}")


if __name__ == "__main__":
    main()
