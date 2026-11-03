"""
Fetch logged training metrics of FBRL runs from wandb into a single CSV (used by the Bellman error plots
in visualize_FBRL.py).

The CSV has a "Step" column and one column per run and metric, named "<experiment_name>_<seed> - train/<metric>",
so runs are identified by their experiment name and seed (as set by scripts/run_experiments.sh).

Example:
    python scripts/fetch_wandb_metrics.py --entity <your_entity> --env RoomsEnv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import wandb

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", default=None, help="wandb entity (default: your default entity)")
    parser.add_argument("--project", default="spectral_alignment", help="wandb project (config/logger/wandb.yaml)")
    parser.add_argument("--env", default="RoomsEnv", help="environment class name, e.g. RoomsEnv")
    parser.add_argument("--metrics", nargs="+", default=["train/fb_loss_bellman_offdiag", "train/actor_loss"])
    parser.add_argument("--output", default=None, help="default: models/FBRL/<env>/misc/train_metrics.csv")
    args = parser.parse_args()

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    # oldest first, so that a re-run of the same experiment/seed overwrites the older one below
    runs = api.runs(path, order="+created_at")

    columns = {}
    for run in runs:
        config = run.config
        if not config.get("environment", {}).get("_target_", "").endswith(f".{args.env}"):
            continue
        if not config.get("algorithm", {}).get("_target_", "").endswith(".FBRL"):
            continue
        run_key = f"{config['experiment_name']}_{config['train']['seed']}"
        logging.info(f"Fetching {run_key} ({run.name})")
        for metric in args.metrics:
            history = [(row["_step"], row[metric]) for row in run.scan_history(keys=["_step", metric])]
            if not history:
                continue
            col = f"{run_key} - {metric}"
            if col in columns:
                logging.warning(f"Multiple runs for {run_key}, keeping the most recent one ({run.name})")
            steps, values = zip(*history)
            columns[col] = pd.Series(values, index=steps)

    if not columns:
        raise RuntimeError(f"No FBRL runs for {args.env} found in wandb project '{path}'")

    data = pd.DataFrame(columns).sort_index()
    data.index.name = "Step"
    output = Path(args.output or f"models/FBRL/{args.env}/misc/train_metrics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output)
    logging.info(f"Saved {len(columns)} columns x {len(data)} steps to {output}")


if __name__ == "__main__":
    main()
