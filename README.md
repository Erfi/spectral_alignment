# Spectral Alignment in Forward–Backward Representations via Temporal Abstraction

Code for the paper *Spectral Alignment in Forward–Backward Representations via Temporal Abstraction*.
It contains our Forward–Backward RL (FBRL) implementation, with temporal abstraction through action
repeat and action chunking, on [OGBench](https://github.com/seohongpark/ogbench) environments. It also
contains the scripts and notebooks that produce the paper figures.

## Installation

We use [uv](https://docs.astral.sh/uv/) with Python 3.10 or 3.11.

```bash
git clone --recursive https://github.com/erfi/spectral_alignment.git
cd spectral_alignment
git submodule update --init --recursive   # only if you cloned without --recursive
uv sync                                   # add `--extra notebooks` for the analysis notebooks
source .venv/bin/activate
```

`third_party/ogbench` is a submodule pointing to our fork of OGBench, and `uv sync` installs it in
editable mode. The fork adds:
- randomized start and goal positions for the point mazes, with static start–goal pairs kept for single-task use;
- optional randomized tasks for `cube` and `cube_double`;
- extra render modes.
OGBench datasets (needed for the manipulation environments) are downloaded to `~/.ogbench/data` on
first use.

Training logs to [Weights & Biases](https://wandb.ai) (project `spectral_alignment`). Run `wandb login`
first, or set `WANDB_MODE=disabled` to skip logging. Without wandb logging, the Bellman-error plots
can't be made (see [Reproducing the figures](#reproducing-the-figures)).

## Repository layout

```
main.py                         # entry point: train / resume / evaluate (Hydra)
config/                         # Hydra configs: algorithm (fbrl + sub-components), environment, logger, train
spectral_alignment/
  algorithms/fbrl.py            # FBRL agent (forward/backward nets, actor, offline+online replay)
  algorithms/base.py            # shared training loop, validation and checkpointing
  networks/fb_networks.py       # forward/backward networks, actor, features extractors (RBF, CNN, MLP, goal)
  environments/ogbench_env.py   # OGBench environment wrappers and dataset loading
  environments/wrappers.py      # action repeat, action chunking, sparse reward, rendering
  utils/                        # config, evaluation, spectral metrics, dataset relabeling, plotting helpers
scripts/                        # experiment launcher, evaluation sweep, wandb export, figure scripts
notebooks/                      # discrete four-rooms analysis and cover figure
```

## Training

`main.py` is a [Hydra](https://hydra.cc) app. Settings are passed as `key=value` overrides:

```bash
python main.py mode=train environment=rooms algorithm.gamma=0.95 algorithm.z_dim=100 environment.action_repeat_count=5
```

**Environments:** `rooms` (four rooms), `maze`, `largemaze`, `giantmaze`, `arena`, `cube`,
`cube_double`, `scene`.

**Encoders:** the state encoder (`algorithm/features_extractor`, used by the forward network and the actor) and the
goal encoder (`algorithm.goal_features_extractor`, used by the backward network) are chosen separately:

```bash
python main.py mode=train environment=rooms algorithm/features_extractor=mlp \
    algorithm/features_extractor@algorithm.goal_features_extractor=mlp
```

Options are `rbf` and `mlp` (state observations) and `cnn` (images; use it for both state and goal). For
manipulation, `cube_goal`, `cube_double_goal` and `scene_goal` select the task-relevant part of the goal.

**Temporal abstraction** comes in two forms, and they are different methods:
- **Action repeat:** `environment.action_repeat_count=k` repeats each action `k` times.
- **Action chunking:** `environment.action_chunk_count=k` makes the actor output a chunk of `k` actions.

Only one of them can be larger than 1. Offline datasets are relabeled to match.

### Paper settings

[`scripts/run_experiments.sh`](scripts/run_experiments.sh) launches one run with the settings used in the
paper. Each environment gets its own features extractor, network sizes, and offline dataset. You set the
run through environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `ENVIRONMENT` | `rooms` | one of the environments above |
| `ACT_REPEAT` | `1` | action repeat `k` (run named `act-rep-k__...`) |
| `ACT_CHUNK` | unset | action chunk length `k`. If set, even to `1`, the run is named `act-chunk-k__...`. Can't be combined with `ACT_REPEAT` |
| `GAMMA` | `0.95` | discount factor |
| `ZDIM` | `100` | embedding dimension `d` |
| `SEED` | `0` | random seed (the paper uses seeds 0–4) |
| `FEATURES_EXTRACTOR` | `rbf` | navigation environments: `rbf` or `mlp` (state observations), or `cnn` (pixel observations), used for both state and goal. Manipulation environments always use `mlp` for the state and `identity` on the task-relevant part of the goal |
| `NO_DEMO` | `0` | `1` trains manipulation environments without the offline dataset |
| `GPU` | `0` | CUDA device index |
| `WANDB_ENTITY` | `null` | your wandb entity |

```bash
ENVIRONMENT=rooms ACT_REPEAT=10 GAMMA=0.95 ZDIM=100 SEED=0 bash scripts/run_experiments.sh
ENVIRONMENT=largemaze ACT_CHUNK=5 bash scripts/run_experiments.sh
ENVIRONMENT=scene ACT_REPEAT=3 GAMMA=0.99 bash scripts/run_experiments.sh
```

Each run is named by its settings. In navigation environments the name starts with the encoder:
`rbf__act-rep-10__gamma-0.95__zdim-100`, `mlp__act-chunk-5__...`, or `visual__...` for CNN. Manipulation
environments always use an MLP encoder, so their names have no encoder prefix: `act-rep-3__gamma-0.95__zdim-100`,
or `...__no-demo` when trained without the offline dataset. Checkpoints are saved to:

```
models/FBRL/<EnvClass>/<experiment_name>/seed_<seed>/<timestamp>__<pid>/
  config.yaml          # full config, used to rebuild the agent when loading
  evaluations.npz      # returns from validation during training
  model_top_1.zip      # best checkpoint by validation return (also model_top_2.zip)
  last_<step>.zip      # most recent checkpoint
```

All evaluation and plotting scripts look for models in this layout, using the experiment names above.
Trained checkpoints are not included in the repository (`models/` is git-ignored), so train the runs a script
needs before running it; [Reproducing the figures](#reproducing-the-figures) lists the runs for each figure.

### Resuming

```bash
python main.py mode=resume load_model_path=models/FBRL/RoomsEnv/<experiment>/seed_0/<run>/last_<step>.zip
```

This continues the same wandb run. The replay buffer is not saved, so it starts empty again.

## Evaluation

```bash
# 50 episodes on random OGBench evaluation tasks -> test_evaluations__model_top_1.npz
python main.py mode=evaluate load_model_path=<run_dir>/model_top_1.zip

# a single OGBench evaluation task (1-5) -> test_evaluations__task2__model_top_1.npz
python main.py mode=evaluate load_model_path=<run_dir>/model_top_1.zip eval_task_id=2

# watch the agent
python main.py mode=evaluate load_model_path=<run_dir>/model_top_1.zip environment.render_mode=human
```

To evaluate every checkpoint matching a pattern, spread across all idle GPUs (runs that already
have results are skipped unless you pass `--override`):

```bash
python scripts/evaluate_all_fbrl_models.py --pattern "models/FBRL/**/model_top_1.zip"
python scripts/evaluate_all_fbrl_models.py --pattern "models/FBRL/SceneEnv/**/model_top_1.zip" --task_ids 1 2 3 4 5
```

## Reproducing the figures

1. **Train** the runs a figure needs (table below) with `scripts/run_experiments.sh`, using seeds 0–4.
2. **Evaluate** them with `scripts/evaluate_all_fbrl_models.py`. Add `--task_ids 1 2 3 4 5` for Scene.
3. **Bellman-error plots only:** export the training losses from wandb:
   `python scripts/fetch_wandb_metrics.py --entity <your_entity> --env RoomsEnv`
   (writes `models/FBRL/RoomsEnv/misc/train_metrics.csv`).
4. **Plot.** Run the scripts from the repository root. Figures are saved to `plots/`.

| Figure(s) | Runs needed | Script |
|---|---|---|
| Four rooms: return vs. `k`, `γ`, `d` (bar, line, grid, parallel-coordinate plots) | `rooms`, `ACT_REPEAT ∈ {1,3,5,10,20,50}` × `GAMMA ∈ {0.9,0.95,0.99,0.999}` (`ZDIM=100`), and `ZDIM ∈ {25,50,100,200,400}` (`GAMMA=0.95`). The `experiments` list at the top of `visualize_FBRL.py` has the full set | `python scripts/visualize_FBRL.py` |
| Bellman error vs. `k`, `γ`, `d` | subset of the four-rooms runs above, with wandb logging | `fetch_wandb_metrics.py`, then `visualize_FBRL.py` |
| RBF vs. CNN encoders | `rooms`, `maze`, `largemaze` with `FEATURES_EXTRACTOR=cnn` at `ACT_REPEAT ∈ {1,10}`, plus RBF runs at `ACT_REPEAT=10` | `visualize_FBRL.py` |
| Cube: vary `k`, demo vs. no demo | `cube`, `ACT_REPEAT ∈ {1..5}`; `ACT_REPEAT=3 NO_DEMO=1` | `visualize_FBRL.py` |
| Scene: vary `k`, overall and per task | `scene`, `GAMMA=0.99`, `ACT_REPEAT ∈ {1,3,5,10,20}`, evaluated with `--task_ids 1 2 3 4 5` | `visualize_FBRL.py` |
| Action repeat vs. action chunking | `largemaze` (`k ∈ {1,3,5,10,20,50}`) and `cube` (`k ∈ {1..5}`), each run once with `ACT_REPEAT=k` and once with `ACT_CHUNK=k` (including `k=1`) | `visualize_FBRL.py` |
| Continuous spectrum of `M = FBᵀ` (singular values, stable rank, spectral entropy) | `rooms`, `ACT_REPEAT ∈ {1,3,5,10,20}` × `GAMMA ∈ {0.9,0.95,0.99,0.999}` | `python scripts/fbrl_continuous_spectral_plot.py` |
| Cover figure, continuous SR / Q heatmaps | `rooms`, `ZDIM=200` (`GAMMA=0.95`, `ACT_REPEAT=1`) | `python scripts/cover_figure_continuous.py` |
| Cover figure, discrete vs. continuous | `rooms`: baseline, `ZDIM=25`, `GAMMA=0.999`, `ACT_REPEAT=10` | `notebooks/cover_figure.ipynb` |
| Discrete four-rooms spectral analysis | none (computed in closed form) | `notebooks/eigen_analysis.ipynb` |

`visualize_FBRL.py` skips any experiment that has no results yet, so you can produce figures as runs
finish. To sweep a grid of settings, loop over the launcher:

```bash
for SEED in 0 1 2 3 4; do for K in 1 3 5 10 20 50; do
  ENVIRONMENT=rooms ACT_REPEAT=$K SEED=$SEED bash scripts/run_experiments.sh
done; done
```

## Citation

```bibtex
@inproceedings{azad2026spectral,
  title     = {Spectral Alignment in Forward-Backward Representations via Temporal Abstraction},
  author    = {Azad, Seyed Mahdi B and Hoffmann, Jasper and Nematollahi, Iman and
               Zhu, Hao and Valada, Abhinav and Boedecker, Joschka},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
```

## License

MIT, see [LICENSE](LICENSE).
