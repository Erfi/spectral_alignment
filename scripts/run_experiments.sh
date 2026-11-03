#!/bin/bash
# Launch one FBRL training run with the per-environment settings used in the paper.
#
# Every setting below can be overridden from the command line, e.g.:
#   ENVIRONMENT=rooms ACT_REPEAT=5 SEED=1 bash scripts/run_experiments.sh
#   ENVIRONMENT=cube ACT_CHUNK=3 GAMMA=0.99 bash scripts/run_experiments.sh
#   ENVIRONMENT=maze FEATURES_EXTRACTOR=cnn bash scripts/run_experiments.sh   # visual observations
#   ENVIRONMENT=rooms FEATURES_EXTRACTOR=mlp bash scripts/run_experiments.sh  # MLP encoder on state observations
#   ENVIRONMENT=cube ACT_REPEAT=3 NO_DEMO=1 bash scripts/run_experiments.sh   # manipulation without the offline dataset
#
# Environments: rooms, maze, largemaze, giantmaze, arena, cube, cube_double, scene

GPU=${GPU:-0}
SEED=${SEED:-0}

GAMMA=${GAMMA:-0.95}
ZDIM=${ZDIM:-100}
# Temporal abstraction: action repeat or action chunking (two different methods, never combined).
# Setting ACT_CHUNK (even ACT_CHUNK=1) names the run act-chunk-k, otherwise it is named act-rep-k.
[ -n "${ACT_CHUNK+x}" ] && USE_CHUNK=1 || USE_CHUNK=0
ACT_REPEAT=${ACT_REPEAT:-1}  # temporal abstraction via action repeat (k)
ACT_CHUNK=${ACT_CHUNK:-1}    # temporal abstraction via action chunking (k)
ENVIRONMENT=${ENVIRONMENT:-rooms}

FEATURES_EXTRACTOR=${FEATURES_EXTRACTOR:-rbf}  # rbf or mlp (state), cnn (visual); always mlp for manipulation envs
WANDB_ENTITY=${WANDB_ENTITY:-null}
WANDB_PROJECT=${WANDB_PROJECT:-spectral_alignment}
TRAIN_STEPS=${TRAIN_STEPS:-1000000}
NO_DEMO=${NO_DEMO:-0}  # 1 = manipulation envs train online only, without the OGBench play dataset
EXTRA_ARGS=""

mkdir -p job_logs/

# Only one form of temporal abstraction at a time
if [ "$USE_CHUNK" = "1" ] && [ "$ACT_REPEAT" -gt 1 ]; then
    echo "Error: Both ACT_REPEAT and ACT_CHUNK are set. Please set only one of them."
    exit 1
elif [ "$USE_CHUNK" = "1" ]; then
    EXPERIMENT_NAME=act-chunk-${ACT_CHUNK}__gamma-${GAMMA}__zdim-${ZDIM}
else
    EXPERIMENT_NAME=act-rep-${ACT_REPEAT}__gamma-${GAMMA}__zdim-${ZDIM}
fi

# Visual vs. state observations
if [ "$FEATURES_EXTRACTOR" = "cnn" ]; then
    VISUAL=True
    BUFFER_SIZE=500000
    BATCH_SIZE=128
    EXPERIMENT_NAME="visual__${EXPERIMENT_NAME}"
else
    VISUAL=False
    BUFFER_SIZE=1000000
    BATCH_SIZE=512
fi

# Navigation envs: the goal (input of the backward net B) uses the same type of encoder as the state,
# and the run name starts with the encoder (rbf__, mlp__, or visual__ for CNN) so runs with different encoders don't mix.
# (manipulation envs always use MLP and keep the plain name; their goal encoder is set below)
case "$ENVIRONMENT" in
    cube|cube_double|scene) ;;
    *)
        EXTRA_ARGS+=" algorithm/features_extractor@algorithm.goal_features_extractor=${FEATURES_EXTRACTOR}"
        case "$FEATURES_EXTRACTOR" in
            rbf|mlp) EXPERIMENT_NAME="${FEATURES_EXTRACTOR}__${EXPERIMENT_NAME}" ;;
        esac
        ;;
esac

# Per-environment settings (override the common ones above)
if [ "$ENVIRONMENT" = "largemaze" ]; then
    EXTRA_ARGS+=" algorithm.actor.hidden_dims=[512,256] algorithm.backwardnet.hidden_dims=[512,256]"
    if [ "$FEATURES_EXTRACTOR" = "rbf" ]; then
        EXTRA_ARGS+=" algorithm.features_extractor.high=40 algorithm.features_extractor.dim=20"
    fi
elif [ "$ENVIRONMENT" = "cube" ] || [ "$ENVIRONMENT" = "cube_double" ] || [ "$ENVIRONMENT" = "scene" ]; then
    # Manipulation envs: state observations only, MLP encoder, goal encoder on the task-relevant state subset,
    # and offline+online training with the OGBench play dataset.
    case "$ENVIRONMENT" in
        cube)        GOAL_EXTRACTOR=cube_goal;        DATASET_NAME=cube-single-play-v0 ;;
        cube_double) GOAL_EXTRACTOR=cube_double_goal; DATASET_NAME=cube-double-play-v0 ;;
        scene)       GOAL_EXTRACTOR=scene_goal;       DATASET_NAME=scene-play-v0 ;;
    esac
    FEATURES_EXTRACTOR=mlp
    EXTRA_ARGS+=" algorithm.features_extractor.hidden_dims=[128,128]"
    EXTRA_ARGS+=" algorithm.features_extractor.output_dim=128"
    EXTRA_ARGS+=" algorithm/features_extractor@algorithm.goal_features_extractor=${GOAL_EXTRACTOR}"
    EXTRA_ARGS+=" algorithm/features_extractor@algorithm.goal_features_extractor.features_extractor=identity"
    EXTRA_ARGS+=" algorithm/obs_normalizer=identity"
    if [ "$NO_DEMO" = "1" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME}__no-demo"
    else
        EXTRA_ARGS+=" environment.dataset_config.dataset_type=ogbench"
        EXTRA_ARGS+=" environment.dataset_config.dataset_name=${DATASET_NAME}"
        EXTRA_ARGS+=" environment.dataset_config.dataset_fraction=1.0"
    fi
fi

COMMAND="python main.py \
        mode=train \
        environment=$ENVIRONMENT \
        environment.action_repeat_count=$ACT_REPEAT \
        environment.action_chunk_count=$ACT_CHUNK \
        environment.visual=$VISUAL \
        algorithm=fbrl \
        algorithm.z_dim=$ZDIM \
        algorithm.gamma=$GAMMA \
        algorithm.buffer_size=$BUFFER_SIZE \
        algorithm/features_extractor=$FEATURES_EXTRACTOR \
        train.device=cuda:$GPU \
        train.seed=$SEED \
        train.batch_size=$BATCH_SIZE \
        train.train_steps=$TRAIN_STEPS \
        experiment_name=$EXPERIMENT_NAME \
        logger.entity=$WANDB_ENTITY \
        logger.project=$WANDB_PROJECT \
        $EXTRA_ARGS"

LOGFILE="job_logs/${ENVIRONMENT}_fbrl_${EXPERIMENT_NAME}_seed-${SEED}.log"

echo $COMMAND | tee $LOGFILE
eval $COMMAND 2>&1 | tee -a $LOGFILE
