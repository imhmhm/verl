#!/bin/bash
# SkillRL on verl v0.7.1 (NPU) - ALFWorld, template skill retrieval, GRPO.
# 通用配置在 examples/skillrl/config/skillrl.yaml，这里只覆盖 alfworld 专属项。
set -x

MTP_DATASET_HOME=${MTP_DATASET_HOME:-/home/ma-user/work/dataset/dataset_zhh_guiyang}
export PYTHONPATH=$MTP_DATASET_HOME/github/verl_v0.7.1:$PYTHONPATH
cd $MTP_DATASET_HOME/github/verl_v0.7.1

export ALFWORLD_DATA=${ALFWORLD_DATA:-$MTP_DATASET_HOME/hf_data/alfworld}
export MODEL_PATH=${MODEL_PATH:-$MTP_DATASET_HOME/hf_model/Qwen2.5-1.5B-Instruct}
TRAIN_DATA=${TRAIN_DATA:-$MTP_DATASET_HOME/verl_data/skillrl/text/train.parquet}
VAL_DATA=${VAL_DATA:-$MTP_DATASET_HOME/verl_data/skillrl/text/test.parquet}

python -m examples.skillrl.main_skillrl \
    --config-name skillrl \
    actor_rollout_ref.model.path=$MODEL_PATH \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    env.env_name=alfworld/AlfredTWEnv \
    env.max_steps=50 \
    env.rollout.n=8 \
    env.skills_only_memory.skills_json_path=examples/skillrl/memory_data/alfworld/claude_style_skills.json \
    trainer.experiment_name=alfworld_skillrl \
    $@
