#!/bin/bash
# SkillRL on verl v0.7.1 (NPU) - WebShop, template skill retrieval, GRPO.
set -x

MTP_DATASET_HOME=${MTP_DATASET_HOME:-/home/ma-user/work/dataset/dataset_zhh_guiyang}
export PYTHONPATH=$MTP_DATASET_HOME/github/verl_v0.7.1:$PYTHONPATH
cd $MTP_DATASET_HOME/github/verl_v0.7.1

export MODEL_PATH=${MODEL_PATH:-$MTP_DATASET_HOME/hf_model/Qwen2.5-1.5B-Instruct}
TRAIN_DATA=${TRAIN_DATA:-$MTP_DATASET_HOME/verl_data/skillrl/text/train.parquet}
VAL_DATA=${VAL_DATA:-$MTP_DATASET_HOME/verl_data/skillrl/text/test.parquet}

python -m examples.skillrl.main_skillrl \
    --config-name skillrl \
    actor_rollout_ref.model.path=$MODEL_PATH \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.max_prompt_length=6000 \
    data.max_response_length=768 \
    data.truncation=left \
    env.env_name=Webshop \
    env.max_steps=15 \
    env.rollout.n=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    env.skills_only_memory.skills_json_path=examples/skillrl/memory_data/webshop/claude_style_skills.json \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    trainer.experiment_name=webshop_skillrl \
    $@
