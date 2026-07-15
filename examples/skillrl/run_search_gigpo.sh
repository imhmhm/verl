#!/bin/bash
# SkillRL on verl v0.7.1 (NPU) - Search, GiGPO advantage, similarity grouping.
# 需检索后端在 env.search.search_url。
set -x

MTP_DATASET_HOME=${MTP_DATASET_HOME:-/home/ma-user/work/dataset/dataset_zhh_guiyang}
export PYTHONPATH=$MTP_DATASET_HOME/github/verl_v0.7.1:$PYTHONPATH
cd $MTP_DATASET_HOME/github/verl_v0.7.1

export MODEL_PATH=${MODEL_PATH:-$MTP_DATASET_HOME/hf_model/Qwen2.5-7B-Instruct}
TRAIN_DATA=${TRAIN_DATA:-$MTP_DATASET_HOME/verl_data/searchR1_processed_direct/train.parquet}
VAL_DATA=${VAL_DATA:-$MTP_DATASET_HOME/verl_data/searchR1_processed_direct/test.parquet}

python -m examples.skillrl.main_skillrl \
    --config-name skillrl \
    actor_rollout_ref.model.path=$MODEL_PATH \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.max_prompt_length=5000 \
    data.max_response_length=700 \
    data.truncation=left \
    data.train_batch_size=256 \
    data.val_batch_size=512 \
    env.env_name=search \
    env.max_steps=4 \
    env.history_length=4 \
    env.rollout.n=5 \
    env.skills_only_memory.skills_json_path=examples/skillrl/memory_data/search/claude_style_skills.json \
    algorithm.adv_estimator=gigpo \
    algorithm.gamma=0.95 \
    algorithm.gigpo.enable_similarity=True \
    algorithm.gigpo.similarity_thresh=0.9 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    trainer.experiment_name=search_gigpo_skillrl \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    $@
