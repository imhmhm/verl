#!/bin/bash
# SkillRL on verl v0.7.1 (NPU) — Search task, template skill retrieval, GRPO.
#
# Adapted from SkillRL examples/grpo_trainer/run_search_skills.sh for the
# verl_v0.7.1 config schema. See docs/SkillRL_适配verl_v0.7.1_NPU方案.md.
#
# Key additions over stock verl:
#   +env.enable_env_rollout=True            -> driver-side env-driven rollout
#   +env.use_skills_only_memory=True        -> SkillsOnlyMemory skill injection
#   +env.skills_only_memory.*               -> skill bank config (pillar B/C)
#
# NPU: vLLM is NPU-adapted in this environment; trainer.device=npu is set
# automatically by main_ppo.auto_set_device.
#
# Prerequisites:
#   - A Search retrieval backend at env.search.search_url (default :8000)
#   - Search-R1 style QA parquet at data.train_files/val_files
#   - export MODEL_PATH=<your SFT checkpoint>
set -x

ENGINE=${1:-vllm}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
export WANDB_NAME="search_skillrl_qwen2.5_7b"

train_data_size=256
val_data_size=512
group_size=4

TRAIN_DATA="${TRAIN_DATA:-$HOME/data/searchR1_processed_direct/train.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/searchR1_processed_direct/test.parquet}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=5000 \
    data.max_response_length=700 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    +env.enable_env_rollout=True \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url=http://127.0.0.1:8000/retrieve \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/search/claude_style_skills_search.json \
    +env.skills_only_memory.retrieval_mode=template \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.enable_dynamic_update=True \
    +env.skills_only_memory.update_skills_from_train=True \
    +env.skills_only_memory.update_threshold=0.4 \
    +env.skills_only_memory.max_new_skills=3 \
    +env.skills_only_memory.skill_update_freq=5 \
    trainer.critic_warmup=0 \
    trainer.logger=[console,wandb] \
    trainer.project_name=verl_agent_skillrl \
    trainer.experiment_name=search_skillrl_qwen2.5_7b \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@
