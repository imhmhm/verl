#!/bin/bash
# SkillRL on verl v0.7.1 (NPU) — Search task with GiGPO advantage.
#
# GiGPO = episode-level (Eq.3, ==GRPO) + step-level (Eq.7, anchor-state grouping)
# joint advantage (Eq.8). On verl v0.7.1 it is registered via
# @register_adv_est("gigpo") and dispatched through compute_advantage's else
# branch. The env-driven multi_turn_loop writes anchor_obs / traj_uid / rewards /
# active_masks into non_tensor_batch, which the gigpo estimator consumes.
#
# Search uses similarity-based anchor grouping (enable_similarity=True, thresh=0.9)
# because search observations are textually similar but rarely identical.
#
# Prerequisites: same as run_search_skills.sh + a Search retrieval backend.
set -x

ENGINE=${1:-vllm}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
export WANDB_NAME="search_gigpo_qwen2.5_7b"

train_data_size=256
val_data_size=512
group_size=5

TRAIN_DATA="${TRAIN_DATA:-$HOME/data/searchR1_processed_direct/train.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/searchR1_processed_direct/test.parquet}"

# GiGPO config
mode="mean_std_norm"            # "mean_norm" (w/o std, LOO) or "mean_std_norm" (w/ std)
enable_similarity=True          # similarity-based step grouping for search
similarity_thresh=0.9

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    +algorithm.gigpo.step_advantage_w=1.0 \
    +algorithm.gigpo.mode=$mode \
    +algorithm.gigpo.enable_similarity=$enable_similarity \
    +algorithm.gigpo.similarity_thresh=$similarity_thresh \
    algorithm.gamma=0.95 \
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
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
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
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=EpisodeRewardManager \
    reward.reward_manager.module.path=pkg://examples.skillrl.agent_system.reward_manager.episode \
    trainer.critic_warmup=0 \
    trainer.logger=[console,wandb] \
    trainer.project_name=verl_agent_skillrl \
    trainer.experiment_name=search_gigpo_qwen2.5_7b \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@
