#!/bin/bash
# SkillRL on verl v0.7.1 (NPU) — ALFWorld task, template skill retrieval, GRPO.
#
# Adapted from SkillRL examples/grpo_trainer/run_alfworld_skills.sh for the
# verl v0.7.1 config schema. See ../../../../doc/SkillRL_适配verl_v0.7.1_NPU方案.md.
#
# env startup is automatic (no manual setup/serve command per run):
#   - ALFWorld envs are built in-process by env_manager.make_envs ->
#     build_alfworld_envs -> AlfworldEnvs (alfworld lib loads PDDL/game from
#     $ALFWORLD_DATA, config in env_package/alfworld/configs/config_tw.yaml).
#   - One-time prerequisite: `alfworld-download -f` (sets $ALFWORLD_DATA) +
#     `pip install alfworld gymnasium==0.29.1 stable-baselines3==2.6.0`.
#
# Key additions over stock verl:
#   +env.enable_env_rollout=True            -> driver-side env-driven rollout
#   +env.use_skills_only_memory=True        -> SkillsOnlyMemory skill injection
#   reward.reward_manager.*=importlib       -> EpisodeRewardManager (R3)
#
# NPU: vLLM is NPU-adapted in this environment; trainer.device=npu is set
# automatically by main_ppo.auto_set_device.
#
# Prerequisites:
#   - export MODEL_PATH=<your SFT checkpoint> (e.g. Jianwen/Alfworld-7B-SFT)
#   - export ALFWORLD_DATA=<alfworld-download data dir>
#   - run examples/data_preprocess/prepare.py first (see below) to make the
#     placeholder parquet at ~/data/verl-agent/text/
set -x

ENGINE=${1:-vllm}
shift  # Remove first argument so $@ only contains extra params
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
export WANDB_NAME="alfworld_skillrl_qwen2.5_7b"

num_cpus_per_env_worker=0.1  # CPU per alfworld env worker

train_data_size=16   # Moderate size (placeholder parquet only indicates modality/size)
val_data_size=64
group_size=8         # GRPO group size (env.rollout.n)

# Placeholder parquet: only indicates modality + size; real tasks come from the
# alfworld env at reset time. Generates ~/data/verl-agent/text/{train,test}.parquet
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +env.enable_env_rollout=True \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.alfworld.eval_dataset=eval_in_distribution \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json \
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
    trainer.experiment_name=alfworld_skillrl_qwen2.5_7b \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False $@
