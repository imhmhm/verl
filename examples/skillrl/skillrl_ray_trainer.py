"""SkillRL Ray trainer — env-driven multi-turn rollout on verl v0.7.1.

Inherits RayPPOTrainer and overrides init_workers/_validate/fit to add:
- env-driven rollout (bypass async AgentLoopManager, use TrajectoryCollector)
- skill evolution hooks (pillar C)
- invalid action penalty
- env-mode reward computation (EpisodeRewardManager via reward_loop_manager)

Reference: verl recipe/prime, recipe/rep_exp (inherit RayPPOTrainer, override fit).
Framework core (verl/trainer/ppo/ray_trainer.py) stays untouched except the
GiGPO registry forwarding in compute_advantage's else branch.
"""
from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    """SkillRL: penalize invalid env actions (is_action_valid=0)."""
    reward_tensor = data.batch["token_level_scores"]
    step_rewards = data.batch["step_rewards"] if "step_rewards" in data.batch.keys() else None
    for i in range(len(data)):
        data_item = data[i]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        action_valids = data_item.non_tensor_batch["is_action_valid"].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids
        if step_rewards is not None:
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    valid_action_ratio = np.mean(data.non_tensor_batch["is_action_valid"].astype(np.float32)).item()
    metrics = {"episode/valid_action_ratio": valid_action_ratio}
    return data, metrics


class RaySkillRLTrainer(RayPPOTrainer):
    """RayPPOTrainer + SkillRL env-driven rollout + skill evolution."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls=None,
        processor=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
        traj_collector=None,
        envs=None,
        val_envs=None,
    ):
        if ray_worker_group_cls is None:
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        # SkillRL env-driven rollout state
        self.traj_collector = traj_collector
        self.envs = envs
        self.val_envs = val_envs
        self.enable_env_rollout = bool(self.config.env.get("enable_env_rollout", False)) if hasattr(self.config, "env") else False

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        # SkillRL: when env.enable_env_rollout is set, skip the async
        # AgentLoopManager entirely — the driver drives a gym env via
        # TrajectoryCollector.multi_turn_loop using the synchronous worker
        # generate_sequences path. async_rollout_manager stays None.
        if self.enable_env_rollout:
            self.async_rollout_mode = False
            self.async_rollout_manager = None
            self.checkpoint_manager = None
        else:
            self.async_rollout_mode = True

            # Support custom AgentLoopManager via config
            manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
            if manager_class_fqn:
                AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
            else:
                from verl.experimental.agent_loop import AgentLoopManager

            # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
            # agent_reward_loop: streaming reward computation with actor rollout
            # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
            enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

            # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
            # to stream reward computation with actor rollout
            reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
            self.async_rollout_manager = AgentLoopManager.create(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rollout_resource_pool=actor_rollout_resource_pool,
                reward_loop_worker_handles=reward_loop_worker_handles,
            )
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
            self.checkpoint_manager = CheckpointEngineManager(
                config=checkpoint_engine_config,
                trainer=self.actor_rollout_wg,
                replicas=self.async_rollout_manager.rollout_replicas,
            )

        # sleep all replicas to load checkpoint
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.sleep_replicas()

    # ------------------------------------------------------------------ #
    # SkillRL skill-evolution hooks (pillar C) — ported verbatim from    #
    # SkillRL's ray_trainer.py. Triggered when *_success_rate < threshold.#
    # New skills are written to the TRAIN env's retrieval_memory only.   #
    # ------------------------------------------------------------------ #

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        # SkillRL: per-task success_rate collected for skill evolution.
        success_rate_dict = {}

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            if self.enable_env_rollout:
                # SkillRL env-driven validation rollout on the driver.
                test_output_gen_batch_padded = self.traj_collector.multi_turn_loop(
                    gen_batch=test_gen_batch,
                    actor_rollout_wg=self.actor_rollout_wg,
                    envs=self.val_envs,
                    is_train=False,
                )
                pad_size = 0
            else:
                # pad to be divisible by dp_size
                size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

                if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                    # for colocate reward models, we need to sleep rollout model
                    # to spare GPU memory for reward model
                    self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            # SkillRL env mode: compute rule-based reward via the reward_loop_manager
            # (EpisodeRewardManager reads env episode_rewards) since the async
            # rollout manager is bypassed.
            if self.enable_env_rollout and "rm_scores" not in test_batch.batch.keys():
                test_batch_reward = self._compute_reward_colocate(test_batch)
                test_batch = test_batch.union(test_batch_reward)
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            # SkillRL: collect per-task success_rate from env success_evaluator.
            for k in test_batch.non_tensor_batch.keys():
                if "success_rate" in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(float(np.mean(test_batch.non_tensor_batch[k])))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        # SkillRL pillar C: evolve skill bank from validation failures
        # (default path when update_skills_from_train is False).
        if self.enable_env_rollout and self.config.env.get("skills_only_memory", {}).get(
            "enable_dynamic_update", False
        ) and not self.config.env.skills_only_memory.get("update_skills_from_train", False):
            success_rate = {k: float(np.mean(v)) for k, v in success_rate_dict.items()}
            self._update_skills_from_validation(
                sample_inputs=sample_inputs,
                sample_outputs=sample_outputs,
                sample_scores=sample_scores,
                success_rate=success_rate,
            )
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)


    def _update_skills_from_validation(
        self,
        sample_inputs: list,
        sample_outputs: list,
        sample_scores: list,
        success_rate: dict,
    ):
        """Update the skill bank using validation results.

        Only triggers when at least one task type's success rate is below the
        configured threshold. New skills are added to the training env memory
        only — not to val_envs — to avoid inflating future validation scores
        with skills specifically tuned to the validation set.
        """
        if not self.enable_env_rollout or not hasattr(self.config, "env"):
            return
        update_config = self.config.env.skills_only_memory
        threshold = update_config.get("update_threshold", 0.5)

        if not success_rate:
            print("[SkillUpdate] No success_rate metrics found in validation results, skipping update")
            return

        needs_update = False
        low_success_tasks = []
        for task_key, rate in success_rate.items():
            if rate < threshold:
                needs_update = True
                task_type = task_key.replace("_success_rate", "")
                low_success_tasks.append(task_type)

        if not needs_update:
            print(f"[SkillUpdate] All task success rates above {threshold}, skipping update")
            return

        print(f"[SkillUpdate] Low success tasks: {low_success_tasks}, triggering skill update...")

        failed_trajectories = self._collect_failed_trajectories(sample_inputs, sample_outputs, sample_scores)
        if not failed_trajectories:
            print("[SkillUpdate] No failed trajectories found")
            return

        if not hasattr(self, "skill_updater"):
            from examples.skillrl.agent_system.memory.skill_updater import SkillUpdater

            self.skill_updater = SkillUpdater(max_new_skills_per_update=update_config.get("max_new_skills", 3))

        if not (hasattr(self, "envs") and hasattr(self.envs, "retrieval_memory") and self.envs.retrieval_memory):
            print("[SkillUpdate] No retrieval_memory found in training envs, cannot determine current skills")
            return

        print(f"[SkillUpdate] Analyzing {len(failed_trajectories)} failed trajectories...")
        new_skills = self.skill_updater.analyze_failures(
            failed_trajectories=failed_trajectories,
            current_skills=self.envs.retrieval_memory.skills,
        )

        if new_skills:
            self.envs.retrieval_memory.add_skills(new_skills, category="general")
            print(f"[SkillUpdate] Added {len(new_skills)} new skills to training envs")
            save_dir = self.config.trainer.get("default_local_dir", "./outputs")
            save_path = os.path.join(save_dir, f"updated_skills_step{self.global_steps}.json")
            self.envs.retrieval_memory.save_skills(save_path)
            print(f"[SkillUpdate] Saved updated skill bank to {save_path}")
        else:
            print("[SkillUpdate] No new skills generated")

    def _update_skills_from_training(self, batch):
        """Update the skill bank using training batch results (no val leakage)."""
        if not self.enable_env_rollout or not hasattr(self.config, "env"):
            return
        update_config = self.config.env.skills_only_memory
        threshold = update_config.get("update_threshold", 0.5)

        success_rate = {}
        for k in batch.non_tensor_batch.keys():
            if "success_rate" in k:
                success_rate[k] = np.mean(batch.non_tensor_batch[k])

        if not success_rate:
            return

        needs_update = any(rate < threshold for rate in success_rate.values())
        if not needs_update:
            return

        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        if "token_level_scores" not in batch.batch:
            return
        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

        failed_trajectories = self._collect_failed_trajectories(inputs, outputs, scores)
        if not failed_trajectories:
            return

        if not hasattr(self, "skill_updater"):
            from examples.skillrl.agent_system.memory.skill_updater import SkillUpdater

            self.skill_updater = SkillUpdater(max_new_skills_per_update=update_config.get("max_new_skills", 3))

        if not (hasattr(self, "envs") and hasattr(self.envs, "retrieval_memory") and self.envs.retrieval_memory):
            return
        retrieval_memory = self.envs.retrieval_memory

        print(f"[SkillUpdate-Train] Analyzing {len(failed_trajectories)} failed trajectories...")
        new_skills = self.skill_updater.analyze_failures(
            failed_trajectories=failed_trajectories,
            current_skills=retrieval_memory.skills,
        )
        if new_skills:
            retrieval_memory.add_skills(new_skills, category="general")
            save_dir = self.config.trainer.get("default_local_dir", "./outputs")
            save_path = os.path.join(save_dir, f"updated_skills_step{self.global_steps}.json")
            retrieval_memory.save_skills(save_path)
            print(f"[SkillUpdate-Train] Added {len(new_skills)} skills, saved to {save_path}")

    def _collect_failed_trajectories(self, inputs: list, outputs: list, scores: list) -> list:
        """Collect failed trajectories for skill analysis (capped at 10)."""
        failed = []
        for inp, out, score in zip(inputs, outputs, scores):
            if score <= 0:
                failed.append(
                    {
                        "task": self._extract_task_description(inp),
                        "trajectory": self._parse_conversation_to_steps(inp, out),
                        "task_type": self._detect_task_type_from_input(inp),
                    }
                )
        return failed[:10]

    def _extract_task_description(self, inp: str) -> str:
        import re

        patterns = [
            r"(?:Your task is to|Task:|task is to|you need to)[:\s]+(.*?)(?:\n|$)",
            r"(?:goal|objective)[:\s]+(.*?)(?:\n|$)",
        ]
        for pat in patterns:
            m = re.search(pat, inp, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:1000]
        for marker in ("<|im_start|>user\n", "\nHuman: ", "\nUser: "):
            idx = inp.find(marker)
            if idx >= 0:
                start = idx + len(marker)
                return inp[start : start + 1000]
        return inp[:1000]

    def _parse_conversation_to_steps(self, inp: str, out: str) -> list:
        import re

        steps = []
        user_turns = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", inp, re.DOTALL)
        asst_turns = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", inp, re.DOTALL)
        if user_turns and asst_turns:
            for obs, act in zip(user_turns, asst_turns):
                steps.append({"action": act.strip()[:1500], "observation": obs.strip()[:800]})
            steps.append({"action": out[:2000], "observation": ""})
            return steps
        user_turns = re.findall(r"(?:Human|User):\s*(.*?)(?=(?:Human|User|Assistant):|$)", inp, re.DOTALL | re.IGNORECASE)
        asst_turns = re.findall(r"Assistant:\s*(.*?)(?=(?:Human|User|Assistant):|$)", inp, re.DOTALL | re.IGNORECASE)
        if user_turns and asst_turns:
            for obs, act in zip(user_turns, asst_turns):
                steps.append({"action": act.strip()[:1500], "observation": obs.strip()[:800]})
            steps.append({"action": out[:2000], "observation": ""})
            return steps
        steps.append({"action": "", "observation": inp[:3000]})
        steps.append({"action": out[:2000], "observation": ""})
        return steps

    def _detect_task_type_from_input(self, inp: str) -> str:
        inp_lower = inp.lower()
        if "clean" in inp_lower:
            return "clean"
        elif "heat" in inp_lower:
            return "heat"
        elif "cool" in inp_lower:
            return "cool"
        elif "look at" in inp_lower and ("lamp" in inp_lower or "light" in inp_lower):
            return "look_at_obj_in_light"
        elif "examine" in inp_lower:
            return "examine"
        else:
            return "pick_and_place"


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if self.enable_env_rollout:
                            # SkillRL env-driven multi-turn rollout on the driver.
                            # v0.7.1 _get_gen_batch pops only non_tensor keys (batch
                            # may be empty/None). multi_turn_loop's preprocess_single_sample
                            # does its own tokenization from raw_prompt, so it doesn't
                            # need input_ids in gen_batch.batch. But vanilla_multi_turn_loop
                            # uses len(gen_batch.batch) for batch_size - fix that to use
                            # non_tensor_batch len instead.
                            gen_batch_output = self.traj_collector.multi_turn_loop(
                                gen_batch=gen_batch,
                                actor_rollout_wg=self.actor_rollout_wg,
                                envs=self.envs,
                                is_train=True,
                            )
                            gen_batch_output.meta_info.setdefault("timing", {})
                        else:
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch.get("multi_modal_inputs", []):
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        # SkillRL env mode: always compute rule-based reward via the
                        # reward_loop_manager (EpisodeRewardManager reads env
                        # episode_rewards) even when use_rm is False.
                        need_reward_compute = (self.use_rm or self.enable_env_rollout) and "rm_scores" not in batch.batch.keys()
                        if need_reward_compute:
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # SkillRL: penalize invalid env actions (env projection
                        # writes is_action_valid). Only in env-driven rollout mode.
                        if (
                            self.enable_env_rollout
                            and self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", False)
                            and "is_action_valid" in batch.non_tensor_batch
                        ):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # SkillRL pillar C: evolve the skill bank from training-batch
                        # failures (no val leakage). Triggered at skill_update_freq
                        # when success_rate < update_threshold.
                        if self.enable_env_rollout and self.config.env.get("skills_only_memory", {}).get(
                            "enable_dynamic_update", False
                        ) and self.config.env.skills_only_memory.get("update_skills_from_train", False):
                            skill_update_freq = self.config.env.skills_only_memory.get(
                                "skill_update_freq", self.config.trainer.get("test_freq", 5)
                            )
                            if self.global_steps > 0 and self.global_steps % skill_update_freq == 0:
                                self._update_skills_from_training(batch)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            if self.checkpoint_manager is not None:
                                self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
