"""SkillRL training entry point on verl v0.7.1.

Replaces verl.trainer.main_ppo for env-driven SkillRL tasks: uses
RaySkillRLTrainer (subclass of RayPPOTrainer) instead of RayPPOTrainer, and
assembles the gym envs + TrajectoryCollector on the driver when
env.enable_env_rollout is set.

Reference: verl/trainer/main_ppo.py (TaskRunner structure) +
recipe/prime/main_prime.py (use a custom trainer subclass).
"""

import os

import hydra
import ray
from omegaconf import OmegaConf

from examples.skillrl.skillrl_ray_trainer import RaySkillRLTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


from verl.utils.device import auto_set_device

# v0.7.1's vLLM rollout only supports async server mode (sync generate_sequences
# was removed in PR #4411). AsyncActorRolloutRefWorker runs uvloop in the worker
# process, and generate_sequences -> loop.run_until_complete(rollout_mode())
# fails with "this event loop is already running". nest_asyncio patches
# run_until_complete to work inside a running event loop.
import nest_asyncio
nest_asyncio.apply()


@hydra.main(config_path="config", config_name="skillrl", version_base=None)
def main(config):
    auto_set_device(config)
    run_skillrl(config)


def run_skillrl(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_kwargs", {}).get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        # Disable uvloop in worker processes and apply nest_asyncio.
        # v0.7.1's AsyncActorRolloutRefWorker runs uvloop; generate_sequences
        # calls run_until_complete which fails ("already running") because
        # uvloop's Cython run_until_complete can't be patched by nest_asyncio.
        # Switching to standard asyncio lets nest_asyncio work.
        runtime_env["setup_hook"] = (
            "import asyncio; asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy()); "
            "import nest_asyncio; nest_asyncio.apply()"
        )
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        # SkillRL envs + trajectory collector (env-driven rollout)
        enable_env_rollout = bool(getattr(config, "env", {}).get("enable_env_rollout", False))
        envs = val_envs = traj_collector = None
        if enable_env_rollout:
            from examples.skillrl.agent_system.environments import make_envs
            from examples.skillrl.agent_system.multi_turn_rollout import TrajectoryCollector

            envs, val_envs = make_envs(config)
            tokenizer_for_collector = None  # set below after tokenizer load
            # TrajectoryCollector needs the tokenizer; we delay its creation.

        # tokenizer / processor
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if enable_env_rollout:
            traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        # worker classes: use AsyncActorRolloutRefWorker (v0.7.1 vLLM only
        # supports async server mode; sync generate_sequences is removed).
        # The event loop conflict ("this event loop is already running") in
        # generate_sequences -> run_until_complete is resolved by nest_asyncio
        # (applied in main() before run_skillrl).
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, CriticWorker
            actor_rollout_cls = AsyncActorRolloutRefWorker
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker, CriticWorker
            actor_rollout_cls = AsyncActorRolloutRefWorker
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
            mapping[Role.RefPolicy] = global_pool_id

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # datasets
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        train_dataset = RLHFDataset(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )
        val_dataset = RLHFDataset(
            data_files=config.data.val_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )

        from verl.trainer.main_ppo import create_rl_sampler

        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RaySkillRLTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
