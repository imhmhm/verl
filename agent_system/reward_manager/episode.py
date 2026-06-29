# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Episode-level reward manager for SkillRL on verl v0.7.1.

R3 adaptation: implemented as an ``AbstractRewardManager`` subclass so it slots
into verl v0.7.1's native reward pipeline (``reward_loop_manager`` ->
``reward_manager.__call__``) WITHOUT bypassing it. Selected via config:

    reward.reward_manager.source: importlib
    reward.reward_manager.name: EpisodeRewardManager
    reward.reward_manager.module.path: agent_system.reward_manager.episode

The reward comes from the env: ``TrajectoryCollector.multi_turn_loop`` writes
``episode_rewards`` / ``episode_lengths`` / ``*_success_rate`` into each row's
non_tensor_batch (via ``gather_rollout_data``). This manager reads
``episode_rewards`` and places it on the last valid response token — the
SkillRL/verl-agent semantics, retargeted onto the v0.7.1 reward-manager contract.
``compute_score`` is accepted for signature compatibility but unused, because
the env already produced the scalar reward during rollout (there is nothing to
re-score from text).
"""

from collections import defaultdict
from typing import Any, Callable

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager


class EpisodeRewardManager(AbstractRewardManager):
    """Reward manager whose score is the env's per-trajectory episode reward.

    Unlike ``NaiveRewardManager`` (which calls a ``compute_score`` function on
    decoded text), this manager consumes the reward that the env already
    produced during rollout (``non_tensor_batch['episode_rewards']``). This is
    the right fit for SkillRL's env-driven tasks (ALFWorld/WebShop/Search),
    where reward is an env outcome (win/lose, EM), not a function of the
    response text.
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Callable[..., Any] | None = None,
        reward_fn_key: str = "data_source",
        normalize_by_length: bool = False,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score  # unused; env provides the reward
        self.reward_fn_key = reward_fn_key
        self.normalize_by_length = normalize_by_length

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # Honor pre-computed rm_scores if present (e.g. model RM path).
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources: dict[str, int] = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            # decode for logging only
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")

            # ★ SkillRL: reward comes from the env, written by gather_rollout_data.
            episode_rewards = float(data_item.non_tensor_batch["episode_rewards"])
            episode_lengths = float(data_item.non_tensor_batch.get("episode_lengths", 1.0))
            score = episode_rewards / episode_lengths if (self.normalize_by_length and episode_lengths > 0) else episode_rewards

            reward_tensor[i, valid_response_length - 1] = torch.tensor(
                score, dtype=torch.float32, device=prompt_ids.device
            )
            reward_extra_info["episode_reward"].append(episode_rewards)

            # collect per-task success_rate for skill-evolution bookkeeping
            for k in data_item.non_tensor_batch.keys():
                if "success_rate" in k:
                    reward_extra_info[k].append(float(data_item.non_tensor_batch[k]))

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
