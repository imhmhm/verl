# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Instruction following reward function for verl.

This module provides rule-based reward computation for instruction following tasks,
using the IFEvalG instruction checkers from Google Research.
"""

import ast
import json
import logging

logger = logging.getLogger("InstructionFollowingReward")

# Tags for thinking section removal
_THINKING_START_TAG = " hintText"
_THINKING_END_TAG = "hre_result_end"
_ANSWER_START_TAG = "<answer>"
_ANSWER_END_TAG = "</answer>"


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Compute reward for instruction following tasks.

    Args:
        data_source: Dataset identifier (e.g., "instruction_following").
        solution_str: Model's generated response text.
        ground_truth: JSON string containing instruction constraints.
            Format: '[{"instruction_id": [...], "kwargs": [...]}]'
        extra_info: Optional additional information.

    Returns:
        Reward value (0.0 to 1.0), computed as average across all constraints.
    """
    if ground_truth is None or ground_truth == "":
        logger.warning("No ground_truth provided for instruction following reward")
        return 0.0

    if not solution_str:
        logger.warning("Empty solution_str received for instruction following reward")
        return 0.0

    try:
        constraint_dict = _parse_ground_truth(ground_truth)
        answer = _remove_thinking_section(solution_str)

        if not answer:
            logger.warning("Empty answer after removing thinking section")
            return 0.0

        from verl.utils.reward_score.instruction_following import INSTRUCTION_DICT

        instruction_keys = constraint_dict.get("instruction_id", [])
        args_list = constraint_dict.get("kwargs", [])

        if not instruction_keys:
            logger.warning("Empty instruction_id list in ground_truth")
            return 0.0

        rewards = []
        for instruction_key, args in zip(instruction_keys, args_list):
            if args is None:
                args = {}
            args = {k: v for k, v in args.items() if v is not None}

            if instruction_key not in INSTRUCTION_DICT:
                logger.warning("Unknown instruction: %s", instruction_key)
                rewards.append(0.0)
                continue

            try:
                instruction_cls = INSTRUCTION_DICT[instruction_key]
                instruction_instance = instruction_cls(instruction_key)
                instruction_instance.build_description(**args)

                if instruction_instance.check_following(answer):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception as e:
                logger.warning("Error checking instruction %s: %s", instruction_key, e, exc_info=True)
                rewards.append(0.0)

        return sum(rewards) / max(len(rewards), 1)

    except Exception:
        logger.warning("Exception in instruction_following compute_score", exc_info=True)
        return 0.0


def _parse_ground_truth(ground_truth: str) -> dict:
    """Parse ground truth constraints from string format.

    Handles multiple formats:
    1. JSON strings with 'null' values (e.g., from Nemotron format)
    2. Python dict strings with 'None' values (e.g., from IF_multi format)
    3. Nested/double-encoded JSON strings
    """
    constraint_dict = None

    # Try json.loads first (handles 'null' values from JSON)
    try:
        constraint_dict = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback to ast.literal_eval (handles Python dict strings with 'None')
    if constraint_dict is None:
        try:
            constraint_dict = ast.literal_eval(ground_truth)
        except (ValueError, SyntaxError):
            pass

    # Handle nested JSON string (double-encoded case)
    if isinstance(constraint_dict, str):
        try:
            constraint_dict = json.loads(constraint_dict)
        except (json.JSONDecodeError, TypeError):
            pass

    # Handle list format: take first element
    if isinstance(constraint_dict, list):
        if len(constraint_dict) == 0:
            return {"instruction_id": [], "kwargs": []}
        constraint_dict = constraint_dict[0]

    if constraint_dict is None:
        logger.warning("Failed to parse ground_truth: %s", ground_truth[:100])
        return {"instruction_id": [], "kwargs": []}

    return constraint_dict


def _remove_thinking_section(prediction: str) -> str:
    """Remove thinking section and answer tags from prediction.

    Handles the common format where models output:
     hintText...analysis...hre_result_end
    <answer>...final answer...</answer>
    """
    # Remove hre_result_end and split
    prediction = prediction.replace(_THINKING_END_TAG, "").strip()

    # Remove thinking section (everything before last hintText)
    if _THINKING_START_TAG in prediction:
        parts = prediction.split(_THINKING_START_TAG)
        if len(parts) > 1:
            prediction = parts[-1]

    # Remove answer tags
    prediction = prediction.replace(_ANSWER_START_TAG, "").replace(_ANSWER_END_TAG, "")

    return prediction.strip()
