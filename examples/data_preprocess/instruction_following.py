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

"""Preprocess instruction following datasets to verl parquet format.

Supports multiple dataset formats:
1. IF_multi_constraints_upto5 format:
   - messages: List of chat messages (OpenAI format)
   - ground_truth: Constraint information (JSON string)

2. Nemotron-Cascade-RL-Instruction-Following format:
   - prompt: List of chat messages (OpenAI format) or plain text
   - instruction_id_list: Array of instruction IDs
   - kwargs: Array of kwargs dicts for each instruction

Usage:
    # From HuggingFace dataset
    python examples/data_preprocess/instruction_following.py \
        --dataset_path allenai/IF_multi_constraints_upto5 \
        --local_save_dir ~/data/instruction_following

    # From local parquet (Nemotron format)
    python examples/data_preprocess/instruction_following.py \
        --dataset_path /path/to/ifrl_final_release.parquet \
        --local_save_dir ~/data/instruction_following

    # From local parquet with explicit format
    python examples/data_preprocess/instruction_following.py \
        --dataset_path /path/to/data.parquet \
        --dataset_format nemotron \
        --local_save_dir ~/data/instruction_following
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

DATA_SOURCE = "instruction_following"


def build_ground_truth(sample, dataset_format):
    """Build ground_truth JSON string from sample fields."""
    if dataset_format == "nemotron":
        instruction_ids = list(sample["instruction_id_list"])
        kwargs_list = list(sample["kwargs"])
        return json.dumps([{"instruction_id": instruction_ids, "kwargs": kwargs_list}])
    elif dataset_format == "if_multi":
        return sample["ground_truth"]
    else:
        raise ValueError(f"Unknown dataset_format: {dataset_format}")


def build_prompt(sample, dataset_format):
    """Build prompt in OpenAI chat format from sample fields."""
    if dataset_format == "if_multi":
        if "messages" in sample:
            return sample["messages"]
        elif "prompt" in sample:
            prompt_value = sample["prompt"]
            if isinstance(prompt_value, list):
                return prompt_value
            elif isinstance(prompt_value, str):
                return [{"role": "user", "content": prompt_value}]
        elif "question" in sample:
            return [{"role": "user", "content": sample["question"]}]
    elif dataset_format == "nemotron":
        if "prompt" in sample:
            prompt_value = sample["prompt"]
            if isinstance(prompt_value, list):
                return prompt_value
            elif isinstance(prompt_value, str):
                return [{"role": "user", "content": prompt_value}]
        if "messages" in sample:
            return sample["messages"]

    raise ValueError(
        f"Cannot build prompt for format={dataset_format}. "
        f"Available fields: {list(sample.keys())}"
    )


def detect_format(sample):
    """Auto-detect dataset format from sample fields."""
    if "instruction_id_list" in sample and "kwargs" in sample:
        return "nemotron"
    elif "ground_truth" in sample:
        return "if_multi"
    else:
        raise ValueError(
            f"Cannot auto-detect dataset format. "
            f"Available fields: {list(sample.keys())}. "
            f"Please specify --dataset_format explicitly."
        )


def make_map_fn(split, dataset_format):
    """Create a mapping function for dataset preprocessing."""

    def process_fn(example, idx):
        ground_truth = build_ground_truth(example, dataset_format)
        prompt = build_prompt(example, dataset_format)

        data = {
            "data_source": DATA_SOURCE,
            "prompt": prompt,
            "ability": "instruction_following",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset (HuggingFace name or local path).")
    parser.add_argument(
        "--dataset_format",
        default=None,
        choices=["if_multi", "nemotron", None],
        help="Dataset format. If None, auto-detect from field names.",
    )
    parser.add_argument("--dataset_split", default=None, help="Dataset split to load (e.g., 'train'). Auto-detected if None.")
    parser.add_argument("--local_save_dir", default="~/data/instruction_following", help="Local save directory.")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory for copying.")
    args = parser.parse_args()

    # Load dataset
    if args.dataset_path.endswith(".parquet"):
        dataset = datasets.load_dataset("parquet", data_files=args.dataset_path, split=args.dataset_split)
    else:
        dataset = datasets.load_dataset(args.dataset_path, split=args.dataset_split)

    # Handle DatasetDict
    if isinstance(dataset, dict):
        splits = list(dataset.keys())
        if len(splits) == 1:
            dataset = dataset[splits[0]]
        else:
            raise ValueError(f"Dataset has multiple splits {splits}. Please specify --dataset_split.")

    # Auto-detect format if not specified
    if args.dataset_format is None:
        sample = dataset[0]
        dataset_format = detect_format(sample)
        print(f"Auto-detected dataset format: {dataset_format}")
    else:
        dataset_format = args.dataset_format

    # Process dataset
    processed_dataset = dataset.map(
        function=make_map_fn("train", dataset_format),
        with_indices=True,
        remove_columns=dataset.column_names,
    )

    # Save
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    output_path = os.path.join(local_save_dir, "train.parquet")
    processed_dataset.to_parquet(output_path)
    print(f"Saved {len(processed_dataset)} samples to {output_path}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
