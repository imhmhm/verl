"""Tests for instruction following reward function."""

import json

import pytest

from verl.utils.reward_score.instruction_following_reward import (
    _parse_ground_truth,
    _remove_thinking_section,
    compute_score,
)


class TestParseGroundTruth:
    """Test cases for _parse_ground_truth helper."""

    def test_parse_json_string(self):
        """Test parsing standard JSON format."""
        result = _parse_ground_truth('[{"instruction_id": ["test"], "kwargs": []}]')
        assert result["instruction_id"] == ["test"]

    def test_parse_python_dict_string(self):
        """Test parsing Python dict string (ast.literal_eval)."""
        result = _parse_ground_truth("{'instruction_id': ['test'], 'kwargs': []}")
        assert result["instruction_id"] == ["test"]

    def test_parse_nested_string(self):
        """Test parsing nested/double-encoded JSON string."""
        original = {"instruction_id": ["test"]}
        first_encode = json.dumps(original)
        second_encode = json.dumps(first_encode)
        result = _parse_ground_truth(second_encode)
        assert result["instruction_id"] == ["test"]

    def test_parse_list_format(self):
        """Test parsing list format, taking first element."""
        result = _parse_ground_truth('[{"instruction_id": ["test"]}, {"instruction_id": ["other"]}]')
        assert result["instruction_id"] == ["test"]

    def test_parse_empty_list(self):
        """Test parsing empty list."""
        result = _parse_ground_truth("[]")
        assert result["instruction_id"] == []
        assert result["kwargs"] == []

    def test_parse_nemotron_format(self):
        """Test parsing Nemotron-format ground_truth (JSON with null values)."""
        gt = json.dumps([{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"], "num_words": None}]}])
        result = _parse_ground_truth(gt)
        assert result["instruction_id"] == ["keywords:existence"]
        assert len(result["kwargs"]) == 1
        assert result["kwargs"][0]["keywords"] == ["AI"]


class TestRemoveThinkingSection:
    """Test cases for _remove_thinking_section helper."""

    def test_no_thinking(self):
        """Test with plain text (no thinking section)."""
        result = _remove_thinking_section("Hello world")
        assert result == "Hello world"

    def test_thinking_tags(self):
        """Test removing thinking section tags."""
        result = _remove_thinking_section("<think>Let me analyze.</think> The answer is 42")
        assert result == "The answer is 42"

    def test_answer_tags(self):
        """Test removing <answer> tags."""
        result = _remove_thinking_section("<answer>The answer is 42</answer>")
        assert result == "The answer is 42"


class TestComputeScore:
    """Test cases for compute_score function."""

    def test_no_ground_truth(self):
        """Test with no ground truth."""
        assert compute_score("instruction_following", "hello", None) == 0.0

    def test_empty_solution(self):
        """Test with empty solution."""
        gt = json.dumps([{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"]}]}])
        assert compute_score("instruction_following", "", gt) == 0.0

    def test_keywords_existence_correct(self):
        """Test keywords:existence constraint with correct answer."""
        gt = json.dumps([{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"]}]}])
        score = compute_score("instruction_following", "AI is transforming the world.", gt)
        assert score == 1.0

    def test_keywords_existence_incorrect(self):
        """Test keywords:existence constraint with incorrect answer."""
        gt = json.dumps([{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["quantum"]}]}])
        score = compute_score("instruction_following", "AI is transforming the world.", gt)
        assert score == 0.0

    def test_multiple_constraints(self):
        """Test with multiple constraints."""
        gt = json.dumps([
            {"instruction_id": ["keywords:existence", "keywords:frequency"],
             "kwargs": [{"keywords": ["AI"]}, {"keyword": "AI", "frequency": 1, "relation": "at least"}]}
        ])
        score = compute_score("instruction_following", "AI is transforming the AI world.", gt)
        assert score == 1.0

    def test_nemotron_format_ground_truth(self):
        """Test with Nemotron-format ground_truth (JSON with null values)."""
        gt = json.dumps([
            {"instruction_id": ["keywords:existence"],
             "kwargs": [{"keywords": ["test"], "num_words": None, "relation": None}]}
        ])
        score = compute_score("instruction_following", "This is a test response.", gt)
        assert score == 1.0

    def test_unknown_instruction(self):
        """Test with unknown instruction key."""
        gt = json.dumps([{"instruction_id": ["nonexistent:instruction"], "kwargs": [None]}])
        score = compute_score("instruction_following", "hello", gt)
        assert score == 0.0

    def test_title_constraint(self):
        """Test detectable_format:title constraint."""
        gt = json.dumps([{"instruction_id": ["detectable_format:title"], "kwargs": [None]}])
        score = compute_score("instruction_following", "<<My Title>>\nThis is the content.", gt)
        assert score == 1.0

    def test_title_constraint_missing(self):
        """Test detectable_format:title constraint when title is missing."""
        gt = json.dumps([{"instruction_id": ["detectable_format:title"], "kwargs": [None]}])
        score = compute_score("instruction_following", "This has no title.", gt)
        assert score == 0.0
