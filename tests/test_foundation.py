"""Sanity checks for committed templates and the dependency-free episode contract."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from parkour_lab.contracts import EpisodeSummary

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_parseable_and_starts_small() -> None:
    config = tomllib.loads((ROOT / "configs/default.toml").read_text(encoding="utf-8"))

    assert config["simulation"]["backend"] == "mujoco_cpu"
    assert config["simulation"]["headless"] is True
    assert 0 < config["simulation"]["num_worlds"] <= 128
    assert config["training"]["policy_initialization"] == "random"


def test_example_level_matches_starter_contract() -> None:
    level = json.loads((ROOT / "configs/levels/level-01-basics.json").read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "configs/levels/level.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(level)) == []

    assert level["schema_version"] == 1
    assert level["id"].startswith("level-01-")
    assert level["units"]["up_axis"] == "+z"
    assert level["units"]["forward_axis"] == "+x"
    assert level["obstacles"]
    assert len(level["spawn"]["position"]) == 3
    assert len(level["finish"]["position"]) == 3
    assert (ROOT / "configs/levels/level.schema.json").is_file()


def test_episode_summary_rejects_invalid_progress() -> None:
    with pytest.raises(ValueError, match="completion_fraction"):
        EpisodeSummary(
            run_id="run-001",
            level_id="level-01-basics",
            seed=11,
            success=False,
            completion_fraction=1.1,
            episode_return=0.0,
            sim_steps=1,
            termination_reason="fall",
        )


def test_level_schema_requires_motion_for_moving_platform() -> None:
    schema = json.loads((ROOT / "configs/levels/level.schema.json").read_text(encoding="utf-8"))
    level = json.loads((ROOT / "configs/levels/level-01-basics.json").read_text(encoding="utf-8"))
    level["obstacles"][0] = {
        "id": "mover",
        "kind": "moving_platform",
        "position": [1.0, 0.0, 0.0],
        "size": [1.0, 1.0, 0.2],
    }

    errors = list(Draft202012Validator(schema).iter_errors(level))
    assert any("motion" in error.message for error in errors)


def test_episode_summary_accepts_measured_record() -> None:
    summary = EpisodeSummary(
        run_id="run-001",
        level_id="level-01-basics",
        seed=11,
        success=True,
        completion_fraction=1.0,
        episode_return=2.5,
        sim_steps=1200,
        termination_reason="course_complete",
        metrics={"falls": 0.0, "peak_contact_force": 14.2},
    )

    assert summary.success is True
    assert summary.metrics["falls"] == 0.0
