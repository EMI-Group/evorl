"""JSONL recording and merged metric loading contracts."""

import json

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from evorl.recorders import JSONLRecorder, load_jsonl_records
from scripts.recorder_setup import setup_recorders


@pytest.fixture
def recorder(tmp_path):
    recorder = JSONLRecorder(
        project="EvoRL",
        name="metrics",
        config={"seed": 42},
        tags=["test"],
        path=tmp_path,
        group="dev",
    )
    yield recorder
    recorder.close()


def test_jsonl_preserves_write_order_duplicates_and_flushes(tmp_path, recorder):
    path = tmp_path / "metrics.jsonl"
    recorder.init()
    try:
        recorder.write({"train": {"loss": 1.0}}, np.int64(10))
        recorder.write({"eval": {"episode_return": 20}}, jnp.asarray(8))
        recorder.write({"train": {"loss": 0.5}, "step": 99}, 10)
        header, *records = [json.loads(line) for line in path.read_text().splitlines()]
        assert header == {
            "metadata": {
                "project": "EvoRL",
                "name": "metrics",
                "config": {"seed": 42},
                "tags": ["test"],
                "group": "dev",
            }
        }
        assert [record["step"] for record in records] == [10, 8, 10]
        assert [record["data"] for record in records] == [
            {"train": {"loss": 1.0}},
            {"eval": {"episode_return": 20}},
            {"train": {"loss": 0.5}, "step": 99},
        ]
        assert len(path.read_text().splitlines()) == 4
        merged = load_jsonl_records(path)
        assert isinstance(merged, list)
        assert merged == [
            {"step": 8, "data": {"eval": {"episode_return": 20}}},
            {"step": 10, "data": {"train": {"loss": 0.5}, "step": 99}},
        ]
    finally:
        recorder.close()


def test_jsonl_retains_arrays_samples_tables_and_standard_json(tmp_path, recorder):
    path = tmp_path / "metrics.jsonl"
    frame = pd.DataFrame(
        {"pop_id": [2**60 + 1], "learning_rate": [1.23456789012345e-8]}
    )
    original = frame.copy(deep=True)
    recorder.init()
    try:
        recorder.write(
            {
                "nested": {"scalar": np.float32(0.5), "flag": np.bool_(True)},
                "array": jnp.asarray([[1, 2], [3, 4]]),
                "samples": pd.Series([1, pd.NA, 3], dtype="Int64"),
                "table": frame,
                "nonfinite": np.array([np.nan, np.inf, -np.inf]),
                "tuple": (1, np.int32(2)),
                "text": "训练\n评估",
                "missing": pd.NaT,
            },
            0,
        )
    finally:
        recorder.close()
    text = path.read_text(encoding="utf-8")
    assert "训练" in text
    assert len(text.splitlines()) == 2
    record = json.loads(
        text.splitlines()[1], parse_constant=lambda value: pytest.fail(value)
    )
    assert record["data"] == {
        "nested": {"scalar": 0.5, "flag": True},
        "array": [[1, 2], [3, 4]],
        "samples": [1, None, 3],
        "table": {
            "columns": ["pop_id", "learning_rate"],
            "data": [[2**60 + 1, 1.23456789012345e-8]],
        },
        "nonfinite": [None, None, None],
        "tuple": [1, 2],
        "text": "训练\n评估",
        "missing": None,
    }
    pd.testing.assert_frame_equal(frame, original)


def test_jsonl_lifecycle_and_failed_serialization(tmp_path, recorder):
    path = tmp_path / "metrics.jsonl"
    path.write_text("old content\n")
    recorder.close()
    with pytest.raises(RuntimeError, match="Initialize"):
        recorder.write({}, 0)
    recorder.init()
    try:
        assert "old content" not in path.read_text()
        assert load_jsonl_records(path) == []
        recorder.write({"loss": 1}, 0)
        with pytest.raises(RuntimeError, match="already initialized"):
            recorder.init()
        with pytest.raises(TypeError):
            recorder.write({"unsupported": object()}, 1)
        assert load_jsonl_records(path) == [{"step": 0, "data": {"loss": 1}}]
    finally:
        recorder.close()
    recorder.close()
    with pytest.raises(RuntimeError, match="Initialize"):
        recorder.write({}, 1)


@pytest.mark.parametrize(
    "line",
    [
        '{"step":',
        "[]",
        '{"step": 1}',
        '{"step": true, "data": {}}',
        '{"step": 1.5, "data": {}}',
        '{"data": {}}',
    ],
)
def test_load_jsonl_records_reports_invalid_record_location(tmp_path, line):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 0, "data": {}}\n' + line + "\n")
    with pytest.raises(ValueError, match=r"metrics\.jsonl:2:"):
        load_jsonl_records(path)


def test_load_jsonl_records_merges_keys_and_replaces_values(tmp_path, recorder):
    path = tmp_path / "metrics.jsonl"
    recorder.init()
    try:
        recorder.write(
            {
                "train": {"loss": 3, "entropy": 0.5},
                "eval": {"return": 10},
                "array": [1, 2, 3],
                "nullable": 1,
                "to_scalar": {"old": 1},
                "to_dict": 1,
            },
            10,
        )
        recorder.write({"train": {"loss": 8}}, 8)
        recorder.write(
            {
                "train": {"loss": 2},
                "array": [4],
                "nullable": None,
                "to_scalar": 2,
                "to_dict": {"new": 2},
                "step": 999,
            },
            10,
        )
        recorder.write({"train": {"loss": 4}}, 8)
    finally:
        recorder.close()
    assert load_jsonl_records(path) == [
        {"step": 8, "data": {"train": {"loss": 4}}},
        {
            "step": 10,
            "data": {
                "train": {"loss": 2, "entropy": 0.5},
                "eval": {"return": 10},
                "array": [4],
                "nullable": None,
                "to_scalar": 2,
                "to_dict": {"new": 2},
                "step": 999,
            },
        },
    ]


def test_load_jsonl_records_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.touch()
    assert load_jsonl_records(path) == []


@pytest.mark.parametrize(
    "env_name, env_type, tags, filename",
    [
        ("ant", "brax", [], "PPO_ant_brax"),
        ("MetaWorld/Hammer-v3", "envpool", [], "PPO_MetaWorld_Hammer-v3_envpool"),
        ("MetaWorld%2FHammer-v3", "envpool", [], "PPO_MetaWorld%2FHammer-v3_envpool"),
        ("walker2d", "brax", ["evoresearch"], "PPO_walker2d_brax|evoresearch"),
        ("ant", "brax", ["中文"], "PPO_ant_brax|中文"),
    ],
)
def test_jsonl_and_log_recorders_keep_separate_files(
    tmp_path, env_name, env_type, tags, filename
):
    config = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "project": "EvoRL",
            "tags": tags,
            "recorders": ["log", "jsonl"],
            "env": {"env_name": env_name, "env_type": env_type},
        }
    )
    log, jsonl = setup_recorders(config, "PPO")
    assert log.log_path == tmp_path / f"{filename}.log"
    assert jsonl.log_path == tmp_path / f"{filename}.jsonl"
    log.init()
    jsonl.init()
    try:
        log.write({"log_only": 1}, 8)
        jsonl.write({"jsonl_only": 2}, 8)
        with jsonl.log_path.open() as stream:
            name = f"PPO_{env_name}_{env_type}"
            if tags:
                name += "|" + ",".join(tags)
            assert json.loads(next(stream))["metadata"]["name"] == name
    finally:
        log.close()
        jsonl.close()
    assert "log_only" in log.log_path.read_text()
    assert "jsonl_only" not in log.log_path.read_text()
    assert load_jsonl_records(jsonl.log_path) == [
        {"step": 8, "data": {"jsonl_only": 2}}
    ]


def test_setup_selects_jsonl_with_the_script_run_name(tmp_path):
    config = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "project": "EvoRL",
            "tags": ["baseline"],
            "recorders": ["jsonl"],
            "env": {"env_name": "ant", "env_type": "brax"},
        }
    )
    (recorder,) = setup_recorders(config, "PPO", parallel=True)
    assert isinstance(recorder, JSONLRecorder)
    assert recorder.log_path == tmp_path / "PPO_ant_brax|baseline.jsonl"
    recorder.init()
    try:
        assert load_jsonl_records(recorder.log_path) == []
        metadata = json.loads(recorder.log_path.read_text())["metadata"]
        assert metadata["name"] == "PPO_ant_brax|baseline"
        assert metadata["group"] == metadata["name"]
        assert metadata["tags"] == ["PPO", "ant", "brax", "baseline"]
        assert metadata["config"] == OmegaConf.to_container(config, resolve=True)
    finally:
        recorder.close()

    (custom,) = setup_recorders(
        config,
        "PPO",
        recorder_kwargs={
            "jsonl": {"path": tmp_path / "custom", "name": "ablation", "group": None}
        },
    )
    custom.init()
    try:
        assert custom.log_path == tmp_path / "custom" / "ablation.jsonl"
        metadata = json.loads(custom.log_path.read_text())["metadata"]
        assert metadata["name"] == "ablation"
        assert metadata["group"] is None
    finally:
        custom.close()


@pytest.mark.parametrize(
    "lines, line_number",
    [
        ('{"metadata": []}\n', 1),
        ('{"metadata": {}}\n{"metadata": {}}\n', 2),
    ],
)
def test_load_jsonl_records_validates_metadata_header(tmp_path, lines, line_number):
    path = tmp_path / "metrics.jsonl"
    path.write_text(lines)
    with pytest.raises(ValueError, match=rf"metrics\.jsonl:{line_number}:"):
        load_jsonl_records(path)
