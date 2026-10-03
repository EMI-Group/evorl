"""Contract checks for recorder setup and backend data conversion."""

import json
import logging
import subprocess
import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from evorl.recorders import (
    AimRecorder,
    ChainRecorder,
    CometRecorder,
    LogRecorder,
    NeptuneRecorder,
    SwanlabRecorder,
    WandbRecorder,
)
from evorl.recorders.recorder_utils import histogram_values
from omegaconf import OmegaConf
from scripts import recorder_setup


def _sdk(name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def test_setup_accepts_script_options_and_shared_aim_repo(monkeypatch):
    config = OmegaConf.create(
        {
            "output_dir": "/tmp/evorl-run",
            "project": "EvoRL",
            "tags": ["baseline"],
            "recorders": ["aim", "wandb"],
            "env": {"env_name": "ant", "env_type": "brax"},
        }
    )
    monkeypatch.setattr(recorder_setup, "get_original_cwd", lambda: "/tmp/evorl")

    aim, wandb = recorder_setup.setup_recorders(config, "PPO")
    assert aim.group == "dev"
    assert str(aim.path) == "/tmp/evorl/aim"
    assert aim.tags == ["PPO", "ant", "brax", "baseline"]
    assert wandb.wandb_kwargs["group"] == "dev"

    aim, wandb = recorder_setup.setup_recorders(config, "PPO", parallel=True)
    assert aim.group == "PPO_ant_brax|baseline"
    assert wandb.wandb_kwargs["group"] == aim.group

    aim, wandb = recorder_setup.setup_recorders(
        config,
        "PPO",
        recorder_kwargs={
            "aim": {"group": None, "path": "/shared/aim", "experiment": "custom"}
        },
    )
    assert aim.group is None
    assert str(aim.path) == "/shared/aim"
    assert aim.extra_kwargs["experiment"] == "custom"
    assert wandb.wandb_kwargs["group"] == "dev"


def test_setup_rejects_misspelled_backend_options():
    config = OmegaConf.create(
        {
            "output_dir": "/tmp/evorl-run",
            "project": "EvoRL",
            "tags": [],
            "recorders": ["wandb"],
            "env": {"env_name": "ant", "env_type": "brax"},
        }
    )
    with pytest.raises(ValueError, match="unknown keys: wanbd"):
        recorder_setup.setup_recorders(
            config, "PPO", recorder_kwargs={"wanbd": {"group": "ablation"}}
        )


def test_setup_import_does_not_load_jax_or_tracking_sdks():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import scripts.recorder_setup; "
                "assert not set(['evorl', 'jax', 'aim', 'wandb', 'swanlab', "
                "'comet_ml', 'neptune_scale']).intersection(sys.modules)"
            ),
        ],
        check=True,
    )


def test_aim_tracks_nested_metrics_distribution_table_and_raw_array(monkeypatch):
    run = MagicMock()
    distribution = MagicMock(side_effect=lambda **kwargs: ("distribution", kwargs))
    text = MagicMock(side_effect=lambda value: ("text", value))
    monkeypatch.setitem(
        sys.modules,
        "aim",
        _sdk(
            "aim", Run=MagicMock(return_value=run), Distribution=distribution, Text=text
        ),
    )
    recorder = AimRecorder(
        project="EvoRL",
        name="PPO_ant",
        config={"seed": 42},
        tags=["PPO"],
        path="/tmp/aim",
        group="dev",
    )
    recorder.init()
    recorder.write(
        {
            "train": {
                "loss": np.float32(0.5),
                "population": pd.Series([1.0, 2.0]),
                "results": pd.DataFrame({"score": [1, 2]}),
                "raw": np.array([1, 2]),
            }
        },
        step=7,
    )
    assert run.name == "PPO_ant"
    run.track.assert_any_call(0.5, name="train/loss", step=7)
    assert distribution.called
    np.testing.assert_array_equal(distribution.call_args.kwargs["samples"], [1.0, 2.0])
    assert any(
        call.kwargs.get("name") == "train/results" for call in run.track.call_args_list
    )
    text.assert_any_call("[1, 2]")
    assert any('"schema"' in call.args[0] for call in text.call_args_list)
    recorder.close()
    run.close.assert_called_once()


def test_other_backends_keep_group_and_histogram_semantics(monkeypatch):
    swan_run = MagicMock()
    swan = _sdk(
        "swanlab",
        init=MagicMock(return_value=swan_run),
        log=MagicMock(),
        finish=MagicMock(),
        Text=MagicMock(),
        echarts=_sdk("echarts", Bar=MagicMock(), Table=MagicMock()),
    )
    monkeypatch.setitem(sys.modules, "swanlab", swan)
    recorder = SwanlabRecorder(
        project="EvoRL", name="run", config={}, tags=[], path="/tmp/run", group="dev"
    )
    recorder.init()
    assert swan.init.call_args.kwargs["group"] == "dev"
    recorder.write({"dist": pd.Series([1.0, 2.0])}, step=3)
    swan.echarts.Bar.assert_called_once()
    swan_run.log.assert_called_once()
    recorder.close()
    swan_run.finish.assert_called_once()

    experiment = MagicMock()
    comet = _sdk("comet_ml", start=MagicMock(return_value=experiment))
    monkeypatch.setitem(sys.modules, "comet_ml", comet)
    recorder = CometRecorder(
        project="EvoRL", name="run", config={}, tags=[], path="/tmp/run", group="dev"
    )
    recorder.init()
    assert comet.start.call_args.kwargs["mode"] == "create"
    experiment.log_other.assert_any_call("group", "dev")
    recorder.write({"dist": pd.Series([1.0, 2.0])}, step=3)
    experiment.log_histogram_3d.assert_called_once()
    recorder.write({"table": pd.DataFrame({"value": [1]})}, step=3)
    assert experiment.log_table.call_args.kwargs["filename"] == "table.json"
    experiment.set_step.assert_called_with(3)
    assert not hasattr(recorder, "_next_step")
    assert "step" not in experiment.log_histogram_3d.call_args.kwargs
    recorder.close()

    run = MagicMock()
    neptune = _sdk("neptune_scale", Run=MagicMock(return_value=run))
    histogram = MagicMock()
    monkeypatch.setitem(sys.modules, "neptune_scale", neptune)
    monkeypatch.setitem(
        sys.modules,
        "neptune_scale.types",
        _sdk("neptune_scale.types", Histogram=histogram),
    )
    recorder = NeptuneRecorder(
        project="workspace/project",
        name="run",
        config={},
        tags=[],
        path="/tmp/run",
        group="dev",
    )
    recorder.init()
    run.add_tags.assert_any_call(tags=["dev"], group_tags=True)
    recorder.write({"dist": pd.Series([1.0, 2.0])}, step=3)
    run.log_histograms.assert_called_once()
    frame = pd.DataFrame({"pop_id": [0], "learning_rate": [1.23456789012345e-8]})
    recorder.write({"pop": frame}, step=3)
    table_call = run.log_string_series.call_args
    assert table_call.kwargs["step"] == 3
    assert json.loads(table_call.kwargs["data"]["pop"])["data"][0][
        "learning_rate"
    ] == pytest.approx(frame.learning_rate[0], rel=1e-12)
    assert "step" not in frame.columns
    with pytest.raises(TypeError, match="step"):
        recorder.write({"loss": 0.5})
    recorder.close()


def test_wandb_uses_the_run_returned_by_init(monkeypatch):
    run = MagicMock()
    wandb = _sdk(
        "wandb",
        init=MagicMock(return_value=run),
        log=MagicMock(),
        finish=MagicMock(),
        Table=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    recorder = WandbRecorder(
        project="EvoRL", name="run", config={}, tags=[], path="/tmp/run", group="dev"
    )
    recorder.init()
    assert wandb.init.call_args.kwargs["group"] == "dev"
    recorder.write({"loss": 0.5}, step=3)
    run.log.assert_called_once_with({"loss": 0.5}, step=3)
    wandb.log.assert_not_called()
    frame = pd.DataFrame({"pop_id": [0], "learning_rate": [1e-4]})
    recorder.write({"pop": frame}, step=4)
    wandb.Table.assert_called_once_with(dataframe=frame)
    run.log.assert_called_with({"pop": wandb.Table.return_value}, step=4)
    assert "step" not in frame.columns
    recorder.close()
    run.finish.assert_called_once()


def test_nullable_histogram_samples():
    series = pd.Series([1, pd.NA, 3], dtype="Int64")
    np.testing.assert_array_equal(histogram_values(series), [1, 3])
    assert histogram_values(pd.Series([np.nan, np.inf])).size == 0


def test_log_recorders_are_isolated_without_root_handlers(tmp_path, monkeypatch):
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    first_path, second_path = tmp_path / "first.log", tmp_path / "second.log"
    first = LogRecorder(first_path, console=False)
    second = LogRecorder(second_path, console=True)
    first.init()
    second.init()
    try:
        first.write({"first_loss": 1.0}, 1)
        second.write({"second_loss": 2.0}, 2)
        assert not first.logger.propagate
        assert second.logger.propagate
    finally:
        first.close()
        second.close()
    assert "second_loss" not in first_path.read_text()
    assert "first_loss" not in second_path.read_text()
    assert "first_loss" in first_path.read_text()
    assert "second_loss" in second_path.read_text()


def test_chain_closes_all_backends_after_a_failure():
    first, second = MagicMock(), MagicMock()
    first.close.side_effect = RuntimeError("flush failed")
    chain = ChainRecorder((first,))
    chain.add_recorder(second)
    with pytest.raises(RuntimeError, match="flush failed"):
        chain.close()
    second.close.assert_called_once()


@pytest.mark.parametrize(
    "recorder_type",
    [
        ChainRecorder,
        LogRecorder,
        WandbRecorder,
        AimRecorder,
        SwanlabRecorder,
        CometRecorder,
        NeptuneRecorder,
    ],
)
def test_all_recorders_require_an_integer_step(recorder_type, tmp_path):
    if recorder_type is ChainRecorder:
        recorder = recorder_type([])
    elif recorder_type is LogRecorder:
        recorder = recorder_type(tmp_path / "run.log")
    else:
        recorder = recorder_type(
            project="EvoRL", name="run", config={}, tags=[], path=tmp_path
        )

    with pytest.raises(TypeError, match="step"):
        recorder.write({})
    for invalid_step in [None, 1.5, True, "3"]:
        with pytest.raises(TypeError, match="step must be an integer"):
            recorder.write({}, invalid_step)


def test_chain_normalizes_numpy_and_jax_integer_steps():
    import jax.numpy as jnp

    backend = MagicMock()
    chain = ChainRecorder([backend])
    for step in [np.int64(3), jnp.asarray(4, dtype=jnp.uint32)]:
        chain.write({"loss": 0.5}, step)
        forwarded_step = backend.write.call_args.args[1]
        assert type(forwarded_step) is int
        assert forwarded_step == int(step)


@pytest.mark.parametrize("backend", ["wandb", "swanlab"])
def test_close_marks_a_training_exception(monkeypatch, backend):
    run = MagicMock()
    monkeypatch.setitem(
        sys.modules, backend, _sdk(backend, init=MagicMock(return_value=run))
    )
    recorder_type = WandbRecorder if backend == "wandb" else SwanlabRecorder
    recorder = recorder_type(
        project="EvoRL", name="run", config={}, tags=[], path="/tmp/run"
    )
    recorder.init()
    try:
        raise RuntimeError("training failed")
    except RuntimeError:
        recorder.close()
    if backend == "wandb":
        run.finish.assert_called_once_with(exit_code=1)
    else:
        run.finish.assert_called_once_with(state="crashed", error="training failed")


def test_real_wandb_preserves_table_history_step(tmp_path):
    wandb = pytest.importorskip("wandb")

    recorder = WandbRecorder(
        project="EvoRL",
        name="pbt",
        config={"seed": 42},
        tags=["PBT"],
        path=tmp_path,
        group="dev",
        mode="offline",
        settings=wandb.Settings(console="off", silent=True),
    )
    recorder.init()
    try:
        assert recorder.run.group == "dev"
        recorder.write({"loss": 0.5}, step=3)
        assert recorder.run.step == 3
        recorder.write(
            {
                "pop": pd.DataFrame({"pop_id": [0, 1], "learning_rate": [1e-8, 2e-4]}),
                "dist": pd.Series([1.0, 2.0]),
            },
            step=4,
        )
        assert recorder.run.step == 4
    finally:
        recorder.close()

    table_path = next(tmp_path.glob("wandb/offline-run-*/files/media/table/*.json"))
    table = json.loads(table_path.read_text())
    assert table["columns"] == ["pop_id", "learning_rate"]
    assert table["data"] == [[0, 1e-8], [1, 2e-4]]


def test_real_aim_preserves_population_snapshots(tmp_path):
    aim = pytest.importorskip("aim")
    from aim.storage.context import Context

    repo = tmp_path / "aim"
    recorder = AimRecorder(
        project="EvoRL",
        name="pbt",
        config={"seed": 42, "nested": {"value": None}},
        tags=["PBT"],
        path=repo,
        group="dev",
        system_tracking_interval=None,
        capture_terminal_logs=False,
    )
    recorder.init()
    run_hash = recorder.run.hash
    frame = pd.DataFrame({"pop_id": [0, 1], "learning_rate": [1e-8, 2e-4]})
    try:
        recorder.write({"pop": frame, "dist": pd.Series([1.0, 2.0])}, step=3)
        recorder.write({"another_metric": 2.0}, step=3)
        recorder.write({"pop": frame}, step=4)
    finally:
        recorder.close()

    # Read the run chunk directly; the UI normally indexes it in the background.
    run = aim.Run(
        run_hash,
        repo=str(repo),
        system_tracking_interval=None,
        capture_terminal_logs=False,
    )
    try:
        assert run.experiment == "dev"
        assert run["project"] == "EvoRL"
        assert run["config"] == {"seed": 42, "nested": {"value": None}}
        assert run.get_text_sequence("pop", Context({})).data.indices_list() == [3, 4]
        assert run.get_distribution_sequence(
            "dist", Context({})
        ).data.indices_list() == [3]
    finally:
        run.close()


def test_real_comet_table_assets_keep_step(tmp_path, monkeypatch):
    comet = pytest.importorskip("comet_ml")
    import comet_ml.experiment as experiment_module

    recorder = CometRecorder(
        project="EvoRL",
        name="pbt",
        config={},
        tags=[],
        path=tmp_path,
        group="dev",
        online=False,
        experiment_config=comet.ExperimentConfig(
            offline_directory=str(tmp_path),
            display_summary_level=0,
            log_code=False,
            log_graph=False,
            log_git_metadata=False,
            log_git_patch=False,
            log_env_details=False,
            auto_output_logging=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            auto_log_co2=False,
        ),
    )
    recorder.init()
    set_step = MagicMock(wraps=recorder.experiment.set_step)
    monkeypatch.setattr(recorder.experiment, "set_step", set_step)
    process = MagicMock(wraps=experiment_module.preprocess_asset_memory_file)
    monkeypatch.setattr(experiment_module, "preprocess_asset_memory_file", process)
    asset_data = MagicMock(wraps=experiment_module.AssetDataUploadProcessor)
    monkeypatch.setattr(experiment_module, "AssetDataUploadProcessor", asset_data)
    enqueue = MagicMock(wraps=recorder.experiment._enqueue_message)
    monkeypatch.setattr(recorder.experiment, "_enqueue_message", enqueue)
    frame = pd.DataFrame({"pop_id": [0], "learning_rate": [1e-4]})
    try:
        # A table-only write must associate step without a preceding metric call.
        recorder.write({"train/pop": frame}, step=3)
        recorder.write({"train/pop": frame}, step=4)
        assert [call.kwargs["step"] for call in process.call_args_list] == [3, 4]
        assert recorder.experiment.curr_step == 4
        recorder.write(
            {"loss": 0.5, "dist": pd.Series([1.0, 2.0]), "text": "hello"},
            step=5,
        )
        assert [call.args[0] for call in set_step.call_args_list] == [3, 4, 5]
        assert [
            call.kwargs["url_params"]["step"] for call in asset_data.call_args_list
        ] == [5, 5]
        metric_messages = [
            call.args[0]
            for call in enqueue.call_args_list
            if isinstance(call.args[0], experiment_module.MetricMessage)
        ]
        assert metric_messages[-1].metric["step"] == 5
    finally:
        recorder.close()
