"""tests/test_config.py

Phase 1 tests: the configuration system only (model.config module and the
three shipped YAML files). Tokenizer/model/training tests land in their
own phases as those components are implemented.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model.config import (
    MODEL_PRESETS,
    ExperimentConfig,
    ModelConfig,
    get_model_preset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# ModelConfig validation
# --------------------------------------------------------------------------- #


def test_valid_model_config_constructs():
    cfg = ModelConfig(
        name="test", vocab_size=100, d_model=64, n_layers=2,
        n_query_heads=4, n_kv_heads=2, head_dim=16, d_ffn=128,
    )
    assert cfg.d_model == 64
    assert cfg.group_size == 2


def test_rejects_mismatched_query_heads_and_d_model():
    with pytest.raises(ValueError, match="n_query_heads \\* head_dim must equal d_model"):
        ModelConfig(d_model=640, n_query_heads=10, head_dim=63)  # 10*63 != 640


def test_rejects_non_divisible_gqa_groups():
    with pytest.raises(ValueError, match="divisible by n_kv_heads"):
        ModelConfig(d_model=192, n_query_heads=3, n_kv_heads=2, head_dim=64)


def test_rejects_more_kv_heads_than_query_heads():
    with pytest.raises(ValueError, match="cannot exceed"):
        ModelConfig(d_model=128, n_query_heads=2, n_kv_heads=4, head_dim=64)


# --------------------------------------------------------------------------- #
# Analytic parameter counts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "preset_name,expected_approx_millions,tolerance_millions",
    [
        ("7m", 7.15, 0.1),
        ("19m", 18.88, 0.1),
        ("56m", 55.96, 0.1),
    ],
)
def test_preset_param_counts_match_target(preset_name, expected_approx_millions, tolerance_millions):
    cfg = get_model_preset(preset_name)
    total = cfg.analytic_param_count()["total_params"]
    millions = total / 1e6
    assert abs(millions - expected_approx_millions) < tolerance_millions, (
        f"{preset_name} preset: expected ~{expected_approx_millions}M, got {millions:.2f}M"
    )


def test_56m_preset_matches_exact_spec_hyperparameters():
    cfg = get_model_preset("56m")
    assert cfg.d_model == 640
    assert cfg.n_layers == 10
    assert cfg.n_query_heads == 10
    assert cfg.n_kv_heads == 5
    assert cfg.head_dim == 64
    assert cfg.d_ffn == 1728
    assert cfg.vocab_size == 16384
    assert cfg.context_length == 512


def test_all_presets_have_valid_gqa_shapes():
    for name, cfg in MODEL_PRESETS.items():
        assert cfg.n_query_heads * cfg.head_dim == cfg.d_model, name
        assert cfg.n_query_heads % cfg.n_kv_heads == 0, name


def test_untied_embeddings_add_a_second_embedding_table():
    tied = ModelConfig(
        name="t", vocab_size=1000, d_model=64, n_layers=1,
        n_query_heads=4, n_kv_heads=4, head_dim=16, d_ffn=128,
        tie_embeddings=True,
    )
    untied = ModelConfig(
        name="u", vocab_size=1000, d_model=64, n_layers=1,
        n_query_heads=4, n_kv_heads=4, head_dim=16, d_ffn=128,
        tie_embeddings=False,
    )
    diff = untied.analytic_param_count()["total_params"] - tied.analytic_param_count()["total_params"]
    assert diff == 1000 * 64  # exactly one extra embedding table


# --------------------------------------------------------------------------- #
# YAML config loading
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["7m", "19m", "56m"])
def test_yaml_config_loads_and_matches_preset(name):
    path = REPO_ROOT / "configs" / f"{name}.yaml"
    assert path.exists(), f"missing {path}"
    cfg = ExperimentConfig.from_yaml(path)
    preset = get_model_preset(name)
    assert cfg.model.d_model == preset.d_model
    assert cfg.model.n_layers == preset.n_layers
    assert cfg.model.vocab_size == preset.vocab_size


@pytest.mark.parametrize("name", ["7m", "19m", "56m"])
def test_yaml_config_sequence_length_within_context_length(name):
    cfg = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / f"{name}.yaml")
    assert cfg.training.sequence_length <= cfg.model.context_length


def test_yaml_config_rejects_unknown_keys(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("model:\n  d_model: 64\n  not_a_real_field: 123\n")
    with pytest.raises(ValueError, match="Unknown key"):
        ExperimentConfig.from_yaml(bad_yaml)


def test_missing_config_file_raises_helpful_error(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        ExperimentConfig.from_yaml(missing)


def test_experiment_config_round_trips_to_json(tmp_path):
    cfg = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / "7m.yaml")
    out = tmp_path / "cfg.json"
    cfg.save_json(out)
    assert out.exists()
    assert '"d_model": 256' in out.read_text()
