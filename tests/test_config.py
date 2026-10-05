import re

import pytest

from testbed import config

SHA = re.compile(r"[0-9a-f]{40}")


@pytest.fixture(scope="module")
def cfg():
    return config.load()


def test_every_model_pins_a_revision(cfg):
    for m in cfg.models.values():
        assert SHA.fullmatch(m.revision), m


def test_adapters_name_a_registered_base_and_prompt_set(cfg):
    for m in cfg.models.values():
        if m.base is None:
            continue
        assert m.base in cfg.models and cfg.models[m.base].base is None, m
        assert m.prompt_set in {"hint", "nohint"}, m


def test_artifacts_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TESTBED_ARTIFACTS", str(tmp_path))
    assert config.load().artifacts_root == tmp_path


def test_sampling_configs_fully_specified(cfg):
    assert {"train-cfg", "eval-cfg"} <= cfg.sampling.keys()
    for s in cfg.sampling.values():
        assert not s.missing(), (s.name, s.missing())


def test_upstream_pins_recorded(cfg):
    assert cfg.upstream_commit and SHA.fullmatch(cfg.upstream_commit)
    assert cfg.grader["timeout_s"] is not None


def test_extract_defaults_are_valid(cfg):
    e = cfg.extract
    assert e.layers and all(0 <= layer <= 36 for layer in e.layers)  # HF hidden_states indices (DESIGN §6 hooks)
    assert len(set(e.layers)) == len(e.layers)
    assert e.max_gb > 0 and e.stride >= 1


def test_every_dataset_file_pins_a_sha256(cfg):
    assert set(cfg.data.splits) == {"train", "test"}
    for f in cfg.data.splits.values():
        assert re.fullmatch(r"[0-9a-f]{64}", f.sha256), f
