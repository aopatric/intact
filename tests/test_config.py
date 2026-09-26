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


@pytest.mark.xfail(strict=True, reason="T1 records top_k/min_p and eval-cfg from upstream")
def test_sampling_configs_fully_specified(cfg):
    assert {"train-cfg", "eval-cfg"} <= cfg.sampling.keys()
    for s in cfg.sampling.values():
        assert not s.missing(), (s.name, s.missing())


@pytest.mark.xfail(strict=True, reason="T1 pins the upstream commit and grader timeout")
def test_upstream_pins_recorded(cfg):
    assert cfg.upstream_commit and SHA.fullmatch(cfg.upstream_commit)
    assert cfg.grader["timeout_s"] is not None
