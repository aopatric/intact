import re

import pytest

from intact import config

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
    monkeypatch.setenv("INTACT_ARTIFACTS", str(tmp_path))
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
    assert e.layers == ("34", "36pre")  # canonical keys (DESIGN §6 hooks)
    assert e.max_gb > 0 and e.stride >= 1


def test_every_dataset_file_pins_a_sha256(cfg):
    assert set(cfg.data.splits) == {"train", "test"}
    for f in cfg.data.splits.values():
        assert re.fullmatch(r"[0-9a-f]{64}", f.sha256), f


def test_clone_resolves_under_checkout_else_artifacts_root_and_env_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv("INTACT_UPSTREAM", raising=False)
    monkeypatch.setenv("INTACT_ARTIFACTS", str(tmp_path / "artifacts"))
    in_checkout = config.load()
    assert in_checkout.upstream_clone == config.REPO_ROOT / "third_party" / "rl-rewardhacking"
    assert in_checkout.data.splits["test"].path.parent == in_checkout.upstream_clone / "results" / "data"
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path / "site-packages")  # a wheel install: no checkout
    assert config.load().upstream_clone == tmp_path / "artifacts" / "third_party" / "rl-rewardhacking"
    monkeypatch.setenv("INTACT_UPSTREAM", str(tmp_path / "mine"))
    assert config.load().upstream_clone == tmp_path / "mine"


def test_missing_clone_says_how_to_get_it(monkeypatch, tmp_path):
    monkeypatch.setenv("INTACT_UPSTREAM", str(tmp_path / "nowhere"))
    cfg = config.load()
    with pytest.raises(FileNotFoundError, match=r"git clone .* && git -C .* checkout [0-9a-f]{40}"):
        config.upstream(cfg)


def test_clone_at_another_commit_is_refused(cfg):
    import dataclasses

    if not (cfg.upstream_clone / "src").is_dir():
        pytest.skip("upstream clone missing")
    assert config.upstream(cfg) == cfg.upstream_clone
    with pytest.raises(ValueError, match="not the pinned"):
        config.upstream(dataclasses.replace(cfg, upstream_commit="0" * 40))
