"""Config rules: E1 and E2 differ only by the age input (CLAUDE.md; decision 33), seed 42."""
import yaml

from conftest import ROOT

ALLOWED = {("experiment",), ("paths", "out_dir"), ("inputs", "age")}


def flatten(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def load(name):
    return dict(flatten(yaml.safe_load(open(ROOT / "configs" / name))))


def test_e1_e2_differ_only_by_age():
    e1, e2 = load("e1.yaml"), load("e2.yaml")
    assert e1.keys() == e2.keys()
    assert {k for k in e1 if e1[k] != e2[k]} == ALLOWED
    assert e1[("inputs", "age")] is False and e2[("inputs", "age")] is True


def test_no_bag_or_clinical_in_fyp1_experiments():
    for name in ("e1.yaml", "e2.yaml"):
        cfg = load(name)
        assert cfg[("inputs", "bag")] is False and cfg[("inputs", "clinical")] is False


def test_seed_42_everywhere():
    for path in (ROOT / "configs").glob("*.yaml"):
        if path.name == "preprocessing.yaml":
            # The only random step (registration metric sampling) fixes its seed in the code
            code = (ROOT / "src" / "preprocessing" / "preprocess.py").read_text()
            assert "SetMetricSamplingPercentage(0.25, 42)" in code
            continue
        assert yaml.safe_load(open(path)).get("seed") == 42, path.name


def test_frozen_models_have_hashes():
    cfg = yaml.safe_load(open(ROOT / "configs" / "progression_test.yaml"))
    for exp, models in cfg["frozen"].items():
        for model, digest in models.items():
            assert isinstance(digest, str) and len(digest) == 64, f"{exp}/{model}"
